"""PulseGate: event-driven compute that skips channels with nothing to say.

Purpose: stop paying for a channel whose input has not meaningfully changed.
This is the only component in Bhanox whose contribution is *less* work, and at
the design-phase operating point it removed 54% of compute for 0.34% error.

In simple words: if this channel was handed roughly the same input as last
time, reusing last time's answer is fine. Only wake it up when the input has
actually moved.

Architecture (spec D3, frozen)::

    per channel c:  recompute  <=>  |a_c - a_hat_c| > tau_hi,c
                    sleep      <=>  |delta a_c|     < tau_lo,c
    salience override: the top 5% of gate-magnitude channels never skip
    flush: resets every channel to awake

The gap between ``tau_lo`` and ``tau_hi`` is the whole trick. A single
threshold has to sit low enough to catch small changes (and then wakes on
noise) or high enough to ignore noise (and then misses real changes). Hysteresis
gets both: a sleeping channel needs a bigger kick to wake than a sleeping
channel needs to stay asleep.

Bytes touched per token, in the steady state: only the awake channels'
weights. In the fully saturated worst case it degrades to the ungated cost --
it can never be worse, which is what makes it safe to leave on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["PulseGate"]


@dataclass
class PulseGate:
    """Per-channel hysteresis gate with a learned threshold and a salience floor.

    Attributes:
        n_channels: Number of independently gated channels.
        tau_hi: Wake threshold per channel. Recompute when the new input differs
            from the cached value by more than this.
        tau_lo: Sleep threshold per channel. Enter sleep when the difference has
            stayed below this for ``sleep_after`` consecutive steps.
        sleep_after: Consecutive below-threshold steps required to sleep. 1 means
            a single quiet step is enough.
        salience_frac: Fraction of highest-magnitude channels that never sleep.
        salience: Per-channel gate magnitude used to pick the protected set,
            shape ``(n_channels,)``.
        cached: Last computed value per channel, shape ``(n_channels,)``.
        awake: Whether each channel is currently computing, ``(n_channels,)``.
        _quiet: Consecutive-quiet counter per channel.
        events: Lifetime count of {wake, sleep, compute, skip} transitions.
    """

    n_channels: int
    tau_hi: NDArray[np.floating] = field(init=False)
    tau_lo: NDArray[np.floating] = field(init=False)
    sleep_after: int = 2
    salience_frac: float = 0.05
    salience: NDArray[np.floating] = field(init=False)
    cached: NDArray[np.floating] = field(init=False)
    awake: NDArray[np.bool_] = field(init=False)
    _quiet: NDArray[np.int64] = field(init=False)
    events: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate thresholds, salience and the awake mask.

        Why ``tau_hi`` starts at 1.0: an untrained gate should skip almost
        nothing. Being conservative on init means an untrained model behaves
        exactly like an ungated one, so the gate can only be learned into being
        useful -- it can never silently break the network at step 0.
        """
        if not 0.0 <= self.salience_frac < 1.0:
            raise ValueError("salience_frac must be in [0, 1)")
        if self.sleep_after < 1:
            raise ValueError("sleep_after must be >= 1")
        n = self.n_channels
        self.tau_hi = np.ones(n, dtype=np.float32)
        self.tau_lo = np.full(n, 0.25, dtype=np.float32)
        self.salience = np.zeros(n, dtype=np.float32)
        self.cached = np.zeros(n, dtype=np.float32)
        self.awake = np.ones(n, dtype=bool)
        self._quiet = np.zeros(n, dtype=np.int64)
        self.events = {"compute": 0, "skip": 0, "wake": 0, "sleep": 0}

    # -- state ---------------------------------------------------------------

    def flush(self) -> None:
        """Wake every channel and drop the cached values.

        Use at any boundary where staleness is not acceptable: a document
        boundary, a context reset, or the end of a chunk. This is the
        documented escape hatch; there is no other hidden reset.
        """
        self.awake.fill(True)
        self.cached.fill(0.0)
        self._quiet.fill(0)
        self.events["wake"] += int(self.n_channels)

    def reset_stats(self) -> None:
        """Zero the lifetime counters, keeping thresholds and cached state."""
        for key in self.events:
            self.events[key] = 0

    # -- the state machine ---------------------------------------------------

    def step(
        self, a: NDArray[np.floating], gate_magnitude: NDArray[np.floating]
    ) -> NDArray[np.bool_]:
        """Advance the gate by one token; return which channels must compute.

        Args:
            a: New input per channel, shape ``(n_channels,)``.
            gate_magnitude: Per-channel importance, same shape. The top
                ``salience_frac`` are protected and never sleep.

        Returns:
            Boolean mask, ``True`` where the channel must be recomputed.

        Raises:
            ValueError: On a shape mismatch.
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size != self.n_channels:
            raise ValueError(
                f"PulseGate expected {self.n_channels} channels, got {a.size}"
            )
        mag = np.abs(np.asarray(gate_magnitude, dtype=np.float32).reshape(-1))
        protected = self._protected(mag)
        delta = np.abs(a - self.cached)

        was_asleep = ~self.awake
        # Wake on a big change, or if this channel is protected (protected
        # channels are never allowed to be asleep in the first place).
        wake = (delta > self.tau_hi) | protected
        self.awake |= wake
        self.events["wake"] += int(np.count_nonzero(wake & was_asleep))

        # Fall asleep after `sleep_after` consecutive quiet steps.
        quiet = delta < self.tau_lo
        self._quiet = np.where(quiet, self._quiet + 1, 0)
        sleep = (self._quiet >= self.sleep_after) & ~protected & self.awake
        self.awake &= ~sleep
        self.events["sleep"] += int(np.count_nonzero(sleep))

        # A channel that has never been computed must compute now, whatever
        # the thresholds say.
        if self.events["compute"] == 0:
            compute = np.ones(self.n_channels, dtype=bool)
        else:
            compute = self.awake.copy()
        self.cached = np.where(compute, a, self.cached)
        self.events["compute"] += int(np.count_nonzero(compute))
        self.events["skip"] += int(np.count_nonzero(~compute))
        return compute

    def _protected(self, magnitude: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Mark exactly the top ``salience_frac`` channels as never-skip.

        Exactly, because a ``>= cutoff`` threshold over-protects whenever the
        magnitudes tie -- and an all-zero salience vector, which is a perfectly
        ordinary input, would then protect *every* channel and silently disable
        the gate for that step. Ranking and slicing a fixed count cannot tie.
        """
        n_protect = round(self.salience_frac * self.n_channels)
        if n_protect <= 0:
            return np.zeros(self.n_channels, dtype=bool)
        top = np.argsort(magnitude, kind="stable")[self.n_channels - n_protect :]
        out = np.zeros(self.n_channels, dtype=bool)
        out[top] = True
        return out

    # -- reporting -----------------------------------------------------------

    def skip_rate(self) -> float:
        """Fraction of channel-steps skipped since the last reset.

        Returns:
            A float in ``[0, 1)``. The design phase measured 0.54 at 0.34% error.
        """
        total = self.events["compute"] + self.events["skip"]
        return self.events["skip"] / total if total else 0.0

    def skip_rate_histogram(self, n_bins: int = 8) -> list[int]:
        """Coarse histogram of the skip rate, for monitoring logs.

        Args:
            n_bins: Number of bins over ``[0, 1]``.

        Returns:
            ``n_bins`` counts. One non-zero bin is the normal healthy case;
        """
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        rate = self.skip_rate()
        index = min(n_bins - 1, int(rate * n_bins))
        counts = [0] * n_bins
        counts[index] = self.events["compute"] + self.events["skip"]
        return counts
