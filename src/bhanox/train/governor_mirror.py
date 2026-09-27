"""Torch mirror of PulseGate, for training only (law C9).

Runtime stays numpy-only. As with the other mirrors, this reproduces the
reference exactly and adds a backward pass.

A known spec bug is mirrored faithfully here rather than fixed
---------------------------------------------------------------
:class:`~bhanox.governor.pulsegate.PulseGate` allocates, counts and checkpoints a
per-channel ``salience`` array, and **never reads it**. ``step()`` takes a
``gate_magnitude`` argument, ranks *that* in ``_protected``, and ignores
``self.salience`` entirely; both call sites in ``model.py`` pass
``np.abs(step_out)``.

So ``salience`` is inert: it cannot change any output, and it can never receive a
gradient. At nano that is 512 of the 1536 "newly counted" gate values -- a third
of the gate's apparent trainable parameters, 0.024% of the model's total -- all of
which are dead weight that the optimizer will dutifully update forever.

This mirror reproduces that faithfully. The obvious "fix" -- reading
``self.salience`` in ``_protected`` -- would be an architecture change to a frozen
spec, it would change which channels are protected, and it would silently
invalidate the measured 0.54 skip rate at 0.34% error. The bug is reported and
pinned by a test instead; see ``test_salience_is_inert`` and the ROADMAP. The
trainer is expected to exclude ``salience`` from its optimizer.

Where the gradient actually comes from
--------------------------------------
The gate's forward is a state machine over booleans: hysteresis, a quiet counter,
a protected set, and a first-step override. None of that differentiates. Two
learned thresholds do, and the whole design is about not cutting them off.

The returned mask is the only thing that carries a graph, and it is built by
running the state machine in *float* with 0.0/1.0 values, where ``max``, ``1 - x``
and ``*`` reproduce ``|``, ``~`` and ``&`` exactly. Each decision boundary that
involves a learnable threshold is wrapped in :func:`~bhanox.train.ste.ste_gt` or
:func:`~bhanox.train.ste.ste_ge` on the *signed margin*, so the gradient reaches
the threshold and not merely the input.

The quiet counter is where this is easiest to get wrong. ``_quiet`` is integer
state, so naively detaching it would sever ``tau_lo`` completely: ``sleep``
requires ``_quiet >= sleep_after``, and a hard 0/1 there means nothing downstream
depends on ``tau_lo`` at all. The reference line is::

    _quiet = np.where(quiet, _quiet + 1, 0)

which has a *reset* in it, and the reset is load-bearing. Reading it as
``_quiet + quiet`` -- "increment when quiet" -- looks equivalent and is not: when
a step is loud, ``quiet`` is 0, so the count keeps its old value instead of
clearing. A channel that had been quiet twice, taken one big kick, and so should
be wide awake, would still read 2 and immediately fall back asleep. It would then
skip every other step forever, and the aggregate skip rate would look perfectly
plausible throughout. The count is therefore rebuilt as a hard value from the
comparison and re-anchored so the gradient survives::

    q_value = where(quiet_ste.detach() > 0.5, _quiet_int + 1, 0)
    _q_inc  = q_value + (quiet_ste - quiet_ste.detach())   # value exact, grad kept

Same re-anchoring idea as the DeltaBank shadow, applied to a counter.

The first step has no gradient, and that is correct
---------------------------------------------------
``compute = np.where(has_run, awake, True)``: a stream that has never computed
must compute, whatever the thresholds say. So on the first step the mask is the
constant 1.0 and carries no gradient. This is not a gap in the estimator, it is
the architecture -- and it is why the gradient tests run a multi-step sequence
rather than a single call, and why a "one call, expect a gradient" test would be
testing the wrong thing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from bhanox.governor.pulsegate import PulseGate
from bhanox.train.ste import ste_ge, ste_gt

__all__ = ["PulseGateMirror"]


class PulseGateMirror(nn.Module):
    """Torch mirror of one :class:`~bhanox.governor.pulsegate.PulseGate`.

    Args:
        gate: The numpy reference gate to mirror. Thresholds are copied, not
            shared, so training cannot mutate the reference.

    Note on ``salience``: it is mirrored as a parameter because the reference
    counts and checkpoints it, but it is deliberately **not read** -- see the
    module docstring. Keeping the attribute is what makes the mirror's parameter
    inventory match the reference's, which is what lets a checkpoint round-trip.
    """

    # Bare annotations: ``nn.Module.__getattr__`` is typed ``Tensor | Module``, so
    # without these every ``self.tau_hi`` and in-place buffer write is a type
    # error. Same remedy as the other mirrors.
    tau_hi: nn.Parameter
    tau_lo: nn.Parameter
    salience: nn.Parameter
    cached: Tensor
    awake: Tensor
    _quiet: Tensor
    _has_run: Tensor

    def __init__(self, gate: PulseGate) -> None:
        super().__init__()
        self.gate = gate
        self.n_channels = int(gate.n_channels)
        self.sleep_after = int(gate.sleep_after)
        self.salience_frac = float(gate.salience_frac)

        def _p(a: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.tensor(np.array(a), dtype=torch.float32))

        self.tau_hi = _p(gate.tau_hi)
        self.tau_lo = _p(gate.tau_lo)
        # Mirrored but never read. See the module docstring and
        # ``test_salience_is_inert``.
        self.salience = _p(gate.salience)

        self.register_buffer(
            "cached", torch.tensor(np.array(gate.cached), dtype=torch.float32)
        )
        self.register_buffer(
            "awake", torch.tensor(np.array(gate.awake), dtype=torch.bool)
        )
        self.register_buffer(
            "_quiet", torch.tensor(np.array(gate._quiet), dtype=torch.int64)
        )
        self.register_buffer(
            "_has_run", torch.tensor(np.array(gate._has_run), dtype=torch.bool)
        )
        self.reset_stats()

    # -- state ---------------------------------------------------------------

    def reset_stats(self) -> None:
        """Zero the lifetime counters and per-step record, keeping state.

        Mirrors ``PulseGate.reset_stats``. ``step_rates`` has no reference
        counterpart: the reference's histogram is lifetime-only, and a D5 monitor
        needs the *distribution* across steps, not one aggregate number. One
        coarse bin holding every channel-step tells you the run skipped 54%
        overall and nothing about whether that was steady or a single collapse.
        """
        self.events: dict[str, int] = {"compute": 0, "skip": 0, "wake": 0, "sleep": 0}
        self.step_rates: list[float] = []

    def ensure_batch(self, batch: int) -> None:
        """Grow the per-channel state, mirroring ``PulseGate.ensure_batch``.

        New rows start asleep-clean: zeros for the cache and the counter, and
        ``_has_run`` False so a fresh row must compute on its first token. The
        reference initialises ``awake`` True for new rows and this does too --
        a fresh row is awake, it simply has not run yet.
        """
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        have = int(self.awake.shape[0])
        if batch <= have:
            return
        pad = batch - have
        self.cached = torch.cat(
            [self.cached, torch.zeros((pad, self.n_channels), dtype=torch.float32)]
        )
        self.awake = torch.cat(
            [self.awake, torch.ones((pad, self.n_channels), dtype=torch.bool)]
        )
        self._quiet = torch.cat(
            [self._quiet, torch.zeros((pad, self.n_channels), dtype=torch.int64)]
        )
        self._has_run = torch.cat([self._has_run, torch.zeros(pad, dtype=torch.bool)])

    @torch.no_grad()
    def flush(self) -> None:
        """Wake every channel and drop the cache, mirroring ``PulseGate.flush``.

        ``_has_run`` is cleared as well, and has to be: it exists to force the
        first computation on a never-run channel, so a gate carrying the flag
        across a flush would never warm up again.
        """
        self.events["wake"] += int((~self.awake).sum())
        self.awake.fill_(True)
        self.cached.zero_()
        self._quiet.zero_()
        self._has_run.zero_()

    # -- the protected set ---------------------------------------------------

    def _protected(self, magnitude: Tensor) -> Tensor:
        """Top ``salience_frac`` channels per row, mirroring ``_protected``.

        Exactly a fixed count per row, chosen by a stable ascending argsort. The
        stability is what makes ties safe: a ``>= cutoff`` test would protect
        every channel whenever the magnitudes tie, and an all-equal magnitude
        row -- an entirely ordinary input -- would silently disable the gate for
        that step. Ranking a fixed count cannot over-protect.

        Note what is *not* an input here: ``self.salience``. The reference ranks
        the ``gate_magnitude`` argument, not the learned vector.
        """
        n_protect = round(self.salience_frac * self.n_channels)
        if n_protect <= 0:
            return torch.zeros(
                magnitude.shape, dtype=torch.bool, device=magnitude.device
            )
        order = torch.argsort(magnitude, dim=-1, stable=True)
        top = order[..., self.n_channels - n_protect :]
        out = torch.zeros(magnitude.shape, dtype=torch.bool, device=magnitude.device)
        out.scatter_(-1, top, True)
        return out

    # -- the state machine, in float so it can carry a graph ----------------

    def step(self, a: Tensor, gate_magnitude: Tensor) -> Tensor:
        """Advance the gate by one token; return the compute mask.

        Args:
            a: ``(B, n_channels)`` or ``(n_channels,)`` input activations.
            gate_magnitude: Per-channel importance, same shape. The top
                ``salience_frac`` are protected and never sleep.

        Returns:
            A float32 tensor of 0.0/1.0 shaped like ``a``, where 1.0 means
            "recompute". The *values* are the reference's exact hard mask; unlike
            the reference's bool return, this tensor also carries the autograd
            graph back to ``tau_hi`` and ``tau_lo``.

        Raises:
            ValueError: On a shape mismatch.
        """
        arr = torch.as_tensor(a, dtype=torch.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_channels:
            raise ValueError(
                f"PulseGateMirror expected (B, {self.n_channels}) channels or a "
                f"single ({self.n_channels},) vector, got shape {tuple(arr.shape)}"
            )
        mag = torch.as_tensor(gate_magnitude, dtype=torch.float32).abs()
        if mag.ndim == 1:
            mag = mag[None, :]
        if mag.shape != arr.shape:
            raise ValueError(
                f"gate_magnitude shape {tuple(mag.shape)} does not match input "
                f"{tuple(arr.shape)}"
            )

        batch = int(arr.shape[0])
        self.ensure_batch(batch)

        protected = self._protected(mag)
        protected_f = protected.to(arr.dtype)
        # Cloned, not sliced. These are views into buffers that the state advance
        # below overwrites in place, and autograd's version counter would reject
        # the backward pass: the saved tensors would come back marked dirty
        # mid-graph. The clone is also the honest value here -- the forward must
        # read the *pre-write* cache, exactly as the reference does.
        cached = self.cached[:batch].clone()
        awake_prev = self.awake[:batch].clone()
        has_run = self._has_run[:batch].clone()
        quiet_int = self._quiet[:batch].to(arr.dtype)

        delta = (arr - cached).abs()
        # The signed margins. tau_hi enters with a -1, tau_lo with a +1, so
        # each threshold gets gradient of the useful sign rather than none.
        #
        # ``protected`` is OR'd into the wake, not just into the sleep term: the
        # reference's wake is ``(delta > tau_hi) | protected``, so a protected
        # channel that had gone to sleep wakes on *any* input at all, however
        # small. Leaving it out looks fine -- a protected channel is usually also
        # the loudest one and wakes on its own -- and then only shows up as a
        # handful of quiet channels that never compute, one row in, which is
        # exactly the kind of discrepancy an aggregate skip rate hides.
        wake_f = torch.maximum(ste_gt(delta - self.tau_hi), protected_f)
        quiet_f = ste_gt(self.tau_lo - delta)

        awake_new_f = torch.maximum(awake_prev.to(arr.dtype), wake_f)
        # The reference is ``np.where(quiet, _quiet + 1, 0)`` -- a loud step
        # *resets* the counter, it does not merely decline to increment it. Writing
        # this as ``_quiet + quiet`` is the obvious simplification and it is wrong
        # in a way that survives a long time: a channel that was quiet twice, then
        # took one big kick, would keep its count of 2 and go straight back to
        # sleep on the following step, so it would skip 50% of steps forever while
        # the aggregate skip rate still looked plausible.
        q_value = torch.where(quiet_f.detach() > 0.5, quiet_int + 1.0, quiet_int * 0.0)
        # Re-anchored: exact value from the hard decision, gradient to ``tau_lo``
        # carried by the straight-through comparison.
        q_inc_f = q_value + (quiet_f - quiet_f.detach())
        sleep_f = (
            ste_ge(q_inc_f - float(self.sleep_after))
            * (1.0 - protected_f)
            * awake_new_f
        )
        awake_final_f = awake_new_f * (1.0 - sleep_f)
        compute_f = torch.where(
            has_run[:, None], awake_final_f, torch.ones_like(awake_final_f)
        )

        # ---- state advance, all off the detached hard values ----
        wake_hard = wake_f.detach() > 0.5
        sleep_hard = sleep_f.detach() > 0.5
        compute_hard = compute_f.detach() > 0.5
        with torch.no_grad():
            was_asleep = ~awake_prev
            self.events["wake"] += int((wake_hard & was_asleep).sum())
            self.events["sleep"] += int(sleep_hard.sum())
            self.events["compute"] += int(compute_hard.sum())
            self.events["skip"] += int((~compute_hard).sum())
            self._quiet[:batch] = q_inc_f.detach().to(torch.int64)
            self.awake[:batch] = awake_final_f.detach() > 0.5
            self._has_run[:batch] = True
            self.cached[:batch] = torch.where(compute_hard, arr, cached)
        self.step_rates.append(float((~compute_hard).to(torch.float32).mean()))

        return compute_f[0] if single else compute_f

    def forward(self, a: Tensor, gate_magnitude: Tensor) -> Tensor:
        """Alias for :meth:`step`, so the module is callable like the reference."""
        return self.step(a, gate_magnitude)

    # -- reporting -----------------------------------------------------------

    def skip_rate(self) -> float:
        """Fraction of channel-steps skipped. Mirrors ``PulseGate.skip_rate``."""
        total = self.events["compute"] + self.events["skip"]
        return self.events["skip"] / total if total else 0.0

    def skip_rate_histogram(self, n_bins: int = 8) -> list[int]:
        """Lifetime coarse histogram, mirroring ``PulseGate.skip_rate_histogram``."""
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        rate = self.skip_rate()
        index = min(n_bins - 1, int(rate * n_bins))
        counts = [0] * n_bins
        counts[index] = self.events["compute"] + self.events["skip"]
        return counts

    def step_skip_histogram(self, n_bins: int = 10) -> list[int]:
        """Histogram of *per-step* skip rates, for the D5 monitor.

        Args:
            n_bins: Number of bins over ``[0, 1]``. The last bin is closed, so a
                rate of exactly 1.0 lands in it rather than overflowing.

        Returns:
            ``n_bins`` counts, one per observed step.
        """
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        counts = [0] * n_bins
        for rate in self.step_rates:
            counts[min(n_bins - 1, int(rate * n_bins))] += 1
        return counts

    def param_count(self) -> int:
        """Mirrors ``PulseGate.param_count``: the three per-channel arrays.

        Counts ``salience`` because the reference does, so the two inventories
        agree. That agreement is part of what makes the dead 512 values visible
        rather than quietly absent.
        """
        return int(self.tau_hi.numel() + self.tau_lo.numel() + self.salience.numel())

    def load_from_numpy(self, gate: PulseGate) -> None:
        """Copy reference thresholds and state in, e.g. after a checkpoint load.

        Raises:
            ValueError: If the reference holds *fewer* batch rows than the mirror
                does. A checkpoint carries the whole per-sample state, so the two
                shapes have to agree exactly; a mirror left with extra rows would
                otherwise keep stale hysteresis that the restored run never
                revisits, and quietly train against it.
        """
        rows = int(gate.cached.shape[0])
        have = int(self.awake.shape[0])
        if rows < have:
            raise ValueError(
                f"reference gate has {rows} batch rows but the mirror holds {have}; "
                "flush the mirror before reloading a checkpoint with fewer rows"
            )
        self.ensure_batch(rows)
        with torch.no_grad():
            for name in ("tau_hi", "tau_lo", "salience"):
                getattr(self, name).copy_(
                    torch.tensor(np.array(getattr(gate, name)), dtype=torch.float32)
                )
            self.cached[:rows] = torch.tensor(
                np.array(gate.cached), dtype=torch.float32
            )
            self.awake[:rows] = torch.tensor(np.array(gate.awake), dtype=torch.bool)
            self._quiet[:rows] = torch.tensor(np.array(gate._quiet), dtype=torch.int64)
            self._has_run[:rows] = torch.tensor(
                np.array(gate._has_run), dtype=torch.bool
            )
        self.reset_stats()

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, sleep_after={self.sleep_after}, "
            f"salience_frac={self.salience_frac}, salience_inert=True"
        )
