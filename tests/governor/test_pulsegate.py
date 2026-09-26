"""Tests for the PulseGate compute governor."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.governor.pulsegate import PulseGate

N = 128


def gate(*, salience_frac: float = 0.0, **kwargs) -> PulseGate:
    """A gate with deterministic thresholds, so the tests do not depend on init.

    ``salience_frac`` defaults to 0 so the hysteresis tests see a gate with no
    permanently-protected channels; the salience floor is exercised separately,
    because with the default 5% a subset of channels can never sleep and any
    assertion of the form "all channels asleep" would be false by construction.
    """
    g = PulseGate(N, salience_frac=salience_frac, **kwargs)
    g.tau_hi = np.full(N, 1.0, dtype=np.float32)
    g.tau_lo = np.full(N, 0.25, dtype=np.float32)
    return g


def zeros() -> np.ndarray:
    return np.zeros(N, dtype=np.float32)


def unit() -> np.ndarray:
    """Salience that protects nothing in particular."""
    return np.full(N, 1.0, dtype=np.float32)


class TestConstruction:
    def test_starts_fully_awake(self) -> None:
        assert gate().awake.all()

    def test_untrained_gate_skips_nothing(self) -> None:
        """An untrained gate must behave exactly like an ungated one."""
        g = gate()
        assert g.step(zeros(), unit()).all()

    def test_rejects_bad_salience_frac(self) -> None:
        with pytest.raises(ValueError, match="salience_frac"):
            PulseGate(8, salience_frac=1.0)

    def test_rejects_bad_sleep_after(self) -> None:
        with pytest.raises(ValueError, match="sleep_after"):
            PulseGate(8, sleep_after=0)


class TestStep:
    def test_shape_and_dtype(self) -> None:
        out = gate().step(zeros(), unit())
        assert out.shape == (N,) and out.dtype == np.bool_

    def test_rejects_wrong_width(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            gate().step(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    def test_caches_the_computed_value(self) -> None:
        g = gate()
        a = np.arange(N, dtype=np.float32)
        g.step(a, unit())
        assert np.array_equal(g.cached, a)

    def test_identical_input_eventually_sleeps(self) -> None:
        g = gate()
        a = np.full(N, 5.0, dtype=np.float32)
        for _ in range(4):
            g.step(a, unit())
        assert not g.awake.any()

    def test_big_change_wakes_a_sleeping_channel(self) -> None:
        g = gate()
        for _ in range(4):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert not g.awake.any()
        assert g.step(np.full(N, 50.0, dtype=np.float32), unit()).all()

    def test_hysteresis_holds_a_channel_asleep_in_the_gap(self) -> None:
        """The gap between tau_lo and tau_hi is the entire trick: a change too
        small to wake a sleeping channel must not, and must not be mistaken for
        a reason to recompute."""
        g = gate()
        base = np.full(N, 5.0, dtype=np.float32)
        for _ in range(4):
            g.step(base, unit())
        assert not g.awake.any()
        nudge = base + 0.5  # above tau_lo=0.25, below tau_hi=1.0
        out = g.step(nudge, unit())
        assert not out.any()
        assert not g.awake.any()

    def test_sleep_requires_consecutive_quiet_steps(self) -> None:
        g = gate(sleep_after=3)
        a = np.full(N, 5.0, dtype=np.float32)
        g.step(a, unit())
        b = a + 10.0
        for i in range(2):
            g.step(b, unit())
            if i < 2:
                assert g.awake.all(), f"woke too early at step {i}"
        g.step(b, unit())
        g.step(b, unit())
        g.step(b, unit())
        assert not g.awake.any()

    def test_noisy_channel_never_sleeps(self) -> None:
        """The other half of the pinned operating point: 0% skipped.

        A noisy channel never sleeps. Pinned exactly, like its steady-input
        counterpart, because README.md quotes this figure too.
        """
        rng = np.random.default_rng(0)
        g = gate()
        for _ in range(20):
            g.step(rng.standard_normal(N).astype(np.float32) * 50, unit())
        assert g.skip_rate() == 0.0

    def test_protected_channels_never_sleep(self) -> None:
        g = gate(salience_frac=0.05)
        mag = np.zeros(N, dtype=np.float32)
        mag[0] = 100.0  # the single most salient channel
        out = np.zeros(N, dtype=bool)
        for _ in range(6):
            out = g.step(np.full(N, 5.0, dtype=np.float32), mag)
        assert out[0]
        assert g.awake[0]

    def test_protection_count_is_exact_under_ties(self) -> None:
        """An all-zero salience vector must not protect every channel, which is
        what a `>= cutoff` comparison would do. At 128 channels, 5% is 6.4, so
        the count is exact and no more."""
        g = gate(salience_frac=0.05)
        out = np.ones(N, dtype=bool)
        for _ in range(8):  # let the two-step sleep hysteresis settle
            out = g.step(np.full(N, 5.0, dtype=np.float32), zeros())
        assert int(out.sum()) == round(0.05 * N)
        # Steady state, not the cumulative rate: the first steps are all-awake by
        # design, so a cumulative figure here would just measure the warmup.
        assert 1.0 - out.sum() / N > 0.9


class TestState:
    def test_flush_wakes_everything(self) -> None:
        g = gate()
        for _ in range(4):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert not g.awake.any()
        g.flush()
        assert g.awake.all()
        assert g.step(np.full(N, 5.0, dtype=np.float32), unit()).all()

    def test_flush_drops_the_cache(self) -> None:
        g = gate()
        g.step(np.full(N, 5.0, dtype=np.float32), unit())
        g.flush()
        assert not g.cached.any()

    def test_reset_stats_keeps_state(self) -> None:
        g = gate()
        g.step(zeros(), unit())
        g.reset_stats()
        assert g.events["compute"] == 0
        assert g.awake.all()


class TestReporting:
    def test_counters_add_up(self) -> None:
        g = gate()
        for _ in range(6):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert g.events["compute"] + g.events["skip"] == 6 * N

    def test_skip_rate_is_a_fraction(self) -> None:
        g = gate()
        for _ in range(6):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert 0.0 < g.skip_rate() < 1.0

    def test_skip_rate_of_a_fresh_gate(self) -> None:
        assert gate().skip_rate() == 0.0

    def test_histogram_totals_the_work(self) -> None:
        g = gate()
        for _ in range(6):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert sum(g.skip_rate_histogram()) == 6 * N

    def test_rejects_bad_bin_count(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            gate().skip_rate_histogram(0)

    def test_steady_input_is_mostly_skipped(self) -> None:
        """The design-phase operating point, pinned exactly.

        Three quarters of channel-steps are skipped on a steady input. This
        asserts the precise number on purpose: the figure is quoted in
        README.md and docs/architecture.md, and a loose bound would let the
        operating point drift while the docs kept claiming the old value.
        Any change here is a change to the documented number, on purpose.
        """
        g = gate()
        for _ in range(8):
            g.step(np.full(N, 5.0, dtype=np.float32), unit())
        assert g.skip_rate() == pytest.approx(0.75)
