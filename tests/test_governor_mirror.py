"""Agreement and gradient tests for the PulseGate torch mirror.

The reference is a boolean state machine, so "agreement" here is unusually
strict and unusually easy to fake. The returned mask is 0 or 1, so a mirror that
returned all-ones, or that ran a *different* hysteresis schedule, would still
agree on the first step and would still produce a plausible-looking skip rate.
Every test below therefore checks against the reference on a multi-step sequence
and compares the full state, not just the mask.

Three bugs found this way are pinned rather than left to regress silently:

* a loud step must *reset* the quiet counter, not merely decline to increment it;
* ``protected`` belongs in the wake condition, not only in the sleep condition;
* ``salience`` is never read at all, so it is dead weight (see the module
  docstring of ``governor_mirror``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bhanox.governor.pulsegate import PulseGate
from bhanox.train.governor_mirror import PulseGateMirror

N = 128
# 5% of 128 = 6, and the reference rounds rather than floors.
N_PROTECT = 6
# sleep_after defaults to 2, tau_hi to 1.0, tau_lo to 0.25.
SLEEP_AFTER = 2


def _pair(n_channels: int = N) -> tuple[PulseGate, PulseGateMirror]:
    """A reference gate and a fresh mirror of it, sharing thresholds by copy."""
    gate = PulseGate(n_channels=n_channels)
    return gate, PulseGateMirror(gate)


def _hard(mask: torch.Tensor) -> np.ndarray:
    """The mirror's float 0/1 mask as the reference's exact boolean mask."""
    return mask.detach().numpy() > 0.5


def _assert_state_agrees(
    gate: PulseGate, mirror: PulseGateMirror, batch: int | None = None
) -> None:
    """Every state buffer and the event counters must match, not just the mask."""
    rows = gate.awake.shape[0] if batch is None else batch
    assert np.array_equal(
        gate.cached[:rows], mirror.cached[:rows].numpy()
    ), "cache diverged"
    assert np.array_equal(
        gate.awake[:rows], mirror.awake[:rows].numpy()
    ), "awake diverged"
    assert np.array_equal(
        gate._quiet[:rows], mirror._quiet[:rows].numpy()
    ), "quiet diverged"
    assert np.array_equal(
        gate._has_run[:rows], mirror._has_run[:rows].numpy()
    ), "has_run diverged"
    assert (
        gate.events == mirror.events
    ), f"events diverged: {gate.events} vs {mirror.events}"


# -- agreement -------------------------------------------------------------


def test_single_step_matches_reference():
    """One step, the smallest case that is not vacuous."""
    gate, mirror = _pair()
    a = np.random.default_rng(0).standard_normal((4, N)).astype(np.float32)
    mag = np.abs(np.random.default_rng(1).standard_normal((4, N)).astype(np.float32))
    assert np.array_equal(
        gate.step(a, mag), _hard(mirror.step(torch.tensor(a), torch.tensor(mag)))
    )
    _assert_state_agrees(gate, mirror)


def test_long_sequence_matches_reference_bit_exact():
    """24+ steps, deliberately straddling the wake/sleep/wake cycle.

    The first two steps are all-zero, so every channel quiets down and sleeps;
    step two then delivers a full-amplitude random input, which wakes most of
    them again. That round trip is the one that catches counter-reset bugs: a
    counter that is not reset on a loud step looks fine right up until a channel
    has to wake, and then skips forever.

    Also runs three batch rows, so a bug that couples samples through the
    protected set or the counter would show up as a per-row disagreement.
    """
    gate, mirror = _pair()
    rng = np.random.default_rng(5)
    for step in range(24):
        a = rng.standard_normal((3, N)).astype(np.float32)
        if step < 2:
            a = np.zeros((3, N), dtype=np.float32)
        mag = rng.standard_normal((3, N)).astype(np.float32)
        ref = gate.step(a, mag)
        got = _hard(mirror.step(torch.tensor(a), torch.tensor(mag)))
        assert np.array_equal(ref, got), f"mask diverged at step {step}"
        _assert_state_agrees(gate, mirror, batch=3)
    # A sequence that never sleeps would pass a weaker test trivially.
    assert gate.events["sleep"] > 0
    assert gate.events["wake"] > 0
    assert 0.0 < gate.skip_rate() < 1.0


def test_sequences_match_across_many_seeds():
    """Agreement is not a property of one lucky input draw."""
    for seed in range(12):
        gate, mirror = _pair()
        rng = np.random.default_rng(1000 + seed)
        for step in range(10):
            a = rng.standard_normal((2, N)).astype(np.float32)
            mag = np.abs(rng.standard_normal((2, N)).astype(np.float32))
            assert np.array_equal(
                gate.step(a, mag),
                _hard(mirror.step(torch.tensor(a), torch.tensor(mag))),
            ), f"seed {seed} step {step}"
        _assert_state_agrees(gate, mirror, batch=2)


def test_single_vector_shape_is_preserved():
    """A 1-D input returns a 1-D mask, as the reference promises."""
    gate, mirror = _pair()
    a = np.zeros(N, dtype=np.float32)
    mag = np.zeros(N, dtype=np.float32)
    ref = gate.step(a, mag)
    got = mirror.step(torch.tensor(a), torch.tensor(mag))
    assert ref.shape == (N,)
    assert tuple(got.shape) == (N,)
    assert np.array_equal(ref, _hard(got))


def test_batch_growth_matches_reference():
    """A growing batch must leave the earlier rows untouched and agree exactly.

    ``ensure_batch`` is where a mirror can quietly diverge: the new rows start
    awake with ``_has_run`` False, so a fresh row must compute on its first token
    regardless of what the other rows have already been doing.
    """
    gate, mirror = _pair()
    rng = np.random.default_rng(7)
    for batch in (1, 4, 4, 9, 2):
        a = rng.standard_normal((batch, N)).astype(np.float32)
        mag = np.abs(rng.standard_normal((batch, N)).astype(np.float32))
        assert np.array_equal(
            gate.step(a, mag), _hard(mirror.step(torch.tensor(a), torch.tensor(mag)))
        ), f"batch {batch}"
        _assert_state_agrees(gate, mirror, batch=batch)


def test_fresh_row_computes_even_when_quiet():
    """A newly grown row computes on sight, whatever the thresholds say.

    This is the ``_has_run`` override, and it is per sample. A row that had been
    silent for two steps in the reference is asleep; a row added later has never
    run and must compute.
    """
    gate, mirror = _pair()
    quiet = np.zeros((1, N), dtype=np.float32)
    zero = np.zeros((1, N), dtype=np.float32)
    for _ in range(4):
        gate.step(quiet, zero)
        mirror.step(torch.tensor(quiet), torch.tensor(zero))
    assert gate.awake[0].sum() == N_PROTECT, "row 0 should have slept"

    grown = np.zeros((3, N), dtype=np.float32)
    grown_mag = np.zeros((3, N), dtype=np.float32)
    ref = gate.step(grown, grown_mag)
    got = _hard(mirror.step(torch.tensor(grown), torch.tensor(grown_mag)))
    assert np.array_equal(ref, got)
    assert ref[1].all(), "a never-run row must compute on its first token"
    assert ref[2].all(), "a never-run row must compute on its first token"
    assert ref[0].sum() == N_PROTECT, "the older row keeps its own hysteresis"


def test_flush_matches_reference():
    """``flush`` wakes everything and clears ``_has_run``, in both.

    The state has to go back to the post-construction values, not merely to
    "awake": carrying ``_has_run`` across a flush leaves a channel that has never
    been computed free to skip, forever.
    """
    gate, mirror = _pair()
    rng = np.random.default_rng(3)
    for _ in range(6):
        a = rng.standard_normal((2, N)).astype(np.float32)
        mag = np.abs(rng.standard_normal((2, N)).astype(np.float32))
        gate.step(a, mag)
        mirror.step(torch.tensor(a), torch.tensor(mag))
    assert gate.awake.sum() < gate.awake.size, "expected the gate to have slept"

    gate.flush()
    mirror.flush()
    assert np.array_equal(gate.awake, mirror.awake.numpy())
    assert np.array_equal(gate.cached, mirror.cached.numpy())
    assert np.array_equal(gate._quiet, mirror._quiet.numpy())
    assert np.array_equal(gate._has_run, mirror._has_run.numpy())
    assert gate.awake.all() and not gate._has_run.any()
    # Wake accounting: both count the channels that flush had to wake.
    assert gate.events["wake"] == mirror.events["wake"]


def test_reset_stats_zeroes_counters_in_both():
    """``reset_stats`` clears the lifetime counters but keeps the thresholds."""
    gate, mirror = _pair()
    a = np.zeros((1, N), dtype=np.float32)
    mag = np.zeros((1, N), dtype=np.float32)
    for _ in range(4):
        gate.step(a, mag)
        mirror.step(torch.tensor(a), torch.tensor(mag))
    assert gate.events["compute"] > 0
    tau_before = mirror.tau_hi.detach().clone()
    gate.reset_stats()
    mirror.reset_stats()
    assert gate.events == mirror.events
    assert set(gate.events.values()) == {0}
    assert mirror.step_rates == []
    assert torch.equal(mirror.tau_hi, tau_before)
    # And both are usable again afterwards.
    a2 = np.random.default_rng(4).standard_normal((1, N)).astype(np.float32)
    m2 = np.abs(np.random.default_rng(5).standard_normal((1, N)).astype(np.float32))
    assert np.array_equal(
        gate.step(a2, m2), _hard(mirror.step(torch.tensor(a2), torch.tensor(m2)))
    )


# -- the two bugs this suite exists to pin ---------------------------------


def test_loud_step_resets_the_quiet_counter():
    """A loud step must clear the counter, not just stop incrementing it.

    The regression: written as ``count + quiet``, a channel that slept on two
    quiet steps and then took one big kick still reads 2, so it goes straight
    back to sleep on the next step and skips forever after. Pinned by running
    past exactly that point and requiring the channel to stay awake.
    """
    gate, mirror = _pair()
    zero = torch.zeros(1, N)
    rng = np.random.default_rng(21)
    # Two quiet steps put every unprotected channel to sleep.
    for _ in range(2):
        gate.step(np.zeros((1, N), np.float32), np.zeros((1, N), np.float32))
        mirror.step(zero, zero)
    assert gate._quiet.max() == SLEEP_AFTER

    # One loud step: the counter must be cleared, not left at 2.
    loud = torch.tensor(np.full((1, N), 9.0, dtype=np.float32))
    gate.step(np.full((1, N), 9.0, np.float32), np.zeros((1, N), np.float32))
    mirror.step(loud, zero)
    assert gate._quiet.max() == 0, "a loud step must reset the quiet counter"
    assert np.array_equal(gate._quiet, mirror._quiet.numpy())
    assert gate.awake.sum() > N_PROTECT, "the loud step must wake the sleepers"

    # And they must still be awake on the following quiet step, which is where
    # a stale counter of 2 would put them back to sleep.
    still = np.zeros((1, N), np.float32)
    gate.step(still, still)
    mirror.step(zero, zero)
    assert np.array_equal(gate.awake, mirror.awake.numpy())
    assert (
        gate.awake.sum() > N_PROTECT
    ), "a channel woken by a loud step must not fall straight back to sleep"
    assert rng is not None


def test_protected_channel_wakes_on_any_change():
    """``protected`` is part of the wake condition, not only the sleep condition.

    The reference wakes on ``(delta > tau_hi) | protected``. A mirror that omits
    the ``protected`` term from the wake looks correct for the loud channels --
    which is to say for most protected channels, since the largest gate
    magnitudes usually also move the most -- and quietly leaves the small
    protected ones asleep.

    Note the set up. A protected channel is never allowed to sleep in the first
    place, so a channel that has been protected all along cannot show this bug.
    It needs a channel that went to sleep while unprotected and is *promoted*
    later, which is the ordinary case: ``gate_magnitude`` is recomputed every
    step, so the protected set moves. Phase 1 protects the tail, phase 2
    promotes channel 0, and channel 0 is asleep across the boundary.
    """
    gate, mirror = _pair()
    quiet = np.zeros((1, N), dtype=np.float32)

    # Phase 1: the tail is protected, so everything else sleeps.
    mag_tail = np.zeros((1, N), dtype=np.float32)
    mag_tail[0, -1] = 10.0
    for _ in range(3):
        gate.step(quiet, mag_tail)
        mirror.step(torch.tensor(quiet), torch.tensor(mag_tail))
    assert not bool(mirror.awake[0, 0]), "channel 0 should have slept"
    assert bool(mirror.awake[0, -1]), "a protected channel never sleeps"
    assert bool(
        gate._has_run[0]
    ), "the row has run, so _has_run is not holding it awake"

    # Phase 2: channel 0 becomes the largest magnitude, so it is protected now.
    mag_head = np.zeros((1, N), dtype=np.float32)
    mag_head[0, 0] = 10.0
    promoted = mirror._protected(torch.tensor(mag_head))
    assert bool(promoted[0, 0]), "channel 0 should now be protected"

    # A change far below tau_hi=1.0: not a wake for an ordinary channel. Channel
    # 0's cache is still 0, so delta is 0.05 and the threshold cannot fire.
    tiny = np.full((1, N), 0.05, dtype=np.float32)
    gate.step(tiny, mag_head)
    mirror.step(torch.tensor(tiny), torch.tensor(mag_head))
    assert gate.awake[0, 0], "a newly protected channel must wake on any input"
    assert np.array_equal(gate.awake, mirror.awake.numpy())
    assert gate.cached[0, 0] == pytest.approx(0.05), "and it must then cache the input"


def test_salience_is_inert():
    """``salience`` is allocated and checkpointed but never read. Pin that.

    Changing it cannot change any output, and it can never receive a gradient.
    This is a spec bug, mirrored deliberately rather than fixed: reading
    ``salience`` in ``_protected`` would change which channels are protected and
    would invalidate the measured 0.54 skip rate at 0.34% error. If a future
    change makes ``salience`` live, this test should fail loudly -- that is the
    moment the ROADMAP, the byte accounting and the measured claims all have to
    be revisited together.
    """
    gate, mirror = _pair()
    rng = np.random.default_rng(31)
    a = rng.standard_normal((2, N)).astype(np.float32)
    m = np.abs(rng.standard_normal((2, N)).astype(np.float32))

    # Two mirrors on the same reference, identical except for ``salience``, run
    # over the same sequence. Comparing against the reference instead would
    # prove less: the gate is a state machine, so two calls on the same input
    # legitimately differ, and "the second mask is not the first" says nothing
    # about salience. This way the only variable is salience.
    other = PulseGateMirror(gate)
    with torch.no_grad():
        other.salience.fill_(1e6)
    with torch.no_grad():
        mirror.salience.zero_()
    assert not np.array_equal(
        mirror.salience.detach().numpy(), other.salience.detach().numpy()
    )
    for _ in range(4):
        assert np.array_equal(
            _hard(mirror(torch.tensor(a), torch.tensor(m))),
            _hard(other(torch.tensor(a), torch.tensor(m))),
        ), "salience must not influence the gate"
        assert np.array_equal(
            mirror.cached.numpy(), other.cached.numpy()
        ), "salience must not influence the cache either"

    # And the gradient says the same thing from the other direction.
    mirror.zero_grad(set_to_none=True)
    a_grad = torch.tensor(a, requires_grad=True)
    mirror(a_grad, torch.tensor(m)).sum().backward()
    assert mirror.salience.grad is None, "salience must never receive a gradient"
    assert float(mirror.tau_hi.grad.norm()) > 0.0, "the graph itself is live"
    # It is still counted, which is the whole reason the number is worth quoting.
    assert gate.param_count() == mirror.param_count() == 3 * N


# -- gradients -------------------------------------------------------------


def test_tau_hi_receives_a_gradient():
    """``tau_hi`` is learnable, so it must be on the graph.

    A single step is not a fair test: on the first step the mask is the constant
    1.0 (``_has_run`` is False) and carries no gradient at all. That is the
    architecture, not a gap. So this runs a sequence and requires a non-zero
    gradient on at least one step, then a non-zero total.
    """
    _, mirror = _pair()
    rng = np.random.default_rng(41)
    nonzero_steps = 0
    for _ in range(12):
        a = torch.tensor(rng.standard_normal((2, N)).astype(np.float32))
        mag = torch.tensor(np.abs(rng.standard_normal((2, N)).astype(np.float32)))
        mirror.zero_grad(set_to_none=True)
        mirror(a, mag).sum().backward()
        assert mirror.tau_hi.grad is not None, "tau_hi left the graph entirely"
        if float(mirror.tau_hi.grad.norm()) > 0.0:
            nonzero_steps += 1
    assert nonzero_steps > 0, "tau_hi never received a gradient"


def test_tau_lo_receives_a_gradient():
    """``tau_lo`` is learnable and sits on the sleep path.

    This is the one that is easy to sever by accident. ``_quiet`` is integer
    state, so a mirror that detaches it before comparing against
    ``sleep_after`` produces a correct forward and a permanently zero ``tau_lo``
    gradient -- a threshold that looks trained and never moves.
    """
    _, mirror = _pair()
    # Quiet input, so channels reach the sleep decision and the counter matters.
    steps_with_grad = 0
    for _ in range(8):
        a = torch.zeros(2, N)
        mag = torch.zeros(2, N)
        mirror.zero_grad(set_to_none=True)
        mirror(a, mag).sum().backward()
        assert mirror.tau_lo.grad is not None, "tau_lo left the graph entirely"
        if float(mirror.tau_lo.grad.norm()) > 0.0:
            steps_with_grad += 1
    assert steps_with_grad > 0, "tau_lo never received a gradient on the sleep path"


def test_both_thresholds_and_the_input_are_trained_together():
    """One sequence, one backward: all three contributors are live."""
    _, mirror = _pair()
    rng = np.random.default_rng(43)
    a = None
    for _ in range(10):
        a = torch.tensor(
            rng.standard_normal((2, N)).astype(np.float32), requires_grad=True
        )
        mag = torch.tensor(np.abs(rng.standard_normal((2, N)).astype(np.float32)))
        out = mirror(a, mag)
        out.sum().backward()
    assert float(mirror.tau_hi.grad.norm()) > 0.0
    assert float(mirror.tau_lo.grad.norm()) > 0.0
    assert a is not None and a.grad is not None and float(a.grad.norm()) > 0.0


def test_mask_is_exactly_zero_or_one():
    """The forward must be the hard mask, not a soft relaxation.

    The straight-through terms carry gradient, but their *value* has to be the
    reference's boolean. A mask that leaked fractions would break the skip
    accounting, and would quietly change what the compute/skip split means.
    """
    _, mirror = _pair()
    rng = np.random.default_rng(47)
    for _ in range(6):
        a = torch.tensor(rng.standard_normal((2, N)).astype(np.float32))
        mag = torch.tensor(np.abs(rng.standard_normal((2, N)).astype(np.float32)))
        out = mirror(a, mag)
        assert set(out.detach().flatten().tolist()) <= {0.0, 1.0}


# -- the protected set -----------------------------------------------------


def test_protected_count_is_exact_when_magnitudes_tie():
    """Exactly ``n_protect`` per row, even when every magnitude is equal.

    An all-equal magnitude row is an ordinary input, not a corner case. A
    ``>= cutoff`` test would protect the whole row on it and disable the gate for
    that step, so the reference ranks a fixed count instead.
    """
    _, mirror = _pair()
    for magnitude in (np.zeros((3, N), np.float32), np.ones((3, N), np.float32)):
        protected = mirror._protected(torch.tensor(magnitude))
        assert protected.sum(dim=-1).tolist() == [N_PROTECT] * 3


def test_protected_set_is_chosen_per_row():
    """Each row protects its own largest magnitudes.

    Ranking the batch as a whole would couple the samples through the gate:
    sample b's protection would depend on what sample b' fed it.
    """
    _, mirror = _pair()
    magnitude = np.zeros((2, N), dtype=np.float32)
    magnitude[0, 0] = 100.0  # huge in row 0 only
    protected = mirror._protected(torch.tensor(magnitude))
    assert bool(protected[0, 0])
    assert not bool(protected[1, 0])
    assert int(protected[0].sum()) == N_PROTECT
    assert int(protected[1].sum()) == N_PROTECT
    # Row 1 is all-equal, so it falls back to the tail, deterministically.
    assert protected[1].tolist() == [False] * (N - N_PROTECT) + [True] * N_PROTECT


def test_salience_frac_zero_protects_nothing():
    """``salience_frac=0`` is a legal configuration and protects nothing."""
    gate = PulseGate(n_channels=N, salience_frac=0.0)
    mirror = PulseGateMirror(gate)
    protected = mirror._protected(torch.zeros(2, N))
    assert not bool(protected.any())
    zero = np.zeros((2, N), dtype=np.float32)
    for _ in range(4):
        assert np.array_equal(
            gate.step(zero, zero),
            _hard(mirror.step(torch.tensor(zero), torch.tensor(zero))),
        )
    assert not gate.awake.any(), "with nothing protected everything should sleep"


# -- operating points ------------------------------------------------------


def test_steady_state_skip_rate_hits_75_percent():
    """The 75% skip operating point, as a steady-state rate.

    26 of the 128 channels are kicked every step with an alternating sign, so
    their delta stays at 10.0 -- above ``tau_hi`` -- and they never sleep. The 6
    protected channels also always compute. That leaves 32 computing and 96
    skipping, which is 0.75.

    The first step computes everything: the 26 kickers have no history, and the
    other 96 have only been quiet once, one short of ``sleep_after``. They drop
    out on step one and the rate settles at 0.75 for every step after. So the
    assertion is on the per-step rate rather than the lifetime ``skip_rate``,
    which would be diluted by that one warmup step.
    """
    gate, mirror = _pair()
    zero = np.zeros((1, N), dtype=np.float32)
    for step in range(12):
        a = np.zeros((1, N), dtype=np.float32)
        a[0, :26] = 5.0 if step % 2 == 0 else -5.0
        gate.step(a, zero)
        mirror.step(torch.tensor(a), torch.tensor(zero))
    assert mirror.step_rates[0] == 0.0, "the first step has no history to skip on"
    for rate in mirror.step_rates[1:]:
        assert rate == pytest.approx(0.75), f"expected 75% skip, got {rate}"
    assert mirror.skip_rate() == gate.skip_rate()
    assert int((~gate.awake[0]).sum()) == N - 26 - N_PROTECT


def test_steady_input_leaves_only_protected_channels_awake():
    """A completely silent input sleeps everything unprotected."""
    gate, mirror = _pair()
    zero = np.zeros((1, N), dtype=np.float32)
    for _ in range(12):
        gate.step(zero, zero)
        mirror.step(torch.tensor(zero), torch.tensor(zero))
    assert int(gate.awake[0].sum()) == N_PROTECT
    assert int(mirror.awake[0].sum()) == N_PROTECT
    assert mirror.step_rates[-1] == pytest.approx(1.0 - N_PROTECT / N)


def test_loud_input_skips_nothing_at_all():
    """A signal that always moves well past ``tau_hi`` never skips.

    The other end of the operating range, and the one that matters for the claim
    that gating "can never be worse": on this input the gate is a no-op.
    """
    gate, mirror = _pair()
    zero = np.zeros((1, N), dtype=np.float32)
    for step in range(12):
        a = np.full((1, N), 100.0 if step % 2 == 0 else -100.0, dtype=np.float32)
        gate.step(a, zero)
        mirror.step(torch.tensor(a), torch.tensor(zero))
    assert mirror.step_rates == [0.0] * 12
    assert mirror.skip_rate() == 0.0 == gate.skip_rate()
    assert gate.events["skip"] == 0


def test_random_input_saturates_towards_no_skipping():
    """Real activations keep the gate near-fully awake.

    Not pinned to a number: the exact rate depends on the draw, and the honest
    claim is the direction. A hard ``== 0.0`` here would be testing the seed.
    The bar is that random input essentially never skips, which is the "can never
    be worse" guarantee doing its job.
    """
    gate, mirror = _pair()
    rng = np.random.default_rng(53)
    for _ in range(16):
        a = rng.standard_normal((1, N)).astype(np.float32) * 30.0
        mag = np.abs(rng.standard_normal((1, N)).astype(np.float32))
        gate.step(a, mag)
        mirror.step(torch.tensor(a), torch.tensor(mag))
    assert (
        mirror.skip_rate() < 0.005
    ), f"expected near-zero skipping, got {mirror.skip_rate()}"
    assert mirror.skip_rate() == gate.skip_rate()


def test_skip_rate_is_a_lifetime_average_over_both_outcomes():
    """``skip_rate`` and the histogram must account for every channel-step."""
    gate, mirror = _pair()
    rng = np.random.default_rng(59)
    for _ in range(8):
        a = rng.standard_normal((3, N)).astype(np.float32)
        mag = np.abs(rng.standard_normal((3, N)).astype(np.float32))
        gate.step(a, mag)
        mirror.step(torch.tensor(a), torch.tensor(mag))
    total = 8 * 3 * N
    assert gate.events["compute"] + gate.events["skip"] == total
    assert mirror.events["compute"] + mirror.events["skip"] == total
    assert mirror.skip_rate() == pytest.approx(gate.events["skip"] / total)
    assert sum(mirror.skip_rate_histogram(8)) == total


# -- D5 monitoring ---------------------------------------------------------


def test_step_skip_histogram_counts_every_step():
    """The D5 monitor needs the *distribution* of skip rates across steps.

    The reference's histogram is lifetime-only: one coarse bin holding every
    channel-step. That tells you the run skipped 54% overall and nothing about
    whether it was steady or a single collapse partway through. The per-step
    record is the difference, so it has to actually account for the run.
    """
    _, mirror = _pair()
    zero = np.zeros((1, N), dtype=np.float32)
    for _ in range(12):
        mirror.step(torch.tensor(zero), torch.tensor(zero))
    assert len(mirror.step_rates) == 12
    for n_bins in (1, 4, 10):
        counts = mirror.step_skip_histogram(n_bins)
        assert len(counts) == n_bins
        assert sum(counts) == 12, f"{n_bins} bins did not account for all 12 steps"


def test_step_histogram_separates_a_steady_run_from_a_collapse():
    """A steady 50% run and a run that collapses to 0% must not look alike.

    This is the reason ``step_skip_histogram`` exists: averaged over the run both
    report 0.5, and only the per-step view separates them.
    """
    _, mirror = _pair()
    zero = torch.zeros(1, N)
    for _ in range(8):
        mirror.step(zero, zero)
    # Step 0 has no history and skips nothing; the remaining seven are steady at
    # 1 - 6/128, which lands in the upper bin of a 2-bin histogram.
    steady = mirror.step_skip_histogram(2)
    assert steady == [1, 7]
    mirror.reset_stats()
    for step in range(8):
        loud = torch.full((1, N), 100.0 if step % 2 == 0 else -100.0)
        mirror.step(loud, zero)
    collapsed = mirror.step_skip_histogram(2)
    assert collapsed == [8, 0]
    assert steady != collapsed
    assert mirror.skip_rate() == 0.0, "the collapse averaged out to no skipping"


def test_step_rates_survive_reset_stats_only_where_documented():
    """``reset_stats`` clears the per-step record too.

    Deliberate: a monitor that keeps pre-reset steps would report a rate for a
    window that is not the one being logged. Asserted so the choice is visible.
    """
    _, mirror = _pair()
    mirror.step(torch.zeros(1, N), torch.zeros(1, N))
    assert len(mirror.step_rates) == 1
    mirror.reset_stats()
    assert mirror.step_rates == []


# -- shape and error handling ---------------------------------------------


@pytest.mark.parametrize("bad", [(3, N + 1), (2, 5), (2, 2, N)])
def test_bad_input_shape_is_rejected(bad):
    """A wrong channel count is a hard error, not a silent broadcast."""
    _, mirror = _pair()
    with pytest.raises(ValueError, match="channels"):
        mirror.step(torch.zeros(bad), torch.zeros(bad))


def test_mismatched_gate_magnitude_is_rejected():
    """``gate_magnitude`` must line up with the input, shape for shape."""
    _, mirror = _pair()
    with pytest.raises(ValueError, match="gate_magnitude"):
        mirror.step(torch.zeros(2, N), torch.zeros(3, N))


def test_reference_raises_on_the_same_shapes():
    """The mirror must not be more permissive than the thing it mirrors."""
    gate, mirror = _pair()
    for bad in ((3, N + 1), (2, 5)):
        with pytest.raises(ValueError):
            gate.step(np.zeros(bad, np.float32), np.zeros(bad, np.float32))
        with pytest.raises(ValueError):
            mirror.step(torch.zeros(bad), torch.zeros(bad))
    with pytest.raises(ValueError):
        gate.step(np.zeros((2, N), np.float32), np.zeros((3, N), np.float32))
    with pytest.raises(ValueError):
        mirror.step(torch.zeros(2, N), torch.zeros(3, N))


# -- parameter inventory and reporting -------------------------------------


def test_param_count_matches_the_reference():
    """The two inventories agree, dead 512 values included.

    They have to agree, or a checkpoint round-trip loses state. Quoting the same
    number on both sides is also what keeps the inert-salience cost visible in
    the byte accounting instead of quietly absent.
    """
    gate, mirror = _pair()
    assert gate.param_count() == mirror.param_count() == 3 * N
    names = {name for name, _ in mirror.named_parameters()}
    assert names == {"tau_hi", "tau_lo", "salience"}
    # The runtime state is not parameters, in either.
    buffers = {name for name, _ in mirror.named_buffers()}
    assert buffers == {"cached", "awake", "_quiet", "_has_run"}


def test_training_cannot_mutate_the_reference():
    """The mirror copies the reference's arrays; it does not alias them."""
    gate, mirror = _pair()
    with torch.no_grad():
        mirror.tau_hi.add_(5.0)
    assert gate.tau_hi.tolist() == [1.0] * N


def test_load_from_numpy_restores_a_checkpoint():
    """A reloaded mirror is indistinguishable from a fresh one on the same gate."""
    gate, mirror = _pair()
    rng = np.random.default_rng(61)
    for _ in range(6):
        a = rng.standard_normal((2, N)).astype(np.float32)
        mag = np.abs(rng.standard_normal((2, N)).astype(np.float32))
        gate.step(a, mag)
    with torch.no_grad():
        mirror.tau_hi.mul_(2.0)

    mirror.load_from_numpy(gate)
    assert np.array_equal(mirror.tau_hi.detach().numpy(), gate.tau_hi)
    assert np.array_equal(mirror.cached.numpy(), gate.cached)
    assert np.array_equal(mirror.awake.numpy(), gate.awake)
    assert np.array_equal(mirror._quiet.numpy(), gate._quiet)
    # Deliberate asymmetry, and the one place the two sides are meant to differ.
    # The reference has no load hook, so its lifetime counters keep running from
    # step 0. A resumed run must not re-log the steps that happened before the
    # checkpoint, or the loss and skip curves get a step discontinuity right
    # where the run was resumed. The state is restored; the monitoring restarts.
    assert gate.events["compute"] > 0, "the reference kept its counters"
    assert set(mirror.events.values()) == {0}, "the mirror restarts its monitoring"
    a = rng.standard_normal((2, N)).astype(np.float32)
    mag = np.abs(rng.standard_normal((2, N)).astype(np.float32))
    assert np.array_equal(
        gate.step(a, mag), _hard(mirror.step(torch.tensor(a), torch.tensor(mag)))
    )


def test_repr_reports_the_inert_salience():
    """The summary line says so, so it cannot be forgotten by a reader."""
    _, mirror = _pair()
    text = repr(mirror)
    assert "salience_inert=True" in text
    assert f"n_channels={N}" in text
