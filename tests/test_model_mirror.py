"""The assembled model mirror: does the reference's block order survive assembly?

Every other mirror test checks one component. Those components can all be
correct while the assembled model is wrong, and the ways it goes wrong are
specific to assembly:

- the three pieces of a block applied in the wrong order,
- a mask applied to the per-head read (``n_heads * d_v`` wide) rather than to
  the layer output (``d_model`` wide), which at nano silently rejects 384 of
  the 128 values it is handed.

On the second one: masking the residual instead of the contribution is *not* on
the list, because `x + where(keep, s, 0)` and `where(keep, x + s, x)` are the
same number. That one is worth having ruled out explicitly -- it looks like the
obvious thing to get wrong, and it costs nothing to check.

So this file is mostly about the reference's *order*, and the tolerances are the
measured ones from the module docstring rather than the ones a first guess would
pick.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from bhanox.config import BhanoxConfig, load_config
from bhanox.model import Bhanox
from bhanox.train.bookend_mirror import layer_norm_torch
from bhanox.train.model_mirror import BhanoxMirror

# Two layers, not four. Depth is what makes float drift accumulate, and on a
# two-layer config the integer state still agrees exactly out to 64 tokens, which
# is the claim worth testing tightly. nano gets its own short-window test below.
TINY = BhanoxConfig(
    name="t",
    d_model=32,
    d_k=8,
    d_v=8,
    d_expert=16,
    n_heads=2,
    n_layers=2,
    n_experts=4,
    n_shared_experts=1,
    top_k=2,
    output_vocab=32,
    max_context=64,
    pool_size=1024,
    vocab_table=128,
    n_hashes=2,
    seed=7,
)

# Worst observed over 120 trials up to 64 tokens: 8.4e-6. The documented
# LOGIT_TOL is 1e-4, so this holds the mirror to something 12x tighter than the
# claim being made about it.
TINY_TOL = 1e-5
NANO_TOL = 1e-4


def _np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _ids(seed: int, batch: int, seq: int, vocab: int) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).integers(0, vocab, size=(batch, seq)))


def _states(model: Bhanox) -> list[np.ndarray]:
    return [h.state for b in model.deltabanks for h in b.heads]


def _mirror_states(mirror: BhanoxMirror) -> list[np.ndarray]:
    return [_np(h.state_int) for b in mirror.banks for h in b.heads]


# -- the front end ------------------------------------------------------------


def test_front_end_is_bit_exact_not_merely_close():
    """The hashed sum is an ordered accumulation, so it is exact.

    Worth its own test because the fix is invisible. ``torch.einsum`` and
    ``np.einsum`` reduce the same contraction in different orders and disagree by
    one ULP, so the natural-looking ``torch.einsum`` version passes every
    tolerance-based test in the suite while quietly making the whole integer path
    inexact -- and "inexact" here feeds a recurrent int8 state, where inexactness
    compounds rather than averaging out.
    """
    model = Bhanox(TINY)
    mirror = BhanoxMirror(model)
    ids = _ids(5, 4, 9, TINY.output_vocab)
    assert np.array_equal(model.embed(ids), _np(mirror.embed(ids)))


# -- integer state ------------------------------------------------------------


@pytest.mark.parametrize("seq", [4, 8, 16, 32])
def test_integer_state_is_bit_exact(seq: int):
    """The state is the model. Compare with ``array_equal``, not a tolerance.

    Four seeds, because "the state agrees" is a claim about a trajectory and one
    sequence is not a trajectory. The logits are checked separately and are
    allowed to differ -- see the module docstring for why those two claims have
    different strength.
    """
    for seed in range(4):
        ids = _ids(2000 + seed, 1, seq, TINY.output_vocab)
        # Two independent instances, because the config seed fixes the weights
        # and building the mirror from an already-advanced reference would hand
        # it a state it is supposed to reproduce.
        ref = Bhanox(TINY)
        ref.forward(ids)
        mir = BhanoxMirror(Bhanox(TINY))
        mir.forward_int(ids)
        assert all(
            np.array_equal(a, b)
            for a, b in zip(_mirror_states(mir), _states(ref), strict=False)
        ), f"seq={seq} seed={seed}: integer state diverged"


# -- a state that is not zero --------------------------------------------------


@pytest.mark.parametrize("seq", [8, 16, 32])
def test_a_seeded_non_zero_state_is_reproduced(seq: int):
    """Every other test starts from zero, which is the easy case.

    A fresh model has an all-zero int8 state, and a mirror that simply kept its
    own state at zero would pass all of them. The state a real run carries is the
    one left by the tokens before it, so the mirror has to reproduce a state it
    did not choose for itself: warm the reference up, hand the mirror that exact
    state, and check it carries it forward identically.

    The non-zero assertion is the part that keeps this honest -- if the warmup
    ever stopped writing state, the test would quietly become a second copy of the
    zero-state one.
    """
    vocab = TINY.output_vocab
    for seed in range(3):
        warm = _ids(4000 + seed, 1, 12, vocab)
        ids = _ids(4100 + seed, 1, seq, vocab)

        ref = Bhanox(TINY)
        ref.forward(warm)
        warm_states = _states(ref)
        assert any(np.any(s) for s in warm_states), (
            f"seed={seed}: warmup left the state at zero, so this is not testing "
            "a non-zero state"
        )
        # Built from the advanced reference, so the mirror inherits its state
        # rather than computing it. Only the continuation is compared below.
        mir = BhanoxMirror(ref)
        assert all(
            np.array_equal(a, b)
            for a, b in zip(_mirror_states(mir), warm_states, strict=False)
        ), f"seed={seed}: the mirror did not inherit the state it was handed"

        r_logits = ref.forward(ids)
        m_logits = _np(mir.forward_int(ids))
        assert all(
            np.array_equal(a, b)
            for a, b in zip(_mirror_states(mir), _states(ref), strict=False)
        ), f"seq={seq} seed={seed}: integer state diverged from a non-zero start"
        assert (
            np.abs(r_logits - m_logits).max() < TINY_TOL
        ), f"seq={seq} seed={seed}: max|diff|={np.abs(r_logits - m_logits).max():.3e}"


def test_batched_rows_stay_independent():
    """Sample ``b`` reads only what sample ``b`` wrote.

    Compared against three separate single-sample forwards, which is the property
    that makes a batched loss measure the right thing. It was not true before M2
    -- the state was a single instance, so ``forward([a, b])`` equalled
    ``forward(a ++ b)`` and a batched loss was quietly training on cross-sample
    leakage.
    """
    ids = _ids(11, 3, 6, TINY.output_vocab)
    batched = BhanoxMirror(Bhanox(TINY))
    batched.forward_int(ids)
    rows = _mirror_states(batched)
    for b in range(ids.shape[0]):
        single = BhanoxMirror(Bhanox(TINY))
        single.forward_int(ids[b])
        assert all(
            np.array_equal(s[0], r[b])
            for s, r in zip(_mirror_states(single), rows, strict=False)
        ), f"row {b} is not independent of its batch"


# -- logits -------------------------------------------------------------------


@pytest.mark.parametrize("seq", [4, 16, 32, 64])
def test_logits_agree_at_the_measured_tolerance(seq: int):
    for seed in range(3):
        ids = _ids(2000 + seed, 1, seq, TINY.output_vocab)
        ref = Bhanox(TINY).forward(ids)
        got = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))
        assert (
            np.abs(ref - got).max() < TINY_TOL
        ), f"seq={seq} seed={seed}: max|diff|={np.abs(ref - got).max():.3e}"


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (2, 4), (4, 4)])
def test_nano_short_windows_meet_the_documented_claim(shape: tuple[int, int]):
    """The 1e-4 claim, at the real config, in the window where it holds.

    Deliberately short. At nano the drift has four layers to accumulate through,
    so past a handful of tokens an occasional seed crosses a gate threshold and
    the comparison stops being about floating point at all. That is documented
    rather than tested away, because it is a property of the architecture --
    ``test_the_gate_has_no_dead_band`` is the test that pins why.
    """
    b, t = shape
    for seed in range(3):
        ids = _ids(3000 + seed, b, t, 256)
        ref = Bhanox(load_config("nano")).forward(ids)
        got = _np(BhanoxMirror(Bhanox(load_config("nano"))).forward_int(ids))
        assert np.abs(ref - got).max() < NANO_TOL


# -- the converse -------------------------------------------------------------


def test_a_wrong_state_cannot_hide_inside_the_tolerance():
    """The converse half of the claim.

    A float tolerance is only meaningful if a genuinely wrong state fails it. A
    state populated from a *different* token sequence is a realistic error -- a
    mis-restored checkpoint, or a batch that got its state rows misaligned -- and
    it has to be far outside the band the good cases occupy. Without this, a
    mirror that dropped a whole DeltaBank would pass every tolerance test above,
    because the tolerance is loose enough to absorb the float noise that was
    there in the first place.
    """
    ids = _ids(2000, 1, 8, TINY.output_vocab)
    good = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))

    polluted = Bhanox(TINY)
    polluted.forward(_ids(999, 1, 8, TINY.output_vocab))
    bad = _np(BhanoxMirror(polluted).forward_int(ids))

    assert (
        np.abs(good - bad).max() > 100 * TINY_TOL
    ), "a wrong state is inside the tolerance, so the tolerance proves nothing"


def test_a_one_channel_state_corruption_exceeds_the_tolerance():
    """One element of one head's state, not the whole thing.

    The strongest version of the test above, and the one that would catch a real
    bug: if a single int8 element changing by 64 were invisible, then the gate
    is attenuating state so aggressively that the memory is barely load-bearing.
    That would be a much more interesting finding than a tolerance that needed
    widening, so the test would rather fail than accommodate it.
    """
    ids = _ids(2000, 1, 8, TINY.output_vocab)
    good = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))

    nudged = Bhanox(TINY)
    nudged.forward(ids)  # populate the state
    nudged.deltabanks[0].heads[0].state[0, 0, 0] += 64
    bad = _np(BhanoxMirror(nudged).forward_int(ids))

    assert np.abs(good - bad).max() > 100 * TINY_TOL


# -- ordering, which is the actual point of this file -----------------------


def test_layer_norm_comes_after_the_mixer_addition():
    """Norm placement is order-dependent, unlike the mask and the mixer.

    Two placements that look interchangeable and are not: normalising before the
    mixer's contribution is added leaves the mixer's output unnormalised in the
    residual stream, and the next layer sees a differently-scaled input.

    Worth the test because of the two assembly "bugs" next to it that are *not*
    bugs. ``x + where(keep, s, 0)`` and ``where(keep, x + s, x)`` are the same
    value, and the mixer is token-local by construction, so moving it inside the
    time loop is also the same value. Both were tests in this file, and both were
    removed once they were shown to be testing algebra rather than the mirror.
    """
    ids = _ids(7, 2, 8, TINY.output_vocab)
    good = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))

    with torch.no_grad():
        mir = BhanoxMirror(Bhanox(TINY))
        x = mir.embed(ids)
        for bank, mixer, gate in zip(mir.banks, mir.mixers, mir.gates, strict=True):
            out = torch.zeros_like(x)
            for t in range(x.shape[1]):
                step_out = bank.forward_int(x[:, t])
                out[:, t] = torch.where(
                    gate(step_out, step_out.abs()) > 0.5,
                    step_out,
                    torch.zeros_like(step_out),
                )
            x = layer_norm_torch(x + out)
            x = x + mixer(x)  # norm already spent; the mixer lands unnormalised
        early_norm = _np(mir.unembed(x))

    assert np.abs(good - early_norm).max() > TINY_TOL, (
        "normalising before the mixer gives the same answer, so this test cannot "
        "tell the two placements apart"
    )


def test_layers_run_in_order():
    """The block stack is sequential, and the layer index is load-bearing.

    The model seeds each layer's weights from a stream keyed on the layer index,
    so a mirror that reversed the stack would not be a reordering of the same
    computation -- it would be a different model. Cheap to check, and it is the
    kind of thing a ``zip`` over three parallel lists gets wrong.
    """
    ids = _ids(13, 2, 6, TINY.output_vocab)
    good = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))

    with torch.no_grad():
        mir = BhanoxMirror(Bhanox(TINY))
        x = mir.embed(ids)
        for bank, mixer, gate in zip(
            reversed(mir.banks), reversed(mir.mixers), reversed(mir.gates), strict=True
        ):
            out = torch.zeros_like(x)
            for t in range(x.shape[1]):
                step_out = bank.forward_int(x[:, t])
                out[:, t] = torch.where(
                    gate(step_out, step_out.abs()) > 0.5,
                    step_out,
                    torch.zeros_like(step_out),
                )
            x = layer_norm_torch(x + out + mixer(x + out))
        reversed_stack = _np(mir.unembed(x))

    assert np.abs(good - reversed_stack).max() > TINY_TOL


def test_block_order_is_memory_then_mixer_then_norm():
    """Swapping the memory and the mixer changes the answer.

    The two residual additions are not commutative, so the wrong order moves by
    far more than the tolerance. Asserting that the swapped version differs is
    what makes the correct one mean something.
    """
    ids = _ids(9, 2, 6, TINY.output_vocab)
    good = _np(BhanoxMirror(Bhanox(TINY)).forward_int(ids))

    with torch.no_grad():
        mir = BhanoxMirror(Bhanox(TINY))
        x = mir.embed(ids)
        for bank, mixer, gate in zip(mir.banks, mir.mixers, mir.gates, strict=True):
            x = x + mixer(x)
            out = torch.zeros_like(x)
            for t in range(x.shape[1]):
                step_out = bank.forward_int(x[:, t])
                out[:, t] = torch.where(
                    gate(step_out, step_out.abs()) > 0.5,
                    step_out,
                    torch.zeros_like(step_out),
                )
            x = layer_norm_torch(x + out)
        swapped = _np(mir.unembed(x))

    assert np.abs(good - swapped).max() > TINY_TOL


# -- the documented fragility -------------------------------------------------


def test_the_gate_has_no_dead_band():
    """Why the full-model tolerances are floats, pinned as a mechanism.

    No mirror and no torch. A channel's memory contribution is either added to
    the residual stream or it is not, with nothing in between, so a difference of
    one ULP in the float read is enough to switch a whole channel on. Here the
    channel is asleep and ``delta`` sits exactly on ``tau_hi``, so a single ULP of
    drift decides whether the governor wakes it at all.

    This is the amplifier behind the module docstring's drift measurements. It is
    a property of the frozen architecture, not of any implementation: two float
    libraries that disagree by one ULP will disagree about the governor's
    decisions, and neither is wrong.
    """
    from bhanox.governor.pulsegate import PulseGate

    def decide(nudge: bool) -> np.ndarray:
        gate = PulseGate(8)
        a = np.full((1, 8), 2.0, dtype=np.float32)
        gate.step(a, np.abs(a))  # so _has_run is set
        gate.awake[:] = False  # asleep, so the wake is what decides
        gate.cached = (a - gate.tau_hi).astype(np.float32)  # delta == tau_hi
        probe = np.nextafter(a, np.float32(np.inf)) if nudge else a
        return np.asarray(gate.step(probe, np.abs(probe)))

    at_threshold, one_ulp_up = decide(False), decide(True)
    assert int((at_threshold != one_ulp_up).sum()) > 0, (
        "a 1-ULP change no longer flips a governor decision. That is good news "
        "for reproducibility and means the full-model tolerances can be tightened "
        "-- update the docstring and TINY_TOL with it."
    )


# -- inventory ----------------------------------------------------------------


def test_param_count_matches_the_reference():
    """They are equal, including the dead ``salience`` values.

    Worth pinning because it is easy to get "more correct" in the wrong
    direction: salience is inert, so an argument exists for excluding it. The
    mirror keeps it as a parameter anyway, so that the two inventories agree and
    a checkpoint round-trips. The right place to act on inertness is the
    optimizer, not the parameter list.
    """
    model = Bhanox(TINY)
    assert BhanoxMirror(model).param_count() == model.param_count()


def test_salience_is_a_parameter_and_inert():
    """Both halves, because either alone is misleading.

    It is an ``nn.Parameter`` (see above) *and* it can never receive a gradient
    or change value. A trainer that swept all parameters would put it in the
    optimizer; the exclusion belongs there, and this is the test that says so.
    """
    mirror = BhanoxMirror(Bhanox(TINY))
    names = [n for n, _ in mirror.named_parameters() if n.endswith("salience")]
    assert names, "salience should still be a parameter, for checkpoint parity"

    # The training path, not ``forward_int``: that one is under ``no_grad``, so
    # asking it for a gradient would raise rather than demonstrate anything.
    logits, _ = mirror.step(_ids(1, 2, 4, TINY.output_vocab))
    before = [
        p.detach().clone() for n, p in mirror.named_parameters() if "salience" in n
    ]
    logits[0].sum().backward()
    for n, p in mirror.named_parameters():
        if "salience" in n:
            assert (
                p.grad is None or torch.count_nonzero(p.grad) == 0
            ), f"{n} got a gradient"
    for (_, p), b in zip(
        [(n, p) for n, p in mirror.named_parameters() if "salience" in n],
        before,
        strict=False,
    ):
        assert torch.equal(p.detach(), b), "salience changed value"


def test_state_round_trips_as_a_copy():
    """``to_numpy_state`` is what the checkpoint writes, so it has to be a copy.

    Not a view. A checkpoint sharing memory with a live model is correct until
    the next token, at which point the saved state has silently moved -- and
    since it is written from a background thread while training continues, the
    failure would be a rare wrong-resume rather than a crash.
    """
    mir = BhanoxMirror(Bhanox(TINY))
    mir.forward_int(_ids(3, 1, 4, TINY.output_vocab))
    before = [a.copy() for group in mir.to_numpy_state() for a in group]
    mir.forward_int(_ids(4, 1, 4, TINY.output_vocab))
    after = [a for group in mir.to_numpy_state() for a in group]
    assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=False))
