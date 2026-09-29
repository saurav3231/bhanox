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

from collections import defaultdict

import numpy as np
import pytest
import torch
from torch import Tensor

import bhanox.core.deltabank as db
import bhanox.train.mirror as tm
from bhanox.config import BhanoxConfig, load_config
from bhanox.model import Bhanox
from bhanox.quant.numerics import INT8_MAX
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


# -- the split-regime contract ------------------------------------------------
#
# There are two regimes, and conflating them is what made the old single number
# wrong rather than merely tight.
#
# *Same codes.* When the reference and the mirror quantise to identical int8
# codes, the integer recurrence is fed bit-identical input, so every int32 state
# is bit-exact and the only difference left is float32 accumulation order. Over
# the grid below that is at most 1.79e-6, which is what TINY_TOL / NANO_TOL
# police. This is the regime the mirror claim is really about.
#
# *A boundary flip.* The two float32 stacks do not accumulate in the same order,
# so they differ by ~1e-6 going into the quantiser. Usually that is invisible,
# but if it lands within 1e-6 of a half-integer the two ``round`` calls snap to
# different codes. One flipped code moves the read by one quantisation step, and
# the logits by up to 5.12e-3. That is not a mirror defect: forcing agreement
# would mean reproducing NumPy's exact accumulator order, i.e. abandoning the
# native torch path. It is a property of comparing two float32 implementations
# across a quantiser boundary.
#
# The flip is also not automatically visible. seq=64 seed=6 flips a ``v`` at token
# 58 -- the write path, at the very end -- and moves the logits by 1.67e-6, inside
# the clean bound. So the second regime genuinely needs a *measured* end-to-end
# limit rather than the head-side 1/127, which bounds the read and says nothing
# about what the mixer, layer norm and unembed do to it afterwards.
#
# The limits below are empirical and scoped to the config and lengths named
# beside them. A flip earlier in a longer sequence has had more steps to
# compound, so widening the grid means re-measuring, not inheriting these.
#
# Every ``*_MEASURED`` constant is the worst value actually observed on the pinned
# grid, quoted to three significant figures, and the tests assert the observations
# never exceed it beyond that rounding. That makes the numbers in ROADMAP.md and
# docs/architecture.md test-enforced rather than prose that quietly goes stale. Each
# ``*_TOL`` is the documented headroom above it. A real regression is orders of
# magnitude, not the fourth digit. The 0.5% slack below is the rounding error of a
# 3-significant-figure constant, which can sit that far under the true observation.
_MEASURED_SLACK = 1.005
#
# Two different grids are in play and their figures must not be mixed:
#   * the pinned grids below, which is what the tests enforce and the docs quote;
#   * a wider 20-seed sweep (100 cases, seq 8-64), run while diagnosing this and
#     not pinned, whose clean worst was 2.03e-6 -- 4.9x inside TINY_TOL rather
#     than 5.6x. The pinned grid is the narrower sample, so its 1.79e-6 is the
#     better number for the same tolerance only by luck of which seeds flipped.
TINY_TOL = 1e-5
NANO_TOL = 1e-4
#: Measured worst over the pinned tiny grid: 1.79e-6, i.e. TINY_TOL / 5.6.
TINY_CLEAN_MEASURED = 1.79e-6
#: Measured worst over the same grid: 5.12e-3. TINY_BOUNDARY_TOL / 2.3.
TINY_BOUNDARY_MEASURED = 5.12e-3
TINY_BOUNDARY_TOL = 1.2e-2
#: Measured worst over the pinned nano grid: 3.81e-6, i.e. NANO_TOL / 26.
NANO_CLEAN_MEASURED = 3.81e-6
#: Measured worst over the same grid: 2.32e-4. NANO_BOUNDARY_TOL / 4.3.
NANO_BOUNDARY_MEASURED = 2.32e-4
NANO_BOUNDARY_TOL = 1e-3

#: (seq, seed, regime). The regime is *measured*, then pinned here so a change
#: in which side of a boundary a seed lands on shows up as a named failure rather
#: than as a tolerance that mysteriously stopped holding.
_TINY_GRID: tuple[tuple[int, int, str], ...] = (
    (8, 0, "same"),
    (8, 6, "same"),
    (8, 11, "same"),
    (16, 0, "same"),
    (16, 6, "same"),
    (16, 11, "flip"),  # (L1, head 0, token 9, q)
    (32, 0, "same"),
    (32, 6, "same"),
    (32, 11, "flip"),
    (48, 0, "flip"),  # (L1, head 0, token 32, k)
    (48, 6, "same"),
    (48, 11, "flip"),
    (64, 0, "flip"),
    (64, 6, "flip"),  # (L1, head 1, token 58, v) -- a late write-path flip
    (64, 11, "flip"),
)

#: (batch, seq, seed, regime), measured the same way as _TINY_GRID. At nano the
#: grid is kept to short windows because past a handful of tokens an occasional
#: seed stops being a question about floating point at all.
_NANO_GRID: tuple[tuple[int, int, int, str], ...] = (
    (1, 1, 0, "same"),
    (1, 1, 1, "same"),
    (1, 1, 2, "same"),
    (1, 8, 0, "same"),
    (1, 8, 1, "same"),
    (1, 8, 2, "same"),
    (2, 4, 0, "same"),
    (2, 4, 1, "same"),
    (2, 4, 2, "same"),
    (4, 4, 0, "flip"),  # one late v flip, still well inside NANO_TOL
    (4, 4, 1, "same"),
    (4, 4, 2, "flip"),  # four flips, 2.32e-4 -- the case the old 1e-4 claim missed
)

_OPS = ("k", "q", "v")


class _Trace:
    """Codes, pre-quantisation values and per-token state, tagged by position.

    Tagging is structural, not positional: each side is keyed by the *actual* head
    object plus a per-head call counter, so ``(layer, head, token, op)`` means
    the same thing on both sides regardless of call order. A positional zip would
    have been shorter and would have silently compared a ``k`` against a ``q`` if
    the two orders ever diverged.
    """

    def __init__(self) -> None:
        self.codes_ref: dict[tuple[int, int, int, str], np.ndarray] = {}
        self.codes_mir: dict[tuple[int, int, int, str], np.ndarray] = {}
        self.pre_ref: dict[tuple[int, int, int, str], np.ndarray] = {}
        self.pre_mir: dict[tuple[int, int, int, str], np.ndarray] = {}
        self.state_ref: dict[tuple[int, int, int], np.ndarray] = {}
        self.state_mir: dict[tuple[int, int, int], np.ndarray] = {}
        self.logits_ref: np.ndarray | None = None
        self.logits_mir: np.ndarray | None = None

    def flips(self) -> list[tuple[int, int, int, str]]:
        """Sites where the two sides chose different int8 codes."""
        return [
            k
            for k in self.codes_ref
            if not np.array_equal(self.codes_ref[k], self.codes_mir[k])
        ]

    def first_flip_token(self) -> int | None:
        toks = [k[2] for k in self.flips()]
        return min(toks) if toks else None

    def max_logit_diff(self) -> float:
        assert self.logits_ref is not None and self.logits_mir is not None
        return float(np.abs(self.logits_ref - self.logits_mir).max())


def _trace(config: BhanoxConfig, ids: np.ndarray) -> _Trace:
    """Run both stacks once, recording codes and state at every quantiser site."""
    tr = _Trace()
    ctx: dict[str, object] = {"site": None, "opn": 0}
    count: dict[tuple, int] = defaultdict(int)
    model = Bhanox(config)
    mirror = BhanoxMirror(Bhanox(config))
    nmap = {
        id(h): (li, hi)
        for li, b in enumerate(model.deltabanks)
        for hi, h in enumerate(b.heads)
    }
    mmap = {
        id(h): (li, hi)
        for li, b in enumerate(mirror.banks)
        for hi, h in enumerate(b.heads)
    }

    oq = db.quantize_activation
    oproj, ocodes = tm.DeltaBankHeadMirror._project, tm.DeltaBankHeadMirror._codes
    ohf, ohm = db.DeltaBankHead.forward, tm.DeltaBankHeadMirror.forward_int

    def n_quant(x, *a, **k):
        out = oq(x, *a, **k)
        li, hi, t = ctx["site"]  # type: ignore[misc]
        op = _OPS[ctx["opn"]]  # type: ignore[index]
        ctx["opn"] = int(ctx["opn"]) + 1  # type: ignore[arg-type]
        tr.codes_ref[(li, hi, t, op)] = np.asarray(out, dtype=np.int32).copy()
        tr.pre_ref[(li, hi, t, op)] = np.asarray(x, dtype=np.float64).copy()
        return out

    def n_head(self, x, *a, **k):
        li, hi = nmap[id(self)]
        count["n", li, hi] += 1
        t = count["n", li, hi] - 1
        ctx["site"], ctx["opn"] = (li, hi, t), 0
        out = ohf(self, x, *a, **k)
        assert ctx["opn"] == 3, "a head must quantise exactly k, q and v"
        tr.state_ref[(li, hi, t)] = np.array(self.state)
        ctx["site"] = None
        return out

    def m_proj(self, x):
        out = oproj(self, x)
        li, hi, t = ctx["site"]  # type: ignore[misc]
        for op, v in zip(_OPS, out, strict=True):
            tr.pre_mir[(li, hi, t, op)] = (
                v.detach().cpu().numpy().astype(np.float64).copy()
            )
        return out

    def m_codes(self, x):
        out = ocodes(self, x)
        li, hi, t = ctx["site"]  # type: ignore[misc]
        for op, v in zip(_OPS, out, strict=True):
            tr.codes_mir[(li, hi, t, op)] = (
                v.detach().cpu().numpy().astype(np.int64).copy()
            )
        return out

    def m_head(self, x, *a, **k):
        li, hi = mmap[id(self)]
        count["m", li, hi] += 1
        t = count["m", li, hi] - 1
        ctx["site"], ctx["opn"] = (li, hi, t), 0
        out = ohm(self, x, *a, **k)
        tr.state_mir[(li, hi, t)] = _np(self.state_int).copy()
        ctx["site"] = None
        return out

    db.quantize_activation, db.DeltaBankHead.forward = n_quant, n_head
    tm.DeltaBankHeadMirror._project, tm.DeltaBankHeadMirror._codes = m_proj, m_codes
    tm.DeltaBankHeadMirror.forward_int = m_head
    try:
        tr.logits_ref = model.forward(ids)
        tr.logits_mir = _np(mirror.forward_int(torch.from_numpy(ids)))
    finally:
        db.quantize_activation, db.DeltaBankHead.forward = oq, ohf
        tm.DeltaBankHeadMirror._project, tm.DeltaBankHeadMirror._codes = oproj, ocodes
        tm.DeltaBankHeadMirror.forward_int = ohm

    # Structural self-checks, so a mis-tagged trace fails here rather than
    # producing a plausible-looking flip count.
    seq = ids.shape[1]
    for li, bank in enumerate(model.deltabanks):
        for hi in range(len(bank.heads)):
            assert count["n", li, hi] == seq, (li, hi, count["n", li, hi], seq)
            assert count["m", li, hi] == seq, (li, hi, count["m", li, hi], seq)
    assert set(tr.codes_ref) == set(
        tr.codes_mir
    ), "the two sides tagged different sites"
    for k in tr.codes_ref:
        assert tr.codes_ref[k].shape == tr.codes_mir[k].shape, k
    return tr


def _assert_flips_are_boundary_straddles(tr: _Trace) -> None:
    """Every code difference must be a verified half-integer straddle.

    This is what licenses the flip counts. If a difference showed up where the
    two float64 products did *not* bracket a half-integer, the cause would not be
    boundary proximity -- it would be a real quantiser or indexing bug, and the
    whole split-regime reading would be wrong.
    """
    for k in tr.flips():
        a = np.minimum(tr.pre_ref[k], tr.pre_mir[k]) * INT8_MAX
        b = np.maximum(tr.pre_ref[k], tr.pre_mir[k]) * INT8_MAX
        crossed = np.floor(b + 0.5) != np.floor(a + 0.5)
        assert np.any(crossed), (
            f"{k}: codes differ but the products [{a.min()}, {b.max()}] never "
            f"bracket a half-integer -- not a boundary flip"
        )


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


@pytest.mark.parametrize(("seq", "seed", "regime"), _TINY_GRID)
def test_tiny_logits_split_by_code_regime(seq: int, seed: int, regime: str):
    """The logit claim, stated as the two things it actually is.

    Same codes -> the clean bound, which is the real claim and holds with room to
    spare. A flipped code -> a wider, separately measured bound, because a single
    int8 step genuinely moves the read by 1/127 and the rest of the network
    amplifies that.

    The old single tolerance demanded the clean bound from cases that had left it,
    which is why two seeds failed. It was not that the mirror got worse; the
    assertion stopped describing the situation it was applied to.
    """
    tr = _trace(TINY, _ids(2000 + seed, 1, seq, TINY.output_vocab))
    _assert_flips_are_boundary_straddles(tr)

    observed = "flip" if tr.flips() else "same"
    assert observed == regime, (
        f"seq={seq} seed={seed} is now '{observed}', was pinned as '{regime}'; "
        f"re-measure the grid and relabel it"
    )

    d = tr.max_logit_diff()
    if regime == "same":
        assert (
            d < TINY_TOL
        ), f"seq={seq} seed={seed} shared every code but moved {d:.3e}"
        assert d <= TINY_CLEAN_MEASURED * _MEASURED_SLACK, (
            f"seq={seq} seed={seed} moved {d:.3e}, worse than the recorded worst "
            f"{TINY_CLEAN_MEASURED:.2e}; re-measure the grid and update it"
        )
    else:
        assert (
            d < TINY_BOUNDARY_TOL
        ), f"seq={seq} seed={seed} flipped at {tr.flips()[:2]} and moved {d:.3e}"
        assert d <= TINY_BOUNDARY_MEASURED * _MEASURED_SLACK, (
            f"seq={seq} seed={seed} moved {d:.3e}, worse than the recorded worst "
            f"{TINY_BOUNDARY_MEASURED:.2e}; re-measure the grid and update it"
        )


@pytest.mark.parametrize(("batch", "seq", "seed", "regime"), _NANO_GRID)
def test_nano_logits_split_by_code_regime(batch: int, seq: int, seed: int, regime: str):
    """The same split at the real config, on the short windows where it holds.

    At nano the drift has four layers to accumulate through, so the window is kept
    short. ``test_the_gate_has_no_dead_band`` pins why that is an architecture
    property rather than a test artefact.
    """
    tr = _trace(load_config("nano"), _ids(3000 + seed, batch, seq, 256))
    _assert_flips_are_boundary_straddles(tr)

    observed = "flip" if tr.flips() else "same"
    assert (
        observed == regime
    ), f"b={batch} t={seq} seed={seed} is now '{observed}', pinned as '{regime}'"

    d = tr.max_logit_diff()
    if regime == "same":
        assert (
            d < NANO_TOL
        ), f"b={batch} t={seq} seed={seed} shared every code but moved {d:.3e}"
        assert d <= NANO_CLEAN_MEASURED * _MEASURED_SLACK, (
            f"b={batch} t={seq} seed={seed} moved {d:.3e}, worse than the recorded "
            f"worst {NANO_CLEAN_MEASURED:.2e}; re-measure the grid and update it"
        )
    else:
        assert (
            d < NANO_BOUNDARY_TOL
        ), f"b={batch} t={seq} seed={seed} flipped at {tr.flips()[:2]}, moved {d:.3e}"
        assert d <= NANO_BOUNDARY_MEASURED * _MEASURED_SLACK, (
            f"b={batch} t={seq} seed={seed} moved {d:.3e}, worse than the recorded "
            f"worst {NANO_BOUNDARY_MEASURED:.2e}; re-measure the grid and update it"
        )


@pytest.mark.parametrize(("seq", "seed", "regime"), _TINY_GRID)
def test_integer_state_is_bit_exact_exactly_as_far_as_the_codes_agree(
    seq: int, seed: int, regime: str
):
    """Where the codes match, the int32 state is bit-exact -- not approximately.

    This is the sharp half of the contract and it needs no tolerance at all. Given
    identical codes and an identical incoming state, the integer recurrence is
    deterministic, so any difference here would be a genuine bug rather than float
    noise. The claim stops at the first code divergence *anywhere*, because a flip
    in one head changes the layer's read, which changes the next head's input.
    """
    tr = _trace(TINY, _ids(2000 + seed, 1, seq, TINY.output_vocab))
    first = tr.first_flip_token()

    for (li, hi, t), ref_state in tr.state_ref.items():
        if first is not None and t >= first:
            continue
        assert np.array_equal(ref_state, tr.state_mir[(li, hi, t)]), (
            f"L{li} head{hi} token {t}: state differs at "
            f"{ref_state.nonzero()[:4].tolist()}, first code flip at token {first}"
        )

    if regime == "same":
        # No flip anywhere, so the claim above was unconditional: assert that
        # rather than leaving it vacuous.
        assert first is None
        assert len(tr.state_ref) == 2 * 2 * seq


def test_the_documented_headroom_ratios_are_the_arithmetic_they_claim_to_be():
    """The "5.6x inside 1e-5" style figures in the docs are checked, not asserted.

    These ratios are quoted in ROADMAP.md, in docs/architecture.md and in the
    comments above, and a hand-written multiple is exactly the kind of number
    that goes stale when a constant moves. Recomputing them here means an edit to
    a tolerance that invalidates a documented figure fails here instead.

    It also pins *which* measurement each ratio comes from. The pinned tiny grid
    gives 1.79e-6 and so 5.6x, while the wider unpinned 20-seed sweep gave 2.03e-6
    and so 4.9x. Both are true measurements of different samples; mixing them is
    how a document ends up claiming 5.6x beside a 2.03e-6 figure.
    """
    for measured, limit, claimed, what in (
        (TINY_CLEAN_MEASURED, TINY_TOL, 5.6, "tiny same-codes"),
        (TINY_BOUNDARY_MEASURED, TINY_BOUNDARY_TOL, 2.3, "tiny boundary"),
        (NANO_CLEAN_MEASURED, NANO_TOL, 26.0, "nano same-codes"),
        (NANO_BOUNDARY_MEASURED, NANO_BOUNDARY_TOL, 4.3, "nano boundary"),
    ):
        ratio = limit / measured
        assert ratio == pytest.approx(claimed, rel=0.02), (
            f"{what}: docs say {claimed}x, but "
            f"{limit:.3g} / {measured:.3g} = {ratio:.2f}"
        )
        assert ratio > 1.0, f"{what}: the limit is inside the measurement"

    # The wider sweep, kept here so the 4.9x figure has a stated provenance and
    # is not silently re-derived from the pinned grid's 5.6x.
    assert pytest.approx(4.9, rel=0.02) == TINY_TOL / 2.03e-6


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
