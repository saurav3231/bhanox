"""Does the full model's gradient reach every parameter, and is it the right one?

Two separate claims, and the second needs a caveat that took a while to find, so
they are tested separately rather than as one "gradients work" file.

**Reachability.** A trainer with an unreachable parameter does not crash. It
optimises everything else, reports a falling loss, and quietly ships a model
with a frozen piece. So the first set of tests asserts that every parameter
tensor receives a gradient, by structural group, with the one documented
exception.

**Correctness.** Checked with :mod:`bhanox.train.gradcheck`, which shares no code
with autodiff. The interesting part is *where it cannot be applied*, and the
answer is structural rather than a limitation of the harness:

    the loss is smooth in P only if nothing downstream of P performs an int
    state write or a gate decision.

The last state write is inside the last layer's bank, so the provably smooth
parameters are the last bank's read-out and bypass, the last mixer, and the
unembed -- 40 probes, no mismatches. Everything earlier has a downstream
staircase, and finite differences there are unreliable rather than wrong: at the
default step the edges are crossed too rarely to see (0 of 24), and at a
ten-times-wider step the probes that disagree are exactly the ones whose +/- pair
crossed a boundary. That containment is asserted directly, because a
mismatch-rate cap is also satisfied by a broken harness.

The gate adds a second, separate effect. The mask is a function of the very
parameters whose gradient is being checked -- it thresholds ``abs(step_out)``,
and ``step_out`` ends in ``W_o`` -- so the true loss has a ``d(mask)/d(W_o)``
term that the straight-through estimator omits on purpose. Note this is a
surrogate term, not a discontinuity: the live mask skips a measured 30% of
channels but does not flip under the perturbation at any step size tried, so the
loss is locally constant in that respect and the estimator simply over-counts.
Comparing live against frozen without accounting for it produces confident
nonsense: the first version of this file reported 32 mismatches from a harness
bug, and the second reported 32 more that were a real and intended difference.
Both are pinned as tests now, because a gradient harness that cannot demonstrate
it detects agreement is not evidence of anything -- see the positive control at
the bottom.

The harness also has a silent trap worth naming. ``step`` advances the int32
recurrent state in place, so a loss evaluated twice from a stale state is a
different function. There is a test for exactly that, and it is the precondition
for everything else here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from bhanox.config import BhanoxConfig, load_config
from bhanox.model import Bhanox
from bhanox.train.gradcheck import DEFAULT_STEP, check_tensor, noise_floor
from bhanox.train.model_mirror import BhanoxMirror

#: Same two-layer config as the assembly tests: deep enough for float drift to
#: accumulate, short enough that the integer state still agrees exactly.
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

#: Long enough for a gate to have run at least once -- the mask is the constant
#: 1.0 until a sample has run, so a shorter sequence would not exercise the
#: decision this file is mostly about. Kept short because every finite-difference
#: probe costs two forward passes and the whole file runs in CI on two versions.
SEQ = 6

#: Probes per parameter. Enough that "no mismatches" is not a one-in-ten
#: coincidence, few enough to keep the file inside the runtime budget.
PROBES = 6

#: The one group that is expected to receive no gradient, and the reason is
#: documented rather than accidental -- see the ROADMAP and README.
INERT = "salience"

#: Provably smooth: nothing downstream of these performs a state write or a gate
#: decision. The last layer's bank is the last place either happens.
SMOOTH = [
    "banks.1.W_o",
    "banks.1.G",
    "mixers.1.W1",
    "mixers.1.W2",
    "mixers.1.E",
    "mixers.1.b",
    "unembed.output",
]

#: Smooth locally, staircase downstream, so a *minority* of probes disagree.
#: A genuinely wrong gradient in these would disagree on nearly all of them,
#: which is what makes the minority cap a real test rather than a shrug.
STAIRCASE = [
    "banks.0.W_o",
    "banks.0.G",
    "mixers.0.W1",
    "mixers.0.W2",
]


# -- harness -------------------------------------------------------------------


def _ids(vocab: int, seed: int) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).integers(0, vocab, size=(1, SEQ)))


def _build(config: BhanoxConfig = TINY, *, freeze: bool = False) -> BhanoxMirror:
    mir = BhanoxMirror(Bhanox(config))
    if freeze:
        _freeze_masks(mir)
    return mir


def _capture(mir: BhanoxMirror) -> dict:
    """Every piece of state ``step`` advances, captured by dtype and by role.

    The dtype filter is the whole point. ``state_int`` is int32, so a snapshot
    written as "the float buffers" drops the entire recurrent state without
    complaining -- see :func:`test_a_float_only_snapshot_drops_the_state`.
    """
    return {
        "buffers": {n: b.detach().clone() for n, b in mir.named_buffers()},
        "has_run": [g._has_run.detach().clone() for g in mir.gates],
    }


def _restore(mir: BhanoxMirror, saved: dict) -> None:
    bufs = dict(mir.named_buffers())
    with torch.no_grad():
        for name, val in saved["buffers"].items():
            bufs[name].copy_(val)
    for gate, val in zip(mir.gates, saved["has_run"], strict=True):
        gate._has_run.copy_(val)


def _heads(mir: BhanoxMirror) -> list:
    return [h for b in mir.banks for h in b.heads]


def _loss(mir: BhanoxMirror, ids: np.ndarray, target: Tensor) -> Tensor:
    """Cross-entropy through the training path.

    ``train=False``, ``reanchor=False`` and an identity quantiser, all for one
    reason: anything that mutates state between the two evaluations of a central
    difference contaminates the quotient. ``train=True`` applies the MoE
    load-balancing update as a side effect, so two evaluations of *identical*
    weights would differ and every difference below would be meaningless.

    The vocabulary is read from the mirror rather than from a module constant, so
    that the same helper works at nano.
    """
    logits, _ = mir.step(
        ids,
        shadows=None,
        train=False,
        quantize=lambda t: t,
        reanchor=False,
    )
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, mir.config.output_vocab), target
    )


def _freeze_masks(mir: BhanoxMirror) -> None:
    """Force every gate to always compute, removing the discrete decision.

    The forward *value* keeps its meaning -- an always-computing channel
    contributes -- and ``d(mask)/d(param)`` becomes exactly zero. What remains is
    the derivative of a smooth function, which is what the estimator claims to be
    the gradient of.
    """
    for gate in mir.gates:
        gate.forward = (  # type: ignore[method-assign]
            lambda _a, _b, _one=torch.ones(1): _one  # noqa: B008
        )


def _probes(
    name: str,
    *,
    freeze: bool = True,
    count: int = PROBES,
    step: float = DEFAULT_STEP,
) -> list:
    """Central-difference ``name`` against its analytic gradient."""
    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))
    mir = _build(freeze=freeze)
    saved = _capture(mir)
    params = dict(mir.named_parameters())
    _restore(mir, saved)
    _loss(mir, ids, target).backward()
    grad = params[name].grad.detach().cpu().numpy().copy()
    arr = params[name].detach().cpu().numpy().copy()

    def loss(a: np.ndarray = arr, key: str = name) -> float:
        _restore(mir, saved)
        with torch.no_grad():
            params[key].copy_(torch.from_numpy(a))
        with torch.no_grad():
            return float(_loss(mir, ids, target))

    return check_tensor(name, arr, grad, loss, count=count, step=step)


def _verdicts(results: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.verdict] = out.get(r.verdict, 0) + 1
    return out


# -- the harness itself --------------------------------------------------------


def test_the_loss_is_a_pure_function_of_the_parameters():
    """Same state restored, same loss. The precondition for every check below.

    Worth its own test because the failure is silent and looks like data. A
    harness that snapshotted only the float buffers dropped the int32 state, so
    each evaluation started further along the recurrence than the last, and the
    quotients it reported were state drift dressed up as gradients -- plausible
    numbers, entirely wrong, and all 32 of them "mismatches".
    """
    mir = _build()
    saved = _capture(mir)
    ids, target = _ids(TINY.output_vocab, 0), torch.from_numpy(
        _ids(TINY.output_vocab, 1).reshape(-1)
    )
    values = []
    for _ in range(4):
        _restore(mir, saved)
        with torch.no_grad():
            values.append(float(_loss(mir, ids, target)))
    assert (
        len(set(values)) == 1
    ), f"loss is not idempotent under state restore: {values}"


def test_restore_really_rewinds_the_integer_state():
    """The specific thing the previous harness got wrong."""
    mir = _build()
    saved = _capture(mir)
    with torch.no_grad():
        _loss(
            mir,
            _ids(TINY.output_vocab, 0),
            torch.from_numpy(_ids(TINY.output_vocab, 1).reshape(-1)),
        )
    assert any(
        not torch.equal(h.state_int, s)
        for h, s in zip(
            _heads(mir),
            (
                saved["buffers"][f"banks.{i}.heads.{j}.state_int"]
                for i in range(TINY.n_layers)
                for j in range(TINY.n_heads)
            ),
            strict=True,
        )
    ), "step did not advance the state, so this test would prove nothing"
    _restore(mir, saved)
    assert all(
        torch.equal(h.state_int, s)
        for h, s in zip(
            _heads(mir),
            (
                saved["buffers"][f"banks.{i}.heads.{j}.state_int"]
                for i in range(TINY.n_layers)
                for j in range(TINY.n_heads)
            ),
            strict=True,
        )
    ), "restore did not put the integer state back"


def test_a_float_only_snapshot_drops_the_state():
    """Why :func:`_capture` filters on nothing in particular.

    ``state_int`` is int32, and it *is* a registered buffer -- so a snapshot that
    says "the float buffers" is not a simplification, it is a bug that drops the
    whole recurrent state. Pinned so that "just take the floats" cannot be
    reintroduced as a tidy-up.
    """
    mir = _build()
    state_names = [n for n, t in mir.named_buffers() if n.endswith("state_int")]
    assert len(state_names) == TINY.n_layers * TINY.n_heads
    assert all(dict(mir.named_buffers())[n].dtype == torch.int32 for n in state_names)
    float_only = {n: t for n, t in mir.named_buffers() if t.dtype.is_floating_point}
    missing = set(state_names) - set(float_only)
    assert missing, (
        "the int32 state is now float, so _capture no longer needs the dtype "
        "care it documents; update the test and the comment together"
    )


# -- reachability --------------------------------------------------------------


def _gradient_report(mir: BhanoxMirror, ids: np.ndarray, target: Tensor) -> dict:
    mir.zero_grad(set_to_none=True)
    _loss(mir, ids, target).backward()
    out: dict[str, str] = {}
    for name, p in mir.named_parameters():
        if p.grad is None:
            out[name] = "none"
        elif float(p.grad.abs().max()) == 0.0:
            out[name] = "zero"
        else:
            out[name] = "ok"
    return out


def _group_of(name: str) -> str:
    if name.startswith("embed."):
        return "embed"
    if name.startswith("unembed."):
        return "unembed"
    if name.startswith("mixers."):
        return "mixer"
    if ".heads." in name:
        return "bank.head"
    if name.endswith(".W_o"):
        return "bank.W_o"
    if name.endswith(".G"):
        return "bank.G"
    if "tau_" in name:
        return "gate.threshold"
    if INERT in name:
        return "gate.inert"
    raise AssertionError(
        f"unclassified parameter {name!r}; add it or this test is a lie"
    )


def _reachable(config: BhanoxConfig, seq: int) -> dict[str, str]:
    vocab = config.output_vocab
    mir = BhanoxMirror(Bhanox(config))
    ids = np.asarray(np.random.default_rng(0).integers(0, vocab, size=(1, seq)))
    target = torch.from_numpy(
        np.asarray(np.random.default_rng(1).integers(0, vocab, size=(1, seq)))
    ).reshape(-1)
    return _gradient_report(mir, ids, target)


def test_every_parameter_group_receives_a_gradient():
    """The load-bearing claim: nothing is silently frozen.

    A parameter that never receives a gradient is not a crash. The loss falls,
    the trainer reports progress, and one part of the model stays at its
    initialisation forever. So this asserts on *every* tensor, and the expected
    structure is written out group by group rather than read off the result.
    """
    report = _reachable(TINY, SEQ)
    groups: dict[str, list[str]] = {}
    for name, _verdict in report.items():
        groups.setdefault(_group_of(name), []).append(name)

    assert set(groups) == {
        "embed",
        "unembed",
        "mixer",
        "bank.head",
        "bank.W_o",
        "bank.G",
        "gate.threshold",
        "gate.inert",
    }, f"parameter inventory changed; groups seen: {sorted(groups)}"

    for group, names in groups.items():
        if group == "gate.inert":
            continue
        dead = [n for n in names if report[n] != "ok"]
        assert not dead, f"group {group} has parameters with no gradient: {dead}"


def test_the_inert_group_is_exactly_the_documented_one():
    """The exception is as narrow as the documentation claims.

    ``salience`` is reported as never receiving a gradient, so the set of
    parameters that receive none should be exactly it -- no wider. A test that
    only asserted "some parameters are inert" would pass just as happily if the
    whole mixer had joined them.
    """
    report = _reachable(TINY, SEQ)
    dead = {n for n, v in report.items() if v != "ok"}
    assert dead, "no inert parameters at all -- did salience start receiving one?"
    assert all(INERT in n for n in dead), f"unexpected dead parameters: {sorted(dead)}"
    assert dead == {n for n in report if INERT in n}
    assert not any(
        v == "zero" for v in report.values()
    ), "a zero-but-present gradient is a third category and is not accounted for"


def test_reachability_holds_at_the_real_config():
    """The same claim at nano, where shapes, depth and gate all differ.

    Guards against the tiny config passing for a structural reason nano does not
    share -- four layers, 16 channels, more thresholds to cross.
    """
    nano = load_config("nano")
    report = _reachable(nano, 4)
    dead = {n for n, v in report.items() if v != "ok"}
    assert all(INERT in n for n in dead), f"unexpected dead parameters: {sorted(dead)}"
    assert (
        len(dead) == nano.n_layers
    ), f"expected one inert array per layer, got {sorted(dead)}"


# -- correctness, on the part that can be checked ------------------------------


def test_central_differences_agree_downstream_of_the_last_state_write():
    """The actual gradient check, on the parameters it is valid for.

    Needs the frozen mask, because the mask is a function of these parameters: it
    thresholds ``abs(step_out)`` and ``step_out`` ends in ``W_o``, so the true
    loss has a ``d(mask)/d(W_o)`` term the estimator drops. Freezing the mask
    zeroes that term and leaves a smooth function, and the two methods agree.

    ``below-noise`` is accepted, ``mismatch`` is not. The distinction is the
    point: a gradient smaller than the method can resolve has not been checked,
    and counting it as a pass would be the same mistake in a smaller key.
    """
    for name in SMOOTH:
        results = _probes(name, count=PROBES)
        bad = [r for r in results if r.verdict == "mismatch"]
        assert not bad, f"{name} (mask frozen):\n" + "\n".join(str(r) for r in bad)


def test_earlier_blocks_disagree_only_at_staircase_edges():
    """Where finite differences are unreliable, and the discriminator.

    A parameter with a downstream int state write is piecewise constant, so most
    probes land inside a step and agree, and the ones that straddle an edge
    disagree. Measured: at the default step the edges are too rare to see (0 of
    24), which is what the previous cap-based version of this test was silently
    relying on -- a cap that any gradient, correct or not, passes when the
    denominator is small. So the step is widened to 1e-2, where edges are
    actually crossed, and the assertion is strengthened from a rate to
    containment.

    The strong claim is not "few disagree" but "every probe that disagrees is a
    probe that straddled a hard boundary". A wrong gradient in these parameters
    would disagree on smooth probes too, and that is what this rules out.
    """
    step = 1e-2
    total_mismatched = 0
    for name in STAIRCASE:
        results = _probes(name, count=PROBES, step=step)
        disc = _continuity(name, freeze=True, count=PROBES, step=step)
        mismatched = {r.index for r in results if r.verdict == "mismatch"}
        straddling = {i for i, d in disc.items() if d}
        unexplained = mismatched - straddling
        assert not unexplained, (
            f"{name} at step={step}: {len(unexplained)} probe(s) disagree while the "
            f"+/- pair stayed smooth at indices {sorted(unexplained)} -- that is a "
            f"real gradient error, not a staircase"
        )
        total_mismatched += len(mismatched)
    assert total_mismatched, (
        f"no staircase probe disagreed even at step={step}, so this test is "
        f"vacuous: the cap it replaces is passing for the wrong reason"
    )


def test_a_discontinuity_is_never_excused_as_a_staircase():
    """The converse guard: 'staircase' must not become a blanket excuse.

    A harness that classifies probes as continuous, then quietly excuses any
    disagreement, proves nothing. So widen the step far enough that truncation
    error swamps the derivative in the *smooth* parameters too, and check the
    two sets are told apart: the smooth parameters genuinely never straddle a
    boundary, so their disagreement at a huge step is the honest finite
    difference failing, not the staircase being invoked after the fact.

    The second half is what keeps the first honest. Asserting only "no smooth
    parameter straddles" would also be satisfied by a harness where nothing
    disagrees at any step -- measured, 0.5 does produce smooth-set mismatches --
    so the label is shown to be doing real work rather than always answering
    "no".
    """
    step = 0.5
    guard = 3
    for name in SMOOTH:
        disc = _continuity(name, freeze=True, count=guard, step=step)
        straddling = {i for i, d in disc.items() if d}
        assert not straddling, (
            f"{name}: {len(straddling)} probe(s) straddle a hard boundary at "
            f"step={step}, so this set is not smooth and calling its mismatches "
            f"truncation error would be wrong"
        )
    sample = _probes(SMOOTH[0], freeze=True, count=2, step=step)
    assert _verdicts(sample).get("mismatch", 0), (
        f"{SMOOTH[0]} agrees with its own gradient even at step={step}, so the "
        f"continuity classification is never actually load-bearing and the "
        f"guard above cannot distinguish anything"
    )


def test_the_thresholds_are_not_finite_difference_checkable():
    """Why the gate thresholds are verified elsewhere instead.

    The mask is a hard decision, so the loss is piecewise constant in the
    thresholds: a perturbation either changes nothing at all or flips a channel.
    Measured at three step sizes spanning 1e-1 down to 1e-4, the finite difference
    never comes near the analytic gradient, and the residual does not shrink as
    the step does.
    This is not a gap dressed up as a pass. It is why those two groups are
    verified by the governor's own component tests, where the mask is held fixed
    and the threshold's gradient is meaningful.
    """
    smooth_name = "unembed.output"
    for name in ["gates.0.tau_hi", "gates.1.tau_lo"]:
        # The live mask, necessarily: freezing it would disconnect the thresholds
        # from the loss entirely, and the question here is what the live mask does.
        for step in (1e-1, DEFAULT_STEP, 1e-4):
            results = _probes(name, freeze=False, count=2, step=step)
            for r in results:
                assert r.verdict == "mismatch" or abs(r.numeric) == 0.0, (
                    f"{name} at step={step}: the difference {r.numeric:.3e} "
                    f"landed close to the analytic {r.analytic:.3e}, so the "
                    f"threshold is checkable after all and this test is stale"
                )
    # The contrast that gives the claim teeth: the same measurement on a smooth
    # parameter does converge.
    results = _probes(smooth_name, count=3, step=DEFAULT_STEP)
    assert any(r.verdict == "ok" for r in results), (
        f"{smooth_name} should be checkable; if it is not, the threshold "
        f"result above is not evidence of anything"
    )


def test_the_estimator_omits_the_mask_derivative_on_purpose():
    """The live-versus-frozen contrast, stated as the difference it is.

    With the live mask the finite difference and the analytic gradient disagree
    on most probes; with the mask frozen they all agree. That difference *is*
    the ``d(mask)/d(param)`` surrogate term: real, and deliberately not
    modelled, because modelling it would mean differentiating a comparison
    operator. The magnitude is not quoted because it moved substantially when
    the projection scales were corrected, and a stale factor here is exactly
    the kind of number that quietly stops meaning anything.

    Pinning it matters because the naive version of this check -- live mask, no
    explanation -- reports a large, confident, entirely expected discrepancy that
    is indistinguishable from a gradient bug.
    """
    for name in ["banks.1.W_o", "banks.1.G", "mixers.0.W1"]:
        live = _verdicts(_probes(name, freeze=False, count=4))
        frozen = _verdicts(_probes(name, freeze=True, count=4))
        live_ok = live.get("ok", 0)
        frozen_ok = frozen.get("ok", 0)
        assert frozen_ok > live_ok, (
            f"{name}: freezing the mask did not help ({frozen_ok} ok frozen vs "
            f"{live_ok} ok live), so the discrepancy is not the mask derivative"
        )
    frozen_counts = _verdicts(_probes("banks.1.W_o", count=4))
    assert (
        frozen_counts.get("mismatch", 0) == 0
    ), f"with the mask frozen the read-out should not disagree at all: {frozen_counts}"


def test_the_freeze_really_removes_the_masks_dependence_on_its_input():
    """A guard on :func:`_freeze_masks` itself.

    If the freeze silently stopped working -- wrong object patched, mask still
    live -- the agreement test would quietly degrade to reporting fewer passes
    and the contrast test could still pass for the wrong reason. So the freeze is
    checked directly.
    """
    mir = _build()
    before = float(mir.gates[0].tau_hi.detach().sum())
    _freeze_masks(mir)
    assert (
        float(mir.gates[0].tau_hi.detach().sum()) == before
    ), "the freeze mutated the gate"
    for gate in mir.gates:
        quiet = torch.zeros(4)
        loud = torch.full((4,), 1e3)
        assert torch.equal(
            gate(quiet, quiet.abs()), gate(loud, loud.abs())
        ), "the frozen mask still depends on its input"


# -- is each probe even applicable? ---------------------------------------------


def _fingerprint(mir: BhanoxMirror, masks: list) -> np.ndarray:
    """Everything a hard decision could change: the int32 state and the masks.

    Comparing this between the ``+h`` and ``-h`` evaluations of one probe is
    what makes "discontinuous" a measurement rather than an assumption about
    which parameters happen to sit upstream of a state write.
    """
    parts = [
        t.detach().cpu().numpy().reshape(-1).copy()
        for _, t in sorted(mir.named_buffers())
        if t.dtype == torch.int32
    ]
    parts += [np.asarray(m.detach().cpu().numpy()).reshape(-1) for m in masks]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _continuity(
    name: str, *, freeze: bool, count: int, step: float = DEFAULT_STEP
) -> dict[int, bool]:
    """Per sampled index: did the +/- pair straddle a hard boundary?

    A central difference over an interval that contains a discontinuity is not
    a small-inaccurate estimate, it is the average slope across a jump. So each
    probe has to be classified before its quotient means anything, and the
    classification is: compare the int32 state and the gate masks produced at
    ``+h`` and ``-h``. Equal means the path between them was smooth.

    Note that ``freeze=True`` replaces ``gate.forward`` wholesale, so no mask is
    recorded and only the state write can be detected. That is the honest
    reading: the frozen model has no gate decision left to straddle, which is
    precisely what makes its finite differences meaningful.
    """
    from bhanox.train.gradcheck import _sample_indices

    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))
    mir = _build(freeze=freeze)
    saved = _capture(mir)
    params = dict(mir.named_parameters())
    arr = params[name].detach().cpu().numpy().copy()

    masks: list = []
    for gate in mir.gates:
        inner = gate.step

        def spy(a, b, _inner=inner):
            m = _inner(a, b)
            masks.append(m.detach().clone())
            return m

        gate.step = spy  # type: ignore[method-assign]

    def at(a: np.ndarray) -> tuple[float, np.ndarray]:
        _restore(mir, saved)
        masks.clear()
        with torch.no_grad():
            params[name].copy_(torch.from_numpy(a))
            value = float(_loss(mir, ids, target))
        return value, _fingerprint(mir, masks)

    out: dict[int, bool] = {}
    for i in _sample_indices(arr.size, count, 0):
        plus_arr, minus_arr = arr.copy(), arr.copy()
        plus_arr.reshape(-1)[i] += step
        minus_arr.reshape(-1)[i] -= step
        _, fp_plus = at(plus_arr)
        _, fp_minus = at(minus_arr)
        out[int(i)] = not np.array_equal(fp_plus, fp_minus)
    return out


def test_every_provably_smooth_probe_is_actually_continuous():
    """The smooth region's real evidence is continuity, not a mismatch count.

    The previous version of this file asserted "0 mismatches" over the smooth
    set and stopped there. That is a necessary condition and not a sufficient
    one: zero mismatches is also what you get from a harness that probes nothing,
    or from one whose quotients land in the noise floor. So the strong claim is
    made directly -- across the sampled probes, no +/- pair changes the integer
    state or the gate mask, which is what licenses the quotient in the first
    place. Counts come second.
    """
    for name in SMOOTH:
        broken = [
            i
            for i, disc in _continuity(name, freeze=True, count=PROBES).items()
            if disc
        ]
        assert not broken, (
            f"{name}: {len(broken)} of {PROBES} probes straddle a hard boundary "
            f"at indices {broken}, so their quotients are not derivatives"
        )


def test_the_mask_gate_is_measurably_active_but_not_discontinuous_here():
    """The live/frozen gap is a surrogate term, not a discontinuity.

    An earlier version of this test asserted that turning the live gate on makes
    a mask *flip* under the perturbation. It does not -- not at 1e-3, and not at
    0.5 either. So the usual story ("the loss is piecewise constant, the
    derivative does not exist") is the wrong explanation for this gap, and
    asserting it was asserting a falsehood that happened to pass on other
    platforms.

    What is actually going on: the live mask skips a real fraction of channels
    (measured below), so the straight-through term ``step_out * (compute -
    compute.detach())`` contributes a genuine non-zero gradient, while adding
    exactly zero to the forward value. The true derivative of that locally
    constant mask is zero, so the estimator over-counts here by design. Freezing
    the mask sets ``compute`` to 1, which zeroes that surrogate term and restores
    agreement. Both facts are asserted, so this test now fails loudly if the
    gate ever becomes trivial or genuinely discontinuous.
    """
    name = "banks.1.W_o"
    mir = _build(freeze=False)
    seen: list = []
    for gate in mir.gates:
        inner = gate.step

        def spy(a, b, _inner=inner):
            m = _inner(a, b)
            seen.append(m.detach().clone().reshape(-1).cpu().numpy())
            return m

        gate.step = spy  # type: ignore[method-assign]
    saved = _capture(mir)
    _restore(mir, saved)
    vocab = TINY.output_vocab
    seen.clear()
    with torch.no_grad():
        _loss(mir, _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1)))
    values = np.concatenate(seen)
    assert values.size and (values == 0).any() and (values != 0).any(), (
        f"the live gate is trivial (values {np.unique(values)[:4]}); if it never "
        f"skips a channel there is no surrogate term to explain the live gap"
    )

    frozen = _continuity(name, freeze=True, count=PROBES)
    live = _continuity(name, freeze=False, count=PROBES)
    assert sum(frozen.values()) == 0, (
        "frozen the gate and a probe still moved the state, so the frozen model "
        "is not the smooth conditional path the finite differences assume"
    )
    assert sum(live.values()) == 0, (
        "the live mask now flips under this perturbation, so the live gap is a "
        "genuine discontinuity and the surrogate-term explanation in this test's "
        "docstring is stale -- re-measure before trusting either claim"
    )
    live_ok = _verdicts(_probes(name, freeze=False, count=PROBES))
    frozen_ok = _verdicts(_probes(name, freeze=True, count=PROBES))
    assert frozen_ok.get("ok", 0) > live_ok.get("ok", 0), (
        f"freezing the mask did not improve agreement ({frozen_ok} frozen vs "
        f"{live_ok} live), so the gap is not the surrogate term"
    )


# -- the harness must be able to fail ------------------------------------------


def test_a_corrupted_supplied_gradient_is_caught_at_the_same_point():
    """Positive control on the check itself, independent of model numerics.

    Everything else here compares the model's own gradient to finite
    differences. This one holds the parameters *fixed* and corrupts only the
    gradient array handed to :func:`check_tensor` -- which is the shape of the
    bugs a gradient harness is uniquely able to have: a transposed operand, a
    dropped sign, a stride taken from the wrong axis, a scale factor applied
    twice. The parameters never move, so the numeric derivative is identical in
    every case and any change in verdict is the check working.
    """
    from bhanox.train.gradcheck import _sample_indices

    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))
    mir = _build(freeze=True)
    saved = _capture(mir)
    params = dict(mir.named_parameters())
    _restore(mir, saved)
    _loss(mir, ids, target).backward()
    name = "banks.1.W_o"
    good = params[name].grad.detach().cpu().numpy().copy()
    arr = params[name].detach().cpu().numpy().copy()

    def loss(a: np.ndarray = arr) -> float:
        _restore(mir, saved)
        with torch.no_grad():
            params[name].copy_(torch.from_numpy(a))
        with torch.no_grad():
            return float(_loss(mir, ids, target))

    clean = check_tensor(name, arr, good, loss, count=PROBES, step=DEFAULT_STEP)
    assert (
        _verdicts(clean).get("mismatch", 0) == 0
    ), "the control's premise is broken: the honest gradient must not disagree"

    # Same parameters, same numeric derivative, four different ways to be wrong.
    corruptions = {
        "sign flipped": -good,
        "scaled 1.5x": good * 1.5,
        "zeroed": np.zeros_like(good),
        "shifted by one element": np.roll(good, 1, axis=0),
    }
    indices = _sample_indices(arr.size, PROBES, 0)
    for label, bad in corruptions.items():
        results = check_tensor(name, arr, bad, loss, count=PROBES, step=DEFAULT_STEP)
        caught = {r.index for r in results if r.verdict == "mismatch"}
        assert caught, (
            f"a supplied gradient with the {label} was not caught, so agreement "
            f"elsewhere in this file is not evidence of anything"
        )
        # And it must be caught on the probes that carry signal, not merely on
        # an entry that happens to be tiny.
        assert any(
            abs(float(good.reshape(-1)[i])) > r.floor
            for r in results
            if r.index in caught
            for i in [int(r.index)]
        ), f"the {label} was only caught on a below-noise probe for {name}"
    # Sanity: the probes are the same points, so the comparison is like-for-like.
    assert [r.index for r in clean] == [int(i) for i in indices]


def test_a_broken_gradient_would_be_caught_by_this_harness():
    """Positive control. A checker that cannot fail is not evidence of anything.

    Scales the read-out by 3.0 -- a real error of a size a plausible bug would
    produce -- and requires the check to notice. A 1% error is *not* used,
    because it falls below the noise floor: the method genuinely cannot resolve
    it, and demanding that it catch one would be demanding a lie.

    The multiplier was 1.5 in an earlier version of this test, and it passed for
    the wrong reason. Measured at the corrected scales, 1.5x moves the loss by
    2.1e-2 but the gradient by only ~13%, which is an absolute error of 4.1e-4
    against a noise floor of 4.8e-4 -- so the checker correctly reports "ok" and
    the control asserted nothing. The floor is absolute, not relative, so the
    smallest detectable weight corruption is a property of the loss scale. The
    sweep below pins both ends of that threshold so it cannot drift back.
    """
    name = "banks.1.W_o"
    results = _probes(name, count=4)
    assert (
        _verdicts(results).get("mismatch", 0) == 0
    ), "the control's premise is broken: the unbroken model must not disagree"

    probe = _build(freeze=True)
    probe_saved = _capture(probe)
    probe_params = dict(probe.named_parameters())
    _restore(probe, probe_saved)
    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))
    _loss(probe, ids, target).backward()
    good_grad = probe_params[name].grad.detach().cpu().numpy().copy()

    def caught(mult: float) -> int:
        mir = _build(freeze=True)
        with torch.no_grad():
            mir.banks[1].W_o.mul_(mult)
        saved = _capture(mir)
        params = dict(mir.named_parameters())
        arr = params[name].detach().cpu().numpy().copy()

        def loss(a: np.ndarray = arr) -> float:
            _restore(mir, saved)
            with torch.no_grad():
                params[name].copy_(torch.from_numpy(a))
            with torch.no_grad():
                return float(_loss(mir, ids, target))

        out = check_tensor(name, arr, good_grad, loss, count=4, step=DEFAULT_STEP)
        return sum(r.verdict == "mismatch" for r in out)

    assert caught(3.0) == 4, (
        "tripling the read-out was not caught on every probe, so the passing "
        "gradient checks above are not evidence of anything"
    )
    # And the near-miss, pinned: 1.5x sits under the floor at these scales. If a
    # future change to the loss scale makes this detectable, that is an
    # improvement, and the recorded expectation should be updated -- not left
    # to make the headline number look worse.
    assert caught(1.5) < 4, (
        "a 1.5x read-out error is now fully detected, which means this file has "
        "more resolution than its docstring claims; re-measure and update"
    )


def test_the_noise_floor_is_computed_and_in_range():
    """The floor is derived from the loss scale and dtype, not chosen.

    Keeps the agreement tests honest about the tolerance they lean on: the
    documented ``|loss| * eps / h``, in the range measured for this loss.
    """
    mir = _build()
    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))
    with torch.no_grad():
        scale = float(_loss(mir, ids, target))
    floor = noise_floor(scale, DEFAULT_STEP, np.float32)
    assert (
        1e-5 < floor < 1e-3
    ), f"noise floor {floor:.2e} is outside its documented range"
    # The floor is exactly |loss| * eps / h, so it tracks eps and nothing else.
    # Which is the whole reason a float64 *loss* buys nothing here: the
    # perturbation is rounded to the parameter array's dtype on the way back in.
    f32 = noise_floor(scale, DEFAULT_STEP, np.float32)
    f64 = noise_floor(scale, DEFAULT_STEP, np.float64)
    assert f32 / f64 == pytest.approx(
        np.finfo(np.float32).eps / np.finfo(np.float64).eps
    ), "the floor must scale with the parameter dtype's eps"
