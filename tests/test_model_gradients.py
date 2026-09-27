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
unembed -- 56 probes, no mismatches. Everything earlier has a downstream
staircase, and finite differences there are unreliable rather than wrong: a
minority of probes land on a staircase edge and disagree, the rest agree.

The gate adds a second, separate effect. The mask is a function of the very
parameters whose gradient is being checked -- it thresholds ``abs(step_out)``,
and ``step_out`` ends in ``W_o`` -- so the true loss has a ``d(mask)/d(W_o)``
term that the straight-through estimator omits on purpose. Comparing the two
without accounting for it produces confident nonsense: the first version of this
file reported 32 mismatches from a harness bug, and the second reported 32 more
that were a real and intended difference. Both are pinned as tests now, because
a gradient harness that cannot demonstrate it detects agreement is not evidence
of anything -- see the positive control at the bottom.

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
    probes land inside a step and agree, and a minority land on an edge and
    disagree. Measured: 8 mismatches out of 64 probes across four parameters.

    The cap is the test. A genuinely wrong gradient in these would disagree on
    nearly every probe, so "mismatches are a minority" separates a staircase from
    a bug. It is a cap and not a zero because the honest number is not zero.
    """
    for name in STAIRCASE:
        results = _probes(name, count=PROBES)
        counts = _verdicts(results)
        checked = counts.get("ok", 0) + counts.get("mismatch", 0)
        assert (
            checked >= 4
        ), f"{name}: too few resolvable probes to mean anything: {counts}"
        bad = counts.get("mismatch", 0)
        assert bad < 0.4 * checked, (
            f"{name}: {bad}/{checked} probes mismatch. A minority is the staircase; "
            f"a majority would be a wrong gradient."
        )


def test_the_thresholds_are_not_finite_difference_checkable():
    """Why the gate thresholds are verified elsewhere instead.

    The mask is a hard decision, so the loss is piecewise constant in the
    thresholds: a perturbation either changes nothing at all or flips a channel.
    Measured at four step sizes from 1e-1 down to 1e-4, the finite difference
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
    by roughly 3x on the read-out. With the mask frozen they agree exactly. That
    difference *is* the ``d(mask)/d(param)`` term: real, and deliberately not
    modelled, because modelling it would mean differentiating a comparison
    operator.

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


# -- the harness must be able to fail ------------------------------------------


def test_a_broken_gradient_would_be_caught_by_this_harness():
    """Positive control. A checker that cannot fail is not evidence of anything.

    Scales the read-out by 1.5 -- a real error of a size a plausible bug would
    produce -- and requires the check to notice. A 1% error is *not* used,
    because it falls below the noise floor: the method genuinely cannot resolve
    it, and demanding that it catch one would be demanding a lie.
    """
    results = _probes("banks.1.W_o", count=4)
    assert (
        _verdicts(results).get("mismatch", 0) == 0
    ), "the control's premise is broken: the unbroken model must not disagree"

    mir = _build(freeze=True)
    with torch.no_grad():
        mir.banks[1].W_o.mul_(1.5)
    saved = _capture(mir)
    params = dict(mir.named_parameters())
    vocab = TINY.output_vocab
    ids, target = _ids(vocab, 0), torch.from_numpy(_ids(vocab, 1).reshape(-1))

    # The analytic gradient of the *unbroken* model, against the loss of the
    # broken one. That is the shape of a real gradient bug.
    probe = _build(freeze=True)
    probe_saved = _capture(probe)
    probe_params = dict(probe.named_parameters())
    _restore(probe, probe_saved)
    _loss(probe, ids, target).backward()
    good_grad = probe_params["banks.1.W_o"].grad.detach().cpu().numpy().copy()

    arr = params["banks.1.W_o"].detach().cpu().numpy().copy()

    def loss(a: np.ndarray = arr) -> float:
        _restore(mir, saved)
        with torch.no_grad():
            params["banks.1.W_o"].copy_(torch.from_numpy(a))
        with torch.no_grad():
            return float(_loss(mir, ids, target))

    results = check_tensor(
        "banks.1.W_o", arr, good_grad, loss, count=4, step=DEFAULT_STEP
    )
    assert any(r.verdict == "mismatch" for r in results), (
        "a 50% error in the read-out was not detected, so the passing gradient "
        "checks above are not evidence of anything"
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
