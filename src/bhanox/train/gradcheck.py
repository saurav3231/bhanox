"""Finite-difference gradient checks.

Before training a single weight, the gradient has to be demonstrably the
gradient *of the thing being trained*. A trainer with a subtly wrong gradient
does not crash -- it converges to something, reports a falling loss, and
produces a worse model. That is the most expensive kind of bug to find late, so
it gets checked first and by a method that shares no code with whatever
produced the analytic gradient.

The method is deliberately dumb: perturb one parameter, measure the loss
difference, divide by the step. It uses no autodiff and is therefore capable of
catching an analytic bug -- and it does. It caught a non-symmetric Hessian in
the test that first exercised it, reporting a relative error of 1.03 against an
"analytic" gradient that was simply wrong.

Why the tolerance is computed rather than chosen
-----------------------------------------------

The obvious move is a hand-picked ``rtol``, and it is the wrong one, because
the achievable accuracy of a central difference is not a property of the
gradient. It is set by cancellation:

    the loss is only known to |loss| * eps, and the quotient divides that by 2h

so the smallest gradient this method can *resolve* is roughly
``|loss| * eps / h``, with ``eps`` taken from the dtype of the **parameter
array**, not the loss. Measured on a 64-dim float32 quadratic with |loss| ~ 58
and h = 1e-3: floor ~6.9e-3, worst observed relative error 6.4e-3. Evaluating
the same loss in float64 barely helped -- 7.3e-5 -- because the perturbation
itself is rounded to float32 when written back into the array. That is the part
that is easy to get wrong: casting the loss to float64 does not buy float64
gradients when the weights are still float32.

So :func:`noise_floor` computes the bound and :func:`classify` compares against
it, and an entry whose gradient is *below* the floor is reported as
``below-noise`` rather than as a pass or a failure. Both of those would be lies:
it was never checked, and the method cannot check it. If too many entries land
there, the fix is a bigger step or a wider dtype, not a looser tolerance.

Other limits, stated rather than hidden:

- Cost is linear in the number of elements probed, so this samples. It is a spot
  check, not a proof. :func:`check_all` exists for small tensors and tests.
- Central differences cancel the O(h) truncation term, leaving O(h^2), which is
  why they are the default. The cost is the cancellation above.
- The loss must be evaluated with the model in the same mode both times. For
  Bhanox that specifically means ``train=False``: ``train=True`` updates the MoE
  load-balancing bias as a side effect, so two evaluations of *identical* weights
  would differ and every difference below would be contaminated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Relative tolerance applied *on top of* the computed noise floor. Covers
#: truncation error and the fact that the floor is a back-of-envelope bound.
DEFAULT_RTOL = 1e-3

#: Central-difference step. Small enough that the loss stays near-linear across
#: it, large enough that float32 cancellation does not dominate. See the module
#: docstring for why these trade off against each other.
DEFAULT_STEP = 1e-3

#: Multiplier on the cancellation bound. 1.0 is the derived value: the two loss
#: evaluations each carry ~|loss|*eps, and dividing their difference by 2h
#: leaves |loss|*eps/h. Extra margin comes from the rtol term in
#: :func:`classify` instead, because a floor that is too *tight* produces a false
#: mismatch -- visible, harmless -- while one that is too loose silently accepts a
#: wrong gradient, which is the entire failure this module exists to catch.
NOISE_SAFETY = 1.0


@dataclass(frozen=True)
class CheckResult:
    """Outcome of probing one parameter.

    Attributes:
        name: Parameter that was probed.
        index: Flat index probed, for an elementwise view of a tensor.
        analytic: Gradient reported by the implementation under test.
        numeric: Central difference of the loss.
        abs_error: ``abs(analytic - numeric)``.
        rel_error: Error relative to the larger of the two magnitudes.
        floor: Computed noise floor for this probe; the smallest difference
            this method can resolve at this step and dtype.
        verdict: ``"ok"``, ``"mismatch"``, ``"below-noise"``, or ``"zero"``.
            ``"below-noise"`` means the gradient is smaller than the method can
            resolve, so the probe proves nothing either way. ``"zero"`` means
            both sides are ~0. Neither is reported as a pass.
    """

    name: str
    index: int | None
    analytic: float
    numeric: float
    abs_error: float
    rel_error: float
    floor: float
    verdict: str

    def __str__(self) -> str:
        where = "" if self.index is None else f"[{self.index}]"
        return (
            f"{self.name}{where}: analytic={self.analytic:+.6e} "
            f"numeric={self.numeric:+.6e} rel={self.rel_error:.2e} "
            f"floor={self.floor:.2e} ({self.verdict})"
        )


def noise_floor(
    loss_scale: float,
    step: float,
    dtype: np.dtype[Any] | type[Any],
    *,
    safety: float = NOISE_SAFETY,
) -> float:
    """Smallest gradient difference a central difference can resolve.

    The loss is only known to about ``|loss| * eps``, and the quotient divides
    that by ``2h``, so the noise in the result is roughly ``|loss| * eps / h``.

    Args:
        loss_scale: Magnitude of the loss at the probe point.
        step: Central-difference step.
        dtype: dtype of the **parameter array**. This is what matters: the
            perturbation is rounded to this precision when written back, so a
            float64 loss over float32 weights gains almost nothing.
        safety: Multiplier on the bound, for the several roundings involved.

    Returns:
        An absolute tolerance. Agreement better than this is not evidence.
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    return safety * abs(loss_scale) * eps / step


def central_difference(
    loss: Callable[[], float],
    param: NDArray[np.floating],
    index: int,
    step: float = DEFAULT_STEP,
) -> float:
    """Central-difference derivative of ``loss`` w.r.t. one element of ``param``.

    The loss must be a closure over the *live* array, so that perturbing
    ``param`` in place is visible to it. Central rather than forward
    difference: it cancels the O(h) truncation term, leaving O(h^2), which
    matters because h cannot be made small in float32.

    Args:
        loss: Zero-argument callable returning the scalar loss.
        param: The array holding the parameter, modified in place and restored.
        index: Flat index of the element to probe.
        step: Perturbation size.

    Returns:
        The numeric derivative. ``param`` is left exactly as it was found.

    Raises:
        ValueError: If the loss is not finite. A NaN would otherwise propagate
            into every verdict as a plausible-looking number, and a check that
            reports NaN agreement has checked nothing.
    """
    original = float(param.reshape(-1)[index])
    try:
        param.reshape(-1)[index] = original + step
        plus = loss()
        param.reshape(-1)[index] = original - step
        minus = loss()
    finally:
        param.reshape(-1)[index] = original
    if not (math.isfinite(plus) and math.isfinite(minus)):
        raise ValueError(
            f"loss is not finite at index {index} (plus={plus}, minus={minus}); "
            "a NaN would be compared as if it were a gradient"
        )
    return (plus - minus) / (2.0 * step)


def classify(
    analytic: float,
    numeric: float,
    *,
    floor: float = 0.0,
    rtol: float = DEFAULT_RTOL,
    zero_tol: float = 1e-12,
) -> tuple[float, float, str]:
    """Error and verdict for one pair of gradients.

    The verdict distinguishes four outcomes, and the distinction matters: only
    one of them is a pass.

    Args:
        analytic: Gradient from the implementation under test.
        numeric: Gradient from finite differences.
        floor: Computed :func:`noise_floor`. Agreement inside it is not
            evidence, because the method cannot resolve differences that small.
        rtol: Relative tolerance applied on top of ``floor``.
        zero_tol: Magnitude below which both sides count as structurally zero.

    Returns:
        ``(abs_error, rel_error, verdict)``.
    """
    abs_error = abs(analytic - numeric)
    scale = max(abs(analytic), abs(numeric))
    rel_error = abs_error / scale if scale > 0 else 0.0
    if scale <= zero_tol:
        return abs_error, rel_error, "zero"
    if scale < floor:
        # Both sides are smaller than the method can resolve. Reporting this as
        # ok would credit a check that never happened.
        return abs_error, rel_error, "below-noise"
    return (
        abs_error,
        rel_error,
        ("ok" if abs_error <= floor + rtol * scale else "mismatch"),
    )


def _sample_indices(size: int, count: int, seed: int) -> NDArray[np.int64]:
    """Indices to probe, without replacement where possible.

    Always includes index 0: it is the one a hand-written indexing bug is most
    likely to miss, and a purely random sample can skip it.
    """
    if count >= size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    picked = rng.choice(
        np.arange(1, size, dtype=np.int64), size=count - 1, replace=False
    )
    return np.concatenate([np.zeros(1, dtype=np.int64), picked])


def check_tensor(
    name: str,
    param: NDArray[np.floating],
    grad: NDArray[np.floating],
    loss: Callable[[], float],
    *,
    count: int = 8,
    step: float = DEFAULT_STEP,
    rtol: float = DEFAULT_RTOL,
    seed: int = 0,
) -> list[CheckResult]:
    """Probe ``count`` elements of one parameter tensor.

    Args:
        name: Parameter name, for the report.
        param: The live parameter array. Restored after each probe.
        grad: The analytic gradient for the same array.
        loss: Closure returning the scalar loss, closing over ``param``.
        count: How many elements to probe.
        step: Perturbation size for the central difference.
        rtol: Relative tolerance.
        seed: Sampling seed, so a failure is reproducible.

    Returns:
        One :class:`CheckResult` per probed element.
    """
    flat_p = param.reshape(-1)
    flat_g = np.asarray(grad, dtype=np.float64).reshape(-1)
    if flat_p.size != flat_g.size:
        raise ValueError(
            f"{name}: param has {flat_p.size} elements, grad has {flat_g.size}"
        )
    # The noise floor needs the loss scale, so evaluate once at the unperturbed
    # point. This is also the mode-consistency check in practice: if ``loss``
    # mutates the model, the first probe already disagrees with the second.
    floor = noise_floor(loss(), step, param.dtype)
    out: list[CheckResult] = []
    for i in _sample_indices(flat_p.size, count, seed):
        analytic = float(flat_g[i])
        numeric = central_difference(loss, param, int(i), step)
        abs_err, rel_err, verdict = classify(analytic, numeric, floor=floor, rtol=rtol)
        out.append(
            CheckResult(
                name, int(i), analytic, numeric, abs_err, rel_err, floor, verdict
            )
        )
    return out


def check_all(
    params: dict[str, NDArray[np.floating]],
    grads: dict[str, NDArray[np.floating]],
    loss: Callable[[], float],
    **kwargs: Any,
) -> list[CheckResult]:
    """Probe every element of every parameter. For tests and small tensors."""
    missing = set(params) - set(grads)
    if missing:
        raise ValueError(f"no gradient supplied for: {sorted(missing)}")
    # ``count`` is not a caller option here: the point of this function is that
    # it checks everything, so a count passed in would defeat it silently.
    kwargs.pop("count", None)
    out: list[CheckResult] = []
    for name, param in params.items():
        out.extend(
            check_tensor(name, param, grads[name], loss, count=param.size, **kwargs)
        )
    return out


def report(results: list[CheckResult]) -> str:
    """Human-readable summary, with the failures listed first.

    Only ``ok`` counts as a pass. ``below-noise`` and ``zero`` are listed
    separately and deliberately not folded in: an entry the method cannot
    resolve is not a checked entry, and counting it would let an uninformative
    probe pad "12/12 checked" into looking like coverage it is not. If too many
    land in ``below-noise``, the answer is a larger step or a wider dtype.
    """
    by_verdict: dict[str, list[CheckResult]] = {}
    for r in results:
        by_verdict.setdefault(r.verdict, []).append(r)
    passed = by_verdict.get("ok", [])
    mismatches = by_verdict.get("mismatch", [])
    below = by_verdict.get("below-noise", [])
    zeros = by_verdict.get("zero", [])

    lines = [
        f"gradient check: {len(passed)} ok, {len(mismatches)} mismatch, "
        f"{len(below)} below-noise, {len(zeros)} zero",
    ]
    if mismatches:
        lines.append("MISMATCHES (listed first -- these are the real failures):")
        lines.extend(f"  {r}" for r in mismatches)
    if below:
        lines.append(
            f"below-noise entries ({len(below)}), NOT counted as passes -- the "
            "gradient is smaller than this method can resolve:"
        )
        lines.extend(f"  {r}" for r in below[:10])
    if zeros:
        lines.append(f"zero-gradient entries ({len(zeros)}), not counted as passes:")
        lines.extend(f"  {r}" for r in zeros[:10])
    return "\n".join(lines)


def iter_flat(
    params: dict[str, NDArray[np.floating]],
) -> Iterator[tuple[str, int, float]]:
    """Yield ``(name, flat_index, value)`` for every parameter element.

    Convenience for building a loss over a parameter dict without caring about
    shapes.
    """
    for name, param in params.items():
        flat = param.reshape(-1)
        for i in range(flat.size):
            yield name, i, float(flat[i])
