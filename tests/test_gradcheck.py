"""Tests for the finite-difference gradient-check harness.

The harness's own job is to catch a wrong gradient, so the tests are shaped like
the failure it exists to prevent: start from a loss with a gradient that is known
correct, then corrupt it in specific ways and assert each corruption is caught
*at the right index*. A harness that reported "looks fine" on a broken gradient
would be worse than no harness, and a harness that flagged everything would be
indistinguishable from one that works.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from bhanox.train.gradcheck import (
    DEFAULT_STEP,
    CheckResult,
    central_difference,
    check_all,
    check_tensor,
    classify,
    iter_flat,
    noise_floor,
    report,
)


def quadratic(
    sym: bool = True,
    *,
    n: int = 8,
    seed: int = 0,
    dtype: type[np.floating] = np.float64,
) -> tuple[
    dict[str, NDArray[np.floating]],
    dict[str, NDArray[np.floating]],
    Callable[[], float],
]:
    """``f(w) = 0.5 w'Aw + b'w``.

    With ``sym=True`` the gradient is ``Aw + b``. With ``sym=False`` the
    symmetric part matters and the naive ``Aw + b`` is genuinely wrong -- which
    is the bug the harness caught the first time it was pointed at a real case.

    ``dtype`` sets the dtype of the *parameter* array, and the loss closes over
    that same array. That coupling is the point: a loss closing over a different
    array than the one being perturbed returns a numeric gradient of exactly
    zero, which the harness reports as a confident mismatch rather than as the
    wiring mistake it is.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n).astype(dtype)
    raw = rng.standard_normal((n, n))
    base = (raw @ raw.T + np.eye(n) * 3.0) if sym else (raw + np.eye(n) * 3.0)
    a_mat = base.astype(dtype)
    b = rng.standard_normal(n).astype(dtype)

    def loss() -> float:
        return float(0.5 * w @ a_mat @ w + b @ w)

    grad = (0.5 * (a_mat + a_mat.T) @ w + b) if sym else (a_mat @ w + b)
    return {"w": w}, {"w": np.asarray(grad, dtype=np.float64)}, loss


class TestNoiseFloor:
    def test_floor_scales_with_the_loss(self) -> None:
        assert noise_floor(10.0, 1e-3, np.float64) == pytest.approx(
            2 * noise_floor(5.0, 1e-3, np.float64)
        )

    def test_floor_inverses_with_the_step(self) -> None:
        """A smaller step resolves less, which is the whole reason a floor has
        to exist rather than a hand-picked tolerance.
        """
        assert noise_floor(1.0, 1e-4, np.float64) == pytest.approx(
            10 * noise_floor(1.0, 1e-3, np.float64)
        )

    def test_float32_floor_is_far_coarser_than_float64(self) -> None:
        """The point of computing the floor: the parameter dtype dominates, and
        promoting the *loss* to float64 does not rescue a float32 weight.
        """
        f32 = noise_floor(58.0, 1e-3, np.float32)
        f64 = noise_floor(58.0, 1e-3, np.float64)
        assert f32 / f64 > 1e6

    def test_matches_the_measured_behaviour(self) -> None:
        """A bound nobody checked is a guess. This is the number a real float32
        quadratic produced, so the constant is pinned to evidence.
        """
        assert noise_floor(58.0, 1e-3, np.float32) == pytest.approx(6.9e-3, rel=0.2)


class TestClassify:
    def test_agreement_is_ok(self) -> None:
        assert classify(1.0, 1.0001)[2] == "ok"

    def test_a_real_disagreement_is_a_mismatch(self) -> None:
        assert classify(1.0, 2.0)[2] == "mismatch"

    def test_exactly_zero_is_its_own_bucket_not_a_pass(self) -> None:
        assert classify(0.0, 0.0)[2] == "zero"

    def test_unresolvable_is_not_a_pass(self) -> None:
        """A gradient below the floor was not checked, and saying "ok" would
        claim coverage that does not exist.
        """
        assert classify(1e-9, 2e-9, floor=1e-3)[2] == "below-noise"

    def test_the_floor_takes_priority_over_relative_agreement(self) -> None:
        """Relative agreement inside the floor is not evidence."""
        assert classify(1e-6, 1.0e-6 * 1.01, floor=1e-3)[2] == "below-noise"

    def test_above_the_floor_still_judged_relatively(self) -> None:
        assert classify(1.0, 1.5, floor=1e-6)[2] == "mismatch"

    def test_relative_error_uses_the_larger_magnitude(self) -> None:
        abs_err, rel_err, _ = classify(2.0, 1.0)
        assert abs_err == 1.0
        assert rel_err == 0.5


class TestCentralDifference:
    def test_recovers_a_linear_gradient(self) -> None:
        g = np.array([2.0, -3.0, 0.5])
        p = {"w": np.array([10.0, -4.0, 7.0])}
        loss = lambda: float(g @ p["w"])  # noqa: E731
        for i in range(3):
            assert central_difference(loss, p["w"], i, 1e-5) == pytest.approx(
                g[i], abs=1e-6
            )

    def test_restores_the_parameter_it_perturbed(self) -> None:
        """A check that permanently shifted a weight would report a gradient for
        a model nobody is training, and every later probe would be of the
        perturbed model.
        """
        w = np.array([1.0, 2.0, 3.0])
        before = w.copy()
        central_difference(lambda: float(np.sum(w**2)), w, 1, 1e-3)
        assert np.array_equal(w, before)

    def test_a_nonfinite_loss_propagates(self) -> None:
        w = np.array([1.0])
        with pytest.raises(ValueError, match="finite"):
            central_difference(lambda: float("nan"), w, 0, 1e-3)


class TestCheckTensor:
    def test_accepts_a_correct_gradient(self) -> None:
        p, g, loss = quadratic()
        assert all(r.verdict == "ok" for r in check_all(p, g, loss))

    def test_catches_an_asymmetric_hessian_at_every_index(self) -> None:
        """The live case: ``A`` not symmetric, so the naive ``Aw + b`` is wrong
        while still looking plausible. Every index, not just the worst one.
        """
        p, _, _ = quadratic(sym=False)
        rng = np.random.default_rng(0)
        asym = rng.standard_normal((8, 8)) + np.eye(8) * 3.0

        def loss() -> float:
            w = p["w"]
            return float(0.5 * w @ asym @ w)

        r = check_tensor("w", p["w"], asym @ p["w"], loss, count=8)
        assert all(x.verdict == "mismatch" for x in r)

    def test_locates_a_single_corrupted_element(self) -> None:
        p, g, loss = quadratic()
        g["w"] = g["w"].copy()
        g["w"][3] *= 2.0
        r = check_all(p, g, loss)
        bad = [x.index for x in r if x.verdict == "mismatch"]
        assert bad == [3], f"expected only index 3, got {bad}"

    def test_catches_a_sign_flip(self) -> None:
        p, g, loss = quadratic()
        g["w"] = -g["w"]
        assert all(r.verdict == "mismatch" for r in check_all(p, g, loss))

    def test_catches_a_constant_offset(self) -> None:
        p, g, loss = quadratic()
        g["w"] = g["w"] + 0.5
        assert all(r.verdict == "mismatch" for r in check_all(p, g, loss))

    def test_sampling_is_deterministic(self) -> None:
        p, g, loss = quadratic(n=64)
        a = [r.index for r in check_tensor("w", p["w"], g["w"], loss, count=5, seed=7)]
        b = [r.index for r in check_tensor("w", p["w"], g["w"], loss, count=5, seed=7)]
        assert a == b

    def test_different_seeds_probe_different_elements(self) -> None:
        p, g, loss = quadratic(n=64)
        a = {r.index for r in check_tensor("w", p["w"], g["w"], loss, count=4, seed=1)}
        b = {r.index for r in check_tensor("w", p["w"], g["w"], loss, count=4, seed=2)}
        assert a != b

    def test_index_zero_is_always_probed(self) -> None:
        """The one a hand-written indexing bug is most likely to miss, and a
        purely random sample can skip.
        """
        p, g, loss = quadratic(n=64)
        for seed in range(6):
            r = check_tensor("w", p["w"], g["w"], loss, count=4, seed=seed)
            assert 0 in {x.index for x in r}

    def test_count_above_size_is_clamped_not_an_error(self) -> None:
        p, g, loss = quadratic(n=4)
        assert len(check_all(p, g, loss)) == 4

    def test_check_all_ignores_count_and_checks_everything(self) -> None:
        """Deliberate: a count passed to ``check_all`` would defeat its purpose
        silently, so it is dropped rather than honoured.
        """
        p, g, loss = quadratic(n=16)
        assert len(check_all(p, g, loss, count=2)) == 16

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="elements"):
            check_tensor("w", np.zeros(4), np.zeros(3), lambda: 0.0, count=1)

    def test_reports_the_floor_it_used(self) -> None:
        p, g, loss = quadratic()
        r = check_all(p, g, loss)
        assert all(x.floor > 0 for x in r)

    def test_a_zero_loss_yields_a_zero_floor_not_a_crash(self) -> None:
        w = np.array([1.0, 2.0])
        r = check_tensor("w", w, np.zeros(2), lambda: 0.0, count=2)
        assert all(x.verdict == "zero" for x in r)
        assert all(x.floor == 0.0 for x in r)


class TestFloat32Parameters:
    """The regime the model actually trains in, and where the floor bites."""

    def test_a_correct_float32_gradient_passes(self) -> None:
        p, g, loss = quadratic(dtype=np.float32)
        r = check_all(p, g, loss, step=DEFAULT_STEP)
        assert all(x.verdict == "ok" for x in r), report(r)

    def test_a_wrong_float32_gradient_is_still_caught(self) -> None:
        p, g, loss = quadratic(dtype=np.float32)
        g["w"] = -g["w"]
        r = check_all(p, g, loss, step=DEFAULT_STEP)
        assert all(x.verdict == "mismatch" for x in r)

    def test_the_floor_comes_from_the_parameter_dtype(self) -> None:
        """Promoting the loss to float64 does not buy accuracy when the weight
        being perturbed is still float32. This is the mistake worth being unable
        to make by accident.
        """
        p64, g64, l64 = quadratic()
        p32, g32, l32 = quadratic(dtype=np.float32)
        f64 = check_all(p64, g64, l64)[0].floor
        f32 = check_all(p32, g32, l32)[0].floor
        assert f32 / f64 > 1e6

    def test_a_loss_over_the_wrong_array_gives_a_zero_numeric_gradient(self) -> None:
        """The wiring mistake this module is most likely to hit in real use, and
        why the loss has to close over the live array. It shows up as a
        confident 100% mismatch rather than as a subtle error, which is at least
        diagnosable.
        """
        p, _, _ = quadratic()
        _, _, detached = quadratic()  # a different, unrelated loss
        p32 = {"w": p["w"].astype(np.float32)}
        r = check_tensor("w", p32["w"], np.zeros(8), detached, count=8)
        assert all(x.numeric == 0.0 for x in r)


class TestReport:
    def test_lists_mismatches_first(self) -> None:
        p, g, loss = quadratic()
        g["w"] = g["w"] + 0.5
        out = report(check_all(p, g, loss))
        assert out.splitlines()[1].startswith("MISMATCHES")

    def test_only_ok_counts_as_a_pass(self) -> None:
        """The accounting has to be honest, or "8/8 checked" means nothing."""
        p, g, loss = quadratic()
        g["w"] = g["w"] + 0.5
        out = report(check_all(p, g, loss))
        assert out.splitlines()[0].startswith("gradient check: 0 ok,")

    def test_reports_every_bucket(self) -> None:
        p, g, loss = quadratic()
        out = report(check_all(p, g, loss))
        assert "below-noise" in out and "zero" in out

    def test_a_clean_run_says_so_plainly(self) -> None:
        p, g, loss = quadratic()
        assert report(check_all(p, g, loss)).splitlines()[0] == (
            "gradient check: 8 ok, 0 mismatch, 0 below-noise, 0 zero"
        )


class TestIterFlat:
    def test_yields_every_element(self) -> None:
        p: dict[str, NDArray[np.floating]] = {
            "a": np.zeros(3),
            "b": np.ones((2, 2)),
        }
        assert len(list(iter_flat(p))) == 7

    def test_carries_names_and_values(self) -> None:
        p: dict[str, NDArray[np.floating]] = {"a": np.array([1.0, 2.0])}
        assert list(iter_flat(p)) == [("a", 0, 1.0), ("a", 1, 2.0)]


def test_result_reads_legibly() -> None:
    r = CheckResult("w", 3, 1.0, 1.5, 0.5, 0.33, 1e-9, "mismatch")
    text = str(r)
    assert "w[3]" in text and "mismatch" in text
