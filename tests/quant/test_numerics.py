"""Tests for the numerics regime: int8 absmax, STE, ternary packing, fixed point."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.quant import numerics as q


class TestAbsmaxQuantize:
    def test_round_trip_within_one_step(self) -> None:
        rng = np.random.default_rng(0)
        w = rng.standard_normal((64, 32))
        qt = q.absmax_quantize(w, axis=0)
        assert qt.bits == 8
        assert np.abs(qt.dequantize() - w).max() <= 1.0 / qt.scale.max()

    def test_values_are_integral(self) -> None:
        rng = np.random.default_rng(1)
        qt = q.absmax_quantize(rng.standard_normal((16, 8)))
        q.assert_integral(qt.q, where="test")

    def test_respects_int8_range(self) -> None:
        rng = np.random.default_rng(2)
        qt = q.absmax_quantize(rng.standard_normal((16, 8)) * 1e6)
        assert qt.q.min() >= -128 and qt.q.max() <= 127

    def test_per_column_scale_beats_per_tensor(self) -> None:
        """The reason the scale is per column, not shared."""
        rng = np.random.default_rng(3)
        w = rng.standard_normal((256, 4)) * np.array([1.0, 10.0, 100.0, 0.01])
        per_col = np.abs(q.dequantize(w, axis=0) - w).mean(axis=0)
        flat = np.abs(q.dequantize(w, axis=None) - w).mean(axis=0)
        assert per_col.max() < flat.max()

    def test_zero_tensor_is_safe(self) -> None:
        qt = q.absmax_quantize(np.zeros((4, 4)))
        assert np.all(qt.dequantize() == 0.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            q.absmax_quantize(np.zeros((0, 4)))

    def test_nbytes_reports_packed_size(self) -> None:
        qt = q.absmax_quantize(np.zeros((32, 8)))
        assert qt.nbytes == 32 * 8

    def test_astype_int(self) -> None:
        qt = q.absmax_quantize(np.zeros((4, 4)))
        assert qt.astype_int().dtype == np.int8


class TestStraightThrough:
    def test_ste_round_is_rint_in_forward(self) -> None:
        x = np.array([0.4, 0.6, -1.5, 2.5])
        assert np.array_equal(q.ste_round(x), np.rint(x))

    def test_ste_quantize_matches_hard_quantize(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.standard_normal((32, 8))
        assert np.allclose(
            q.ste_quantize(x, axis=-1), q.absmax_quantize(x, axis=-1).dequantize()
        )

    def test_ste_preserves_value_exactly(self) -> None:
        """The point of the STE: y == x, so the graph stays connected."""
        x = np.array([0.37, -1.21, 4.4])
        assert np.allclose(q.ste_quantize(x, axis=0), x, atol=1e-6)

    def test_ternary_ste_is_exactly_ternary(self) -> None:
        rng = np.random.default_rng(5)
        w = rng.standard_normal((64, 8))
        out = q.ternary_quantize(w)
        assert set(np.unique(out)) <= {-1.0, 0.0, 1.0}

    def test_ternary_threshold_controls_density(self) -> None:
        rng = np.random.default_rng(6)
        w = rng.standard_normal((256, 8))
        dense = np.count_nonzero(q.ternary_quantize(w, threshold=0.0))
        sparse = np.count_nonzero(q.ternary_quantize(w, threshold=1.0))
        assert sparse < dense


class TestTernaryPacking:
    def test_packs_four_per_byte(self) -> None:
        assert q.pack_ternary(np.zeros((10, 8))).shape == (10, 2)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_pack_unpack_round_trip(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        v = rng.choice([-1.0, 0.0, 1.0], size=(16, 12))
        assert np.array_equal(q.unpack_ternary(q.pack_ternary(v)), v)

    def test_footprint_is_two_bits_per_weight(self) -> None:
        """The spec's claim: 4 weights per byte."""
        v = np.ones((4, 40))
        assert q.pack_ternary(v).nbytes == v.size / 4

    def test_rejects_non_ternary_values(self) -> None:
        with pytest.raises(ValueError, match=r"\{-1, 0, \+1\}"):
            q.pack_ternary(np.array([0.5]))

    def test_rejects_unaligned_last_axis(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            q.pack_ternary(np.zeros((2, 6)))


class TestInt4Packing:
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        v = rng.integers(-8, 8, size=(8, 16)).astype(np.int8)
        assert np.array_equal(q.unpack_int4(q.pack_int4(v)), v)

    def test_two_per_byte(self) -> None:
        assert q.pack_int4(np.zeros((4, 8), dtype=np.int8)).shape == (4, 4)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[-8, 7\]"):
            q.pack_int4(np.array([9, -9], dtype=np.int8))

    def test_negative_values_survive(self) -> None:
        v = np.array([-8, -1, 0, 7], dtype=np.int8)
        assert np.array_equal(q.unpack_int4(q.pack_int4(v)), v)


class TestFixedPoint:
    def test_q_round_trip(self) -> None:
        x = np.linspace(-1.0, 1.0, 33)
        assert np.allclose(q.from_q(q.to_q(x)), x, atol=1.0 / q.Q_ONE)

    def test_qmul_is_fixed_point_multiply(self) -> None:
        a, b = 0.5, 0.25
        assert float(q.from_q(q.qmul(q.to_q(a), q.to_q(b)))) == pytest.approx(
            a * b, abs=1e-4
        )

    def test_qmul_of_zero_is_zero(self) -> None:
        assert np.all(q.qmul(q.to_q(np.zeros(4)), q.to_q(np.ones(4))) == 0)

    def test_saturate_clamps_to_int8(self) -> None:
        out = q.saturate_int8(np.array([-1000, -128, 0, 127, 1000]))
        assert out.tolist() == [-128, -128, 0, 127, 127]

    def test_saturation_is_monotone(self) -> None:
        """Clamping, not wrapping: a wrapped int8 turns a large positive
        accumulator into a large negative one, a silent unbounded bug."""
        x = np.arange(-300, 300)
        assert np.all(np.diff(q.saturate_int8(x).astype(np.int64)) >= 0)

    def test_quantize_activation_spans_int8(self) -> None:
        out = q.quantize_activation(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        assert out.tolist() == [-127, -127, 0, 127, 127]


class TestAssertIntegral:
    def test_passes_on_integers(self) -> None:
        q.assert_integral(np.array([1.0, 2.0, -3.0]))

    def test_raises_on_fractional(self) -> None:
        with pytest.raises(AssertionError, match="not integral"):
            q.assert_integral(np.array([1.5]))
