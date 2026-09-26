"""Tests for the MicroExpert sparse mixer."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.mixer.microexpert import (
    MicroExpertLayer,
    gelu_lut_apply,
    top2_balanced,
)

CFG = load_config("nano")


class TestTop2Balanced:
    def test_picks_the_highest_scores(self) -> None:
        scores = np.array([[0.5, 0.3, 0.15, 0.05]], dtype=np.float32)
        idx, _ = top2_balanced(scores, 2)
        assert idx.tolist() == [[0, 1]]

    def test_weights_are_renormalised(self) -> None:
        scores = np.array([[0.5, 0.3, 0.15, 0.05]], dtype=np.float32)
        _, w = top2_balanced(scores, 2)
        assert float(w.sum()) == pytest.approx(1.0)
        assert w[0].tolist() == pytest.approx([0.625, 0.375])

    def test_experts_are_on_the_last_axis(self) -> None:
        """A 3-D score tensor is a batch of sequences. Slicing axis 1 instead of
        the expert axis returns the wrong experts and still looks plausible."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((2, 5, 8))
        e = np.exp(raw - raw.max(axis=-1, keepdims=True))
        scores = e / e.sum(axis=-1, keepdims=True)
        idx, w = top2_balanced(scores, 2)
        assert idx.shape == (2, 5, 2)
        assert w.shape == (2, 5, 2)
        assert np.allclose(w.sum(axis=-1), 1.0)
        for b in range(2):
            for t in range(5):
                row = scores[b, t]
                assert set(idx[b, t].tolist()) == set(np.argsort(-row)[:2].tolist())

    def test_ties_are_broken_stably(self) -> None:
        scores = np.full((1, 4), 0.25, dtype=np.float32)
        assert top2_balanced(scores, 2)[0].tolist() == [[0, 1]]

    def test_rejects_bad_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            top2_balanced(np.ones((1, 4), np.float32), 0)
        with pytest.raises(ValueError, match="exceeds"):
            top2_balanced(np.ones((1, 4), np.float32), 9)


class TestGelu:
    def test_matches_the_definition(self) -> None:
        x = np.linspace(-3, 3, 25).astype(np.float32)
        from math import erf, sqrt

        want = 0.5 * x * (1.0 + np.array([erf(float(v) / sqrt(2.0)) for v in x]))
        assert np.allclose(gelu_lut_apply(x), want, atol=2e-3)

    def test_interpolation_beats_nearest_entry(self) -> None:
        """The design phase found that snapping to the nearest LUT entry cost
        4.5% accuracy. Linear interpolation is what recovers it, so the
        difference is measured against a real nearest-entry lookup rather than
        assumed."""
        from bhanox.mixer.microexpert import _GELU

        x = np.linspace(-6, 6, 401).astype(np.float32)
        lo, hi, n = -8.0, 8.0, _GELU.size
        pos = (np.clip(x, lo, hi) - lo) / (hi - lo) * (n - 1)
        nearest = _GELU[np.clip(np.rint(pos).astype(np.int64), 0, n - 1)]

        from math import erf, sqrt

        exact = 0.5 * x * (1.0 + np.array([erf(float(v) / sqrt(2.0)) for v in x]))
        err_interp = float(np.abs(gelu_lut_apply(x) - exact).max())
        err_nearest = float(np.abs(nearest - exact).max())
        assert err_interp < 1e-3
        assert err_interp < err_nearest / 5.0

    def test_handles_extremes(self) -> None:
        assert np.all(np.isfinite(gelu_lut_apply(np.array([-50.0, 50.0], np.float32))))


class TestLayer:
    def test_shape_for_a_single_token(self) -> None:
        m = MicroExpertLayer(CFG)
        out = m.forward(np.ones(CFG.d_model, dtype=np.float32))
        assert out.shape == (CFG.d_model,)

    def test_shape_for_a_batch(self) -> None:
        m = MicroExpertLayer(CFG)
        assert m.forward(np.ones((7, CFG.d_model), np.float32)).shape == (
            7,
            CFG.d_model,
        )

    def test_shape_for_a_batch_of_sequences(self) -> None:
        m = MicroExpertLayer(CFG)
        out = m.forward(np.ones((2, 3, CFG.d_model), np.float32))
        assert out.shape == (2, 3, CFG.d_model)

    def test_token_results_are_independent(self) -> None:
        """Batching must not couple tokens: the router, the expert gather and
        the load-balancing statistics are all per token."""
        m = MicroExpertLayer(CFG)
        one = np.ones((1, CFG.d_model), np.float32)
        two = np.repeat(one, 5, axis=0)
        assert np.allclose(m.forward(one)[0], m.forward(two)[0], atol=1e-5)

    def test_shared_experts_always_contribute(self) -> None:
        m = MicroExpertLayer(CFG)
        out = m.forward(np.ones((4, CFG.d_model), np.float32))
        shared = sum(
            m._expert(np.ones((4, CFG.d_model), np.float32), j)
            for j in range(CFG.n_shared_experts)
        )
        routed = out - shared
        assert np.abs(routed).sum() > 0

    def test_touches_far_fewer_bytes_than_dense(self) -> None:
        m = MicroExpertLayer(CFG)
        assert m.dense_nbytes() / m.active_nbytes() > 4.0

    def test_active_nbytes_matches_the_shape(self) -> None:
        m = MicroExpertLayer(CFG)
        per = 2 * CFG.d_model * CFG.d_expert
        assert m.active_nbytes() == (CFG.n_shared_experts + CFG.top_k) * per

    def test_param_count_matches_stored_arrays(self) -> None:
        m = MicroExpertLayer(CFG)
        assert m.param_count() == (m.E.size + m.b.size + m.W1.size + m.W2.size)

    def test_weights_are_not_int8_codes(self) -> None:
        """The mixer stores real-valued weights, not raw int8 codes.

        Codes are +/-127, and a float matmul against a 127x-scaled matrix gives
        router logits with a spread of ~500, which saturates the softmax into
        always picking the same two experts. The int8 regime is a property of
        the deployed op sequence, not of the reference's storage dtype.
        """
        m = MicroExpertLayer(CFG)
        assert np.abs(m.E).max() < 1.0
        assert np.abs(m.W1).max() < 1.0


class TestLoadBalancing:
    def test_train_mode_updates_the_bias(self) -> None:
        m = MicroExpertLayer(CFG)
        before = m.b.copy()
        m.forward(np.ones((16, CFG.d_model), np.float32), train=True)
        assert not np.array_equal(before, m.b)

    def test_inference_mode_leaves_the_bias_alone(self) -> None:
        m = MicroExpertLayer(CFG)
        before = m.b.copy()
        m.forward(np.ones((16, CFG.d_model), np.float32), train=False)
        assert np.array_equal(before, m.b)

    def test_loads_are_recorded(self) -> None:
        """Only the routed experts are counted. Shared experts are always on, so
        counting them would dilute the routed-expert statistics that the bias
        update is supposed to equalise."""
        m = MicroExpertLayer(CFG)
        m.forward(np.ones((16, CFG.d_model), np.float32), train=True)
        assert m.loads[: CFG.n_experts].sum() == 16 * CFG.top_k

    def test_entropy_is_recorded(self) -> None:
        m = MicroExpertLayer(CFG)
        m.forward(np.ones((16, CFG.d_model), np.float32), train=True)
        assert 0.0 <= m.last_entropy <= np.log(CFG.n_experts)

    def test_routing_spreads_across_experts(self) -> None:
        """With genuinely varied inputs the router must not collapse.

        The collapse case is *identical* tokens: they necessarily route the same
        way, so an all-ones batch says nothing about balancing.
        """
        m = MicroExpertLayer(CFG)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, CFG.d_model)).astype(np.float32)
        for _ in range(20):
            m.forward(x, train=True)
        assert m.load_entropy() > 0.9 * np.log(CFG.n_experts)

    def test_bias_update_penalises_the_loaded_experts(self) -> None:
        """The mechanism itself, at a gamma strong enough to see in one step."""
        m = MicroExpertLayer(CFG)
        hot = np.array([[0, 1]], dtype=np.int64)
        before = m.b.copy()
        m.update_load_bias(hot, gamma=1.0)
        assert m.b[0] < before[0] and m.b[1] < before[1]
        assert m.b[2:].mean() > before[2:].mean()

    def test_entropy_is_higher_for_a_flat_router(self) -> None:
        m = MicroExpertLayer(CFG)
        m.E = np.zeros_like(m.E)
        m.forward(np.ones((32, CFG.d_model), np.float32), train=True)
        assert m.last_entropy > np.log(CFG.n_experts) - 0.5
