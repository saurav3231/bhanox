"""Tests for the DeltaBank core, including the Phase-7 tournament regressions.

The tournament findings this file pins down (design phase, 23 variants):
  1. K/Q L2 normalisation is mandatory -- without it beta diverges.
  2. Read-before-write beats read-after-write (3.1251 vs 3.1623 BPC).
  3. The read gate is worth +0.037 BPC.
  4. The delta rule beats additive writes (recall 1.000 vs 0.815).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bhanox.config import load_config
from bhanox.core.deltabank import (
    DeltaBankHead,
    l2_normalize,
    recip_lut,
)
from bhanox.core.deltabank_layer import DeltaBankLayer
from bhanox.quant.numerics import INT8_MAX

D_IN, D_K, D_V, BANKS = 32, 64, 32, 8
RATES = np.array([1.0 - 2.0**-b for b in range(1, BANKS + 1)], dtype=np.float32)


def make_head(**kwargs) -> DeltaBankHead:
    """Build a head with deterministic weights for a repeatable test."""
    head = DeltaBankHead(d_k=D_K, d_v=D_V, d_in=D_IN, n_banks=BANKS)
    rng = np.random.default_rng(11)
    for name in ("W_k", "W_q", "W_v", "W_r"):
        w = rng.standard_normal(getattr(head, name).shape)
        setattr(head, name, (w / np.sqrt(w.shape[0])).astype(np.float32))
    for key, value in kwargs.items():
        setattr(head, key, value)
    return head


def train(head: DeltaBankHead, xs: np.ndarray, repeats: int = 10) -> DeltaBankHead:
    """Drive a head with the same associations repeatedly so it converges."""
    head.reset()
    for _ in range(repeats):
        for x in xs:
            head.forward(x, RATES)
    return head


def unit_rows(xs: np.ndarray) -> np.ndarray:
    return xs / np.linalg.norm(xs, axis=-1, keepdims=True)


def key_of(head: DeltaBankHead, x: np.ndarray) -> np.ndarray:
    k = x @ head.W_k
    return k / max(float(np.linalg.norm(k)), 1e-6)


def targets(head: DeltaBankHead, xs: np.ndarray) -> np.ndarray:
    return np.stack([np.clip(x @ head.W_v / INT8_MAX, -1, 1) for x in xs])


def recall_error(
    head: DeltaBankHead,
    xs: np.ndarray,
    wants: NDArray[np.floating] | None = None,
) -> float:
    """Mean max-abs error of retrieving each stored association.

    Args:
        head: The trained head.
        xs: Probe inputs, already in whatever regime the caller is testing.
        wants: Optional explicit targets, for tests that use inputs outside the
            unit regime that :func:`targets` assumes.
    """
    want = targets(head, xs) if wants is None else wants
    errs = []
    for x, w in zip(xs, want, strict=True):
        k = x @ head.W_k
        if head.normalize_keys:
            k = k / max(float(np.linalg.norm(k)), 1e-6)
        k8 = np.rint(k * INT8_MAX).astype(np.int32)
        # `state[0]`: the state is (B, d_k, d_v) since M2, and these probes
        # train a single stream, so sample 0 is the one under test.
        got = (head.state[0].T @ k8) / float(INT8_MAX**2)
        errs.append(float(np.abs(got - w).max()))
    return float(np.mean(errs))


@pytest.fixture
def xs() -> np.ndarray:
    rng = np.random.default_rng(3)
    return unit_rows(rng.standard_normal((6, D_IN)).astype(np.float32))


class TestHelpers:
    def test_l2_normalize_gives_unit_rows(self) -> None:
        out = l2_normalize(np.array([[3.0, 4.0]]))
        assert float(np.linalg.norm(out)) == pytest.approx(1.0)

    def test_l2_normalize_leaves_zero_row_alone(self) -> None:
        assert np.all(l2_normalize(np.zeros((1, 4))) == 0.0)

    def test_recip_lut_is_one_indexed(self) -> None:
        table = recip_lut(256)
        assert table[0] == pytest.approx(1.0)
        assert table[1] == pytest.approx(0.5)

    def test_recip_lut_respects_i3_bound(self) -> None:
        with pytest.raises(ValueError, match="256"):
            recip_lut(512)


class TestState:
    def test_state_starts_empty_and_resets(self) -> None:
        head = make_head()
        assert not np.any(head.state)
        head.forward(np.ones(D_IN, dtype=np.float32), RATES)
        assert np.any(head.state)
        head.reset()
        assert not np.any(head.state)

    def test_state_stays_inside_int8(self) -> None:
        head = make_head()
        rng = np.random.default_rng(5)
        for _ in range(300):
            head.forward(rng.standard_normal(D_IN).astype(np.float32), RATES)
            assert head.state.min() >= -128
            assert head.state.max() <= 127

    def test_state_is_integral(self) -> None:
        """The int8 regime: a fractional state means a float op leaked in."""
        head = make_head()
        rng = np.random.default_rng(6)
        for _ in range(50):
            head.forward(rng.standard_normal(D_IN).astype(np.float32), RATES)
        assert np.all(head.state == np.rint(head.state))

    def test_state_nbytes_is_constant(self) -> None:
        head = make_head()
        before = head.state_nbytes
        rng = np.random.default_rng(7)
        for _ in range(100):
            head.forward(rng.standard_normal(D_IN).astype(np.float32), RATES)
        assert head.state_nbytes == before == D_K * D_V

    def test_counters_track_steps(self) -> None:
        head = make_head()
        for _ in range(4):
            head.forward(np.ones(D_IN, dtype=np.float32), RATES)
        assert head.reads == head.writes == 4

    def test_rejects_wrong_input_width(self) -> None:
        with pytest.raises(ValueError, match="d_in"):
            make_head().forward(np.ones(5, dtype=np.float32), RATES)


class TestRecall:
    def test_converges_towards_the_stored_value(self, xs: np.ndarray) -> None:
        head = train(make_head(), xs)
        assert recall_error(head, xs) < 0.7

    def test_error_converges_instead_of_drifting(self, xs: np.ndarray) -> None:
        """The delta rule reaches a fixed point in a few passes and then stays
        put. Repeating forever must not make it worse or better -- that is what
        "corrected on write" buys over "appended to"."""
        head = train(make_head(), xs, repeats=16)
        settled = recall_error(head, xs)
        train(head, xs, repeats=48)
        assert recall_error(head, xs) == pytest.approx(settled, abs=0.02)

    def test_recall_does_not_write(self, xs: np.ndarray) -> None:
        head = train(make_head(), xs)
        before = head.state.copy()
        head.recall(key_of(head, xs[0]))
        assert np.array_equal(before, head.state)


class TestPhase7Regressions:
    def test_delta_rule_beats_additive_writes(self, xs: np.ndarray) -> None:
        """Tournament T3, reproduced by the reference: recall 0.213 (delta) vs
        0.375 (additive).

        The design-phase tournament scored this 1.000 vs 0.815 on its own
        harness. The reference is an untrained, unoptimized head, so the
        absolute numbers are much lower -- what matters here is the sign, and
        the test asserts only that. Do not quote the tournament figures as this
        implementation's output.
        """
        delta = train(make_head(), xs)
        addy = make_head()
        addy.write_mode = "additive"
        train(addy, xs)
        assert recall_error(delta, xs) < recall_error(addy, xs)

    def test_additive_baseline_saturates_the_state(self, xs: np.ndarray) -> None:
        """Why it loses: plain accumulation has no error term to cancel the
        earlier writes, so it drives the int8 state into saturation."""
        addy = make_head()
        addy.write_mode = "additive"
        train(addy, xs, repeats=20)
        assert np.abs(addy.state).max() >= np.abs(train(make_head(), xs).state).max()

    def test_unnormalized_keys_break_the_update(self) -> None:
        """Finding 1. The pathology only shows up once ||k|| is actually far
        from 1: at unit input the projection is already near-unit, so the two
        settings agree. Scale the input until ||k|| ~ 12 and the missing
        normalisation is unmistakable, because the reciprocal LUT is then read
        at the wrong index and beta collapses toward zero."""
        big = 8.0 * unit_rows(np.random.default_rng(3).standard_normal((6, D_IN)))

        def run(normalize: bool) -> float:
            head = make_head()
            head.normalize_keys = normalize
            head.reset()
            for _ in range(16):
                for x in big:
                    head.forward(x, RATES)
            return recall_error(head, big)

        assert run(False) > 2 * run(True)

    def test_read_before_write_is_used_by_default(self, xs: np.ndarray) -> None:
        head = make_head()
        assert head.read_before_write

    def test_read_after_write_differs(self, xs: np.ndarray) -> None:
        """Finding 2: the two orderings must not be silently equivalent."""
        rbw = train(make_head(), xs)
        raw = make_head()
        raw.read_before_write = False
        train(raw, xs)
        assert not np.allclose(rbw.state, raw.state)

    def test_read_gate_changes_the_output(self, xs: np.ndarray) -> None:
        """Finding 3: the gate is a real component, not a no-op. It can only
        show up once the state holds something -- gating a zero read is
        multiplying zero by anything."""
        gated = train(make_head(), xs)
        plain = train(make_head(), xs)
        plain.use_read_gate = False
        probe = xs[0]
        assert not np.allclose(gated.forward(probe, RATES), plain.forward(probe, RATES))

    def test_decay_banks_outlast_a_single_fast_decay(self) -> None:
        """Tournament T2, reproduced by the reference: after 8 steps of the same
        probe, total state energy is 5,430 (banks) vs 2,456 (single fast decay),
        a 2.2x difference.

        The tournament reported this as "worst-bin retention 0.94 vs 0.16". That
        metric is not computed here -- this test measures total state energy,
        which is the property that actually matters (a decayed state is a state
        that has forgotten). The tournament figure is not this implementation's
        output.
        """
        banks = make_head()
        single = make_head()
        single.bank_logits[:] = 0.0
        single.bank_logits[0] = 20.0  # force the fastest bank only
        single.bank_logits[1:] = -20.0
        probe = np.ones(D_IN, dtype=np.float32) / np.sqrt(D_IN)
        for _ in range(8):
            banks.forward(probe, RATES)
            single.forward(probe, RATES)
        banks_energy = float(np.abs(banks.state).sum())
        single_energy = float(np.abs(single.state).sum())
        assert banks_energy > single_energy
        # Pinned so the docs' quoted ratio cannot drift silently.
        assert banks_energy / single_energy == pytest.approx(2.21, abs=0.01)

    def test_decay_rate_is_a_convex_mix_of_the_prior(self) -> None:
        head = make_head()
        lam = head.decay(RATES)
        assert lam.shape == (D_K,)
        assert lam.min() >= RATES.min() - 1e-6
        assert lam.max() <= RATES.max() + 1e-6

    def test_decay_q_matches_decay(self) -> None:
        head = make_head()
        assert np.allclose(head.decay_q(RATES) / 65536.0, head.decay(RATES), atol=1e-4)

    def test_banking_logits_move_the_decay(self) -> None:
        head = make_head()
        before = head.decay(RATES).copy()
        head.bank_logits[-1] += 5.0
        assert head.decay(RATES).mean() > before.mean()


class TestLayer:
    def test_forward_shape(self) -> None:
        """The layer's output must be d_model, or the residual stream changes
        width at the first memory layer and every later layer is a shape error."""
        cfg = load_config("nano")
        out = DeltaBankLayer(cfg).forward(np.ones(cfg.d_model, dtype=np.float32))
        assert out.shape == (cfg.d_model,)

    def test_reset_clears_all_heads(self) -> None:
        cfg = load_config("nano")
        layer = DeltaBankLayer(cfg)
        x = np.ones(cfg.d_model, dtype=np.float32)
        layer.forward(x)
        assert all(np.any(h.state) for h in layer.heads)
        layer.reset()
        assert all(not np.any(h.state) for h in layer.heads)

    def test_state_nbytes_is_heads_times_one_head(self) -> None:
        cfg = load_config("nano")
        layer = DeltaBankLayer(cfg)
        assert layer.state_nbytes == cfg.n_heads * cfg.d_k * cfg.d_v

    def test_bypass_is_not_duplicated_per_head(self) -> None:
        """G is per-timestep in the spec, so a per-head copy would be a 4x
        parameter duplication at nano for no representational gain."""
        cfg = load_config("nano")
        assert DeltaBankLayer(cfg).G.shape == (cfg.d_model, cfg.d_model)

    def test_param_count_matches_stored_arrays(self) -> None:
        layer = DeltaBankLayer(load_config("nano"))
        stored = layer.W_o.size + layer.G.size
        for h in layer.heads:
            stored += h.W_k.size + h.W_q.size + h.W_v.size + h.W_r.size
            stored += h.bank_logits.size
        assert layer.param_count() == stored

    def test_rejects_wrong_width(self) -> None:
        layer = DeltaBankLayer(load_config("nano"))
        with pytest.raises(ValueError, match="d_model"):
            layer.forward(np.ones(3, dtype=np.float32))
