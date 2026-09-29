"""Production numerical scales: does the int8 regime behave as designed?

Every other numeric check in the suite either uses a hand-built component or
replaces the weights with something convenient. This one uses production
initialisation and drives real byte 4-grams through the real front-end and the
real forward pass, because the defect it guards was invisible to every other
test in two separate ways:

1. ``docs/architecture.md`` already records this exact failure for MicroExpert
   -- "a float matmul against a 127x-scaled matrix gives router logits with a
   spread of ~500, which saturates the softmax". The same bug was live in
   :meth:`DeltaBankHead.proj` and :meth:`HashBind.quantize`, which kept
   ``absmax_quantize(...).q`` and dropped the per-column scale.
2. Nothing that checked *agreement* could see it. The Torch mirror copies the
   NumPy weights, so both sides inherited the same wrong scale, and I3's
   "no float dtype" test asserted the pool was integral -- a property the bug
   satisfied and the fix does not.

So the claims below are about the numbers the design actually cares about: the
value projection must land on the unit grid ``quantize_activation`` assumes, and
the read gate must not be pinned at 0 or 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.frontend.hashbind import encode_bytes
from bhanox.model import Bhanox, layer_norm
from bhanox.quant.numerics import INT8_MAX, quantize_activation

PROMPT = "the quick brown fox jumps over the lazy dog repeatedly today"


@pytest.fixture(scope="module")
def model() -> Bhanox:
    """A production Nano model, untouched weights, zero state."""
    return Bhanox(load_config("nano"))


@pytest.fixture(scope="module")
def ids() -> np.ndarray:
    """Real byte 4-gram ids from real text."""
    return encode_bytes(PROMPT)


def _residual_stream(model: Bhanox, ids: np.ndarray) -> np.ndarray:
    """The actual vector each DeltaBank layer receives, one row per token.

    Walks the real forward path -- embed, memory, mixer, layer norm -- because
    the projection scale is only meaningful against the activations that reach
    it. A synthetic unit vector would have passed the broken code too.
    """
    x = model.embed(ids[None, :]).astype(np.float32)
    rows = []
    for t in range(x.shape[1]):
        xx = x[0, t][None, :].copy()
        for bank, mixer, gate in zip(
            model.deltabanks, model.mixers, model.gates, strict=True
        ):
            step_out = bank.forward(xx)
            compute = gate.step(step_out, np.abs(step_out))
            xx = xx + np.where(compute, step_out, 0.0)
            xx = xx + mixer.forward(xx)
            xx = layer_norm(xx)
        rows.append(xx[0].copy())
    return np.stack(rows)


class TestProjectionsAreOnTheUnitGrid:
    """``X @ W_v`` must land where ``quantize_activation`` can represent it."""

    def test_value_projections_are_not_already_out_of_range(
        self, model: Bhanox, ids: np.ndarray
    ) -> None:
        """The median |v| must be inside the grid, not 400x past its edge.

        ``quantize_activation`` is ``clip(rint(x * 127), -127, 127)`` and its
        docstring says activations arrive "roughly in [-1, 1]". If the median
        projection is far outside that, the quantiser is a sign function and
        every value in the state carries one bit.
        """
        x = _residual_stream(model, ids)
        for li, layer in enumerate(model.deltabanks):
            for hi, head in enumerate(layer.heads):
                v = np.abs(x @ head.W_v)
                median = float(np.median(v))
                assert median < 1.0, (
                    f"L{li}H{hi}: median |v| = {median:.1f}. The stored W_v is "
                    f"int8 codes rather than dequantized weights, so the value "
                    f"path is saturating."
                )

    def test_the_value_grid_keeps_its_resolution(
        self, model: Bhanox, ids: np.ndarray
    ) -> None:
        """A working int8 activation path uses most of its 255 codes.

        Measured with the defect: 2-6 distinct values. A quantiser that emits
        six distinct numbers has not quantised anything.
        """
        x = _residual_stream(model, ids)
        for li, layer in enumerate(model.deltabanks):
            for hi, head in enumerate(layer.heads):
                v8 = quantize_activation(x @ head.W_v)
                distinct = len(np.unique(v8))
                assert (
                    distinct > 200
                ), f"L{li}H{hi}: only {distinct} distinct int8 codes out of 255"

    def test_the_read_gate_is_not_saturated(
        self, model: Bhanox, ids: np.ndarray
    ) -> None:
        """``sigmoid(x @ W_r)`` must not be pinned at 0 or 1.

        The spec credits the read gate with +0.037 BPC, and the architecture
        doc credits the MicroExpert router's saturation with the same root
        cause. A gate that is 99% saturated is a hard switch: the logits are
        past the point where tanh is linear, so the channel has no gradient to
        learn from and the model cannot turn it back on.
        """
        x = _residual_stream(model, ids)
        for li, layer in enumerate(model.deltabanks):
            for hi, head in enumerate(layer.heads):
                pre = x @ head.W_r
                gate = 0.5 * (1.0 + np.tanh(0.5 * pre))
                saturated = float(np.mean((gate < 0.01) | (gate > 0.99)))
                assert saturated < 0.01, (
                    f"L{li}H{hi}: {saturated:.1%} of read-gate channels are "
                    f"within 1% of 0 or 1 (pre-activation std "
                    f"{float(pre.std()):.1f})"
                )

    def test_every_projection_shares_one_scale_with_the_readout(
        self, model: Bhanox
    ) -> None:
        """``W_k/W_q/W_v/W_r`` must be on the same footing as ``W_o``/``G``.

        ``DeltaBankLayer`` dequantized its read-out long ago for exactly this
        reason and left a comment saying so. If the inputs are 127x and the
        read-out is 1x, the residual add is a replacement, which is the
        failure that comment warns about.
        """
        layer = model.deltabanks[0]
        readout = max(float(np.abs(layer.W_o).max()), float(np.abs(layer.G).max()))
        for hi, head in enumerate(layer.heads):
            for name in ("W_k", "W_q", "W_v", "W_r"):
                got = float(np.abs(getattr(head, name)).max())
                assert got < 5 * readout, (
                    f"L0H{hi}.{name} absmax {got:.1f} vs readout {readout:.1f}: "
                    f"the projections are not on the same scale"
                )


class TestQuantizeKeepsTheScale:
    """``HashBind.quantize`` must quantize, not rescale."""

    def test_the_embedding_magnitude_survives(
        self, model: Bhanox, ids: np.ndarray
    ) -> None:
        """Quantizing is allowed to change values, not the scale they feed.

        Measured with the defect: embedding RMS 0.070 -> 39.5, because the
        per-column scale (~1.3e-3) was discarded and the raw +/-127 codes were
        handed to ``embed`` as if they were real numbers. The next consumer is
        a residual-stream add, which has no scale to absorb that.
        """
        fresh = Bhanox(load_config("nano"))
        before = fresh.embedder.embed(ids).astype(np.float32)
        fresh.embedder.quantize()
        after = fresh.embedder.embed(ids).astype(np.float32)
        assert float(after.std()) == pytest.approx(float(before.std()), rel=0.02)

    def test_it_is_still_a_real_quantization(self, model: Bhanox) -> None:
        """The stored values are int8 codes, and the scale is stored with them.

        The array ``pool`` is what an int8 kernel consumes, so it has to be
        exactly integral. That was previously checked as a one-step fidelity
        bound against dequantized floats, which a float array sitting on the
        int8 grid satisfies while not being a quantization at all; and it was
        checked the other way too, as "every value is integral", which the raw
        unscaled codes satisfied while making the pool 283x-795x too large for
        ``embed`` to consume. Both halves are now named: ``pool`` is the codes,
        ``pool_scale`` is the factor, and dequantizing the two together
        reconstructs the real weights.
        """
        embedder = model.embedder
        original = embedder.pool.copy()
        embedder.quantize()

        # The stored pool is the int8 grid, exactly.
        assert np.array_equal(embedder.pool, np.rint(embedder.pool))
        assert embedder.pool.dtype == np.float32
        assert np.abs(embedder.pool).max() <= INT8_MAX

        # And dequantizing the codes against the stored scale returns the real
        # weights to within one int8 step per column -- the fidelity claim, now
        # stated on the dequantized view instead of on the stored codes.
        step = np.max(np.abs(original), axis=0, keepdims=True) / INT8_MAX
        reconstructed = embedder.dequantized_pool()
        assert reconstructed.shape == original.shape
        assert np.all(np.abs(reconstructed - original) <= step + 1e-12)

        # The scale is not a unit factor smuggled back in: it is the real
        # per-column absmax, so it is small, positive and column-varying.
        assert np.all(embedder.pool_scale > 0)
        assert embedder.pool_scale.max() < 1.0
        assert embedder.pool_scale.max() / embedder.pool_scale.min() > 1.0
