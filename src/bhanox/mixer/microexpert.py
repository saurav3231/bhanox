"""MicroExpert mixer: fine-grained sparse feed-forward with top-2 routing.

Purpose: give the model more capacity per active byte than a dense FFN, by
keeping every expert tiny and activating only two of them per token.

In simple words: instead of one big matrix that always runs, keep a cupboard of
small matrices and pick the two that suit this word. A shared expert always
runs, so nothing is unreachable.

Architecture (spec D3, frozen)::

    g = softmax(E x + b)                 E: (n_experts, d_model)
    {j1, j2} = top-2(g), renormalised
    y = MLP_shared(x) + g1 MLP_j1(x) + g2 MLP_j2(x)
    MLP_j(x) = W2_j act(W1_j x),  act = LUT-quantised GELU approx

Softmax routing beat sigmoid in the design-phase tournament (3.0611 vs 3.0801
BPC), so softmax is the default and the load-balancing machinery is built for
it.

Bytes touched per token: ``(n_shared + top_k) * (2 * d_model * d_expert)``
int8 weights, versus ``2 * d_model * (n_experts + n_shared) * d_expert`` for a
dense FFN of the same total capacity. At nano that is 5.3x fewer bytes, and the
gap widens with scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bhanox.config import BhanoxConfig
from bhanox.quant.numerics import absmax_quantize
from bhanox.seeding import init_rng

__all__ = ["MicroExpertLayer", "gelu_lut", "top2_balanced"]


def gelu_lut(n_entries: int = 256, *, bits: int = 16) -> NDArray[np.float32]:
    """LUT-quantised GELU approximation table.

    Args:
        n_entries: Table size. Capped at 256 by invariant I3.
        bits: Fixed-point fraction bits for the table entries.

    Returns:
        ``float32`` table of ``gelu`` sampled on ``[-8, 8]``.

    Raises:
        ValueError: If ``n_entries`` exceeds 256.

    Why a LUT: ``exp`` is a floating-point op and is not in the I3 whitelist.
    A 256-entry table indexed by a saturating shift is the sanctioned
    replacement, and it is also faster on a CPU than an accurate ``exp``.
    """
    if n_entries > 256:
        raise ValueError(f"gelu_lut size {n_entries} exceeds the 256-entry bound (I3)")
    x = np.linspace(-8.0, 8.0, n_entries)
    from math import erf  # local import: keeps the module import graph flat

    y = 0.5 * x * (1.0 + np.array([erf(float(v) / np.sqrt(2.0)) for v in x]))
    # float32, not float64: the LUT is looked up once per token per layer, so a
    # float64 table doubles its footprint for precision the int8 regime discards
    # anyway.
    return (np.rint(y * (1 << bits)) / (1 << bits)).astype(np.float32)


_GELU = gelu_lut(256)


def gelu_lut_apply(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply the GELU LUT with linear interpolation between entries.

    Args:
        x: Input array.

    Returns:
        ``gelu(x)`` approximated to within the LUT's step size.
    """
    lo, hi = -8.0, 8.0
    n = _GELU.size
    pos = (np.clip(x, lo, hi) - lo) / (hi - lo) * (n - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
    frac = (pos - i0).astype(np.float32)
    return (_GELU[i0] * (1.0 - frac) + _GELU[i0 + 1] * frac).astype(np.float32)


def top2_balanced(
    scores: NDArray[np.floating], top_k: int
) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
    """Pick the top-k experts per token and renormalise their gates.

    Args:
        scores: Routing scores, post-softmax, with experts on the **last** axis.
            Any leading shape is accepted, so ``(n_tokens, n_experts)`` and
            ``(batch, seq, n_experts)`` both work. Sorting on the last axis
            rather than a hard-coded ``[:, :top_k]`` is what makes that true: a
            3-D input would otherwise have its *time* axis sliced and silently
            return the wrong experts.
        top_k: Experts to activate per token.

    Returns:
        ``(indices, weights)``, both with shape ``scores.shape[:-1] + (top_k,)``,
        and ``weights`` summing to 1 along the last axis.

    Why renormalise: a soft top-k that does not renormalise silently rescales
    the whole layer output by the retained gate mass, so the mixer quietly
    changes layer scale depending on how peaked routing happens to be.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k > scores.shape[-1]:
        raise ValueError(f"top_k={top_k} exceeds {scores.shape[-1]} experts")
    order = np.argsort(-scores, axis=-1, kind="stable")[..., :top_k]
    picked = np.take_along_axis(scores, order, axis=-1)
    total = picked.sum(axis=-1, keepdims=True)
    weights = picked / np.maximum(total, 1e-6)
    return order.astype(np.int64), weights.astype(np.float32)


@dataclass
class MicroExpertLayer:
    """Top-2 routed fine-grained MoE with always-on shared experts.

    Attributes:
        config: Owning model config.
        E: Router weights, shape ``(n_experts, d_model)``.
        b: Router bias, shape ``(n_experts,)``. Updated by load balancing.
        W1, W2: Expert up/down projections. Stored as
            ``(n_total, d_expert, d_model)`` and ``(n_total, d_model, d_expert)``
            so the active experts can be gathered, not strided.
        loads: Lifetime per-expert activation count, used for the bias
            auto-correction and reported as load entropy.
        last_entropy: Load entropy of the most recent call, in nats.
    """

    config: BhanoxConfig
    layer_index: int = 0
    E: NDArray[np.floating] = field(init=False)
    b: NDArray[np.floating] = field(init=False)
    W1: NDArray[np.floating] = field(init=False)
    W2: NDArray[np.floating] = field(init=False)
    loads: NDArray[np.int64] = field(init=False)
    last_entropy: float = 0.0

    def __post_init__(self) -> None:
        """Allocate router and expert weights in the int8 regime."""
        cfg = self.config
        rng = init_rng(cfg.seed, "moe", cfg.name, self.layer_index)
        n = cfg.total_experts

        def stack(rows: int, cols: int) -> NDArray[np.float32]:
            w = rng.standard_normal((n, rows, cols)) / np.sqrt(rows)
            return absmax_quantize(w, axis=-1).dequantize().astype(np.float32)

        # Weights are stored *dequantized*, not as int8 codes. The codes are
        # +/-127, and a float matmul against a 127x-scaled weight matrix gives
        # router logits with a spread of ~500, which saturates the softmax: the
        # router then picks the same two experts forever and the load-balancing
        # bias (order 1) can never move it. The int8 regime is a property of the
        # deployed op sequence -- see bhanox.audit.INFERENCE_OPS -- not a
        # licence to run the reference forward pass on unscaled codes.
        self.E = (
            absmax_quantize(
                rng.standard_normal((cfg.n_experts, cfg.d_model))
                / np.sqrt(cfg.d_model),
                axis=0,
            )
            .dequantize()
            .astype(np.float32)
        )
        self.b = np.zeros(cfg.n_experts, dtype=np.float32)
        self.W1 = stack(cfg.d_expert, cfg.d_model)
        self.W2 = stack(cfg.d_model, cfg.d_expert)
        self.loads = np.zeros(n, dtype=np.int64)
        self.last_entropy = 0.0

    # -- routing -------------------------------------------------------------

    def route(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Softmax routing scores.

        Args:
            x: ``(n_tokens, d_model)`` inputs.

        Returns:
            ``(n_tokens, n_experts)`` probabilities summing to 1 per row.
        """
        logits = np.asarray(x, dtype=np.float32) @ self.E.T + self.b
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return (p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-6)).astype(np.float32)

    def update_load_bias(
        self, indices: NDArray[np.int64], *, gamma: float = 1e-3
    ) -> NDArray[np.float64]:
        """Auto-correct the router bias toward uniform load.

        Args:
            indices: ``(n_tokens, top_k)`` selected expert indices.
            gamma: Correction rate.

        Returns:
            Per-expert load fraction over routed experts, in ``[0, 1]``.

        Why this exists: top-2 routing is a winner-take-all process, so a
        slightly early advantage compounds and most experts starve. Nudging
        ``b`` down for over-loaded experts is the cheapest fix that needs no
        auxiliary loss term and no extra forward pass.

        Why the counts are per call and not lifetime: a lifetime average cannot
        correct an imbalance that happened once -- it is permanently baked in --
        and it makes the correction weaker the longer training runs, which is
        backwards. Lifetime counts are still accumulated in ``loads``, because
        that is what load *reporting* wants.
        """
        flat = indices.reshape(-1)
        np.add.at(self.loads, flat, 1)
        counts = np.bincount(flat, minlength=self.config.n_experts)
        fraction = counts / max(int(flat.size), 1)
        self.b -= gamma * (fraction - fraction.mean())
        return fraction

    def load_entropy(self) -> float:
        """Entropy of the per-expert load distribution, in nats.

        Returns:
            Entropy in nats. ``log(n_experts)`` means perfectly balanced.
        """
        routed = self.loads[: self.config.n_experts].astype(np.float64)
        total = routed.sum()
        if total == 0:
            return 0.0
        p = routed / total
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    # -- forward -------------------------------------------------------------

    def forward(
        self, x: NDArray[np.floating], *, train: bool = False
    ) -> NDArray[np.floating]:
        """Mix shared and top-2 routed experts.

        Args:
            x: ``(..., d_model)`` inputs. Any leading shape is accepted: a
                single ``(d_model,)`` vector is one token, ``(n_tokens, d_model)``
                is a batch, and ``(B, T, d_model)`` is a batch of sequences.
            train: Apply the load-bias update and record loads.

        Returns:
            Output of the same shape as ``x``.

        Why the active-set gather: computing every expert would be correct and
        would also make the byte count identical to a dense FFN, which is the
        entire reason this layer exists. Gathering only the routed experts is
        what makes the I2 audit show the saving.

        Why the flatten: the token axis is flattened to 2-D once here, so every
        helper below can assume ``(n_tokens, d_model)`` exactly as documented.
        Doing it per-helper instead would mean re-deriving index arithmetic in
        each one, and that is how the ``[:, slot]`` bug below happened.
        """
        arr = np.asarray(x, dtype=np.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        cfg = self.config
        n_shared = cfg.n_shared_experts

        out = np.zeros_like(flat)
        for j in range(n_shared):
            out += self._expert(flat, j)

        scores = self.route(flat)
        indices, weights = top2_balanced(scores, cfg.top_k)
        if train:
            self.update_load_bias(indices)
            p = scores.mean(axis=0)
            nz = p[p > 0]
            self.last_entropy = float(-(nz * np.log(nz)).sum())
        for slot in range(cfg.top_k):
            out += self._routed(flat, indices[:, slot], weights[:, slot], n_shared)
        return out.reshape(arr.shape)

    def _expert(self, x: NDArray[np.floating], j: int) -> NDArray[np.floating]:
        """Apply one expert to every token."""
        h = gelu_lut_apply(x @ self.W1[j].T)
        return h @ self.W2[j].T

    def _routed(
        self,
        x: NDArray[np.floating],
        idx: NDArray[np.int64],
        w: NDArray[np.floating],
        n_shared: int,
    ) -> NDArray[np.floating]:
        """Apply each token's selected expert, gathering only active weights.

        Args:
            x: ``(n_tokens, d_model)`` inputs.
            idx: ``(n_tokens,)`` expert index per token.
            w: ``(n_tokens,)`` gate weight per token.
            n_shared: Offset from routed index to storage index.

        Returns:
            ``(n_tokens, d_model)`` weighted mixture.

        Why the transpose on ``W1g``: ``W1`` is stored expert-major as
        ``(d_expert, d_model)`` so the gather is contiguous, but the matmul
        needs it token-major as ``(d_model, d_expert)``.
        """
        rows = idx + n_shared
        w1g = self.W1[rows].transpose(0, 2, 1)
        w2g = self.W2[rows]
        h = gelu_lut_apply(np.einsum("td,tdk->tk", x, w1g, optimize=True))
        y = np.einsum("tk,tdk->td", h, w2g, optimize=True)
        return y * w[:, None]

    # -- introspection -------------------------------------------------------

    def active_nbytes(self) -> int:
        """int8 weight bytes touched per token.

        This is the number the 5.3x MicroExpert claim is measured from, and it
        is what :mod:`bhanox.audit` reports for invariant I2.
        """
        cfg = self.config
        per_expert = 2 * cfg.d_model * cfg.d_expert
        return int((cfg.n_shared_experts + cfg.top_k) * per_expert)

    def dense_nbytes(self) -> int:
        """int8 weight bytes a dense FFN of equal capacity would touch."""
        cfg = self.config
        return int(2 * cfg.d_model * cfg.d_expert * cfg.total_experts)

    def param_count(self) -> int:
        """Total stored parameter values."""
        return int(self.E.size + self.b.size + self.W1.size + self.W2.size)


def _demo() -> None:
    """Self-check: top-2 routing is correct and renormalised."""
    scores = np.array([[0.5, 0.3, 0.15, 0.05]], dtype=np.float32)
    idx, w = top2_balanced(scores, 2)
    assert idx.tolist() == [[0, 1]], idx
    assert abs(float(w.sum()) - 1.0) < 1e-6, w
    print(f"I3 mixer OK: top-2 of {scores.shape[-1]} experts, weights {w.tolist()}")


if __name__ == "__main__":
    _demo()
