"""DeltaBankLayer: a bank of independent DeltaBank heads, assembled for a layer.

Purpose: hold the multi-head read-out and the shared bypass, so
:mod:`bhanox.core.deltabank` can stay about the delta rule itself.

In simple words: one DeltaBank is a memory. A layer is several of them, wired
together at the output.

Split out of ``deltabank.py`` for law C5 (one concern per module). The head is
the mechanism; this is the wiring. A re-export from ``deltabank`` would be
circular (this module already imports the head), so importers take
``DeltaBankLayer`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bhanox.config import BhanoxConfig
from bhanox.core.deltabank import DeltaBankHead
from bhanox.quant.numerics import absmax_quantize
from bhanox.seeding import init_rng

__all__ = ["DeltaBankLayer"]


@dataclass
class DeltaBankLayer:
    """A bank of independent :class:`DeltaBankHead` memories for one layer.

    Heads do not talk to each other. That is the point: independent memories
    give the model several timescales and several subspaces without any
    communication cost, and it is why adding heads is the sanctioned way to
    relieve invariant I1.

    Attributes:
        config: The owning model config, for shapes and the decay prior.
        heads: The head list, ``n_heads`` long.
        bank_rates: Frozen decay prior, shape ``(n_banks,)``.
        W_o: Stacked per-head read-outs, shape ``(n_heads * d_v, d_model)``. Each
            head's block is the spec's ``W_o,h``; stacking them is the same
            matrix as a batched matmul over the concatenated reads, and it keeps
            the layer's output at ``d_model`` so the residual stream never
            changes width.
        G: The shared gated bypass, shape ``(d_model, d_model)``. The spec
            subscripts this ``G_t`` (per timestep) while subscripting the
            read-out ``W_o,h`` (per head), so it is shared; a per-head copy would
            be a 4x parameter duplication at nano.
    """

    config: BhanoxConfig
    heads: list[DeltaBankHead] = field(init=False)
    bank_rates: NDArray[np.floating] = field(init=False)
    W_o: NDArray[np.floating] = field(init=False)
    G: NDArray[np.floating] = field(init=False)

    def __post_init__(self) -> None:
        """Build the heads, the read-out, the bypass, and the decay prior."""
        cfg = self.config
        self.bank_rates = np.asarray(cfg.decay_rates, dtype=np.float32)
        self.heads = [
            DeltaBankHead(
                d_k=cfg.d_k,
                d_v=cfg.d_v,
                d_in=cfg.d_model,
                n_banks=cfg.n_banks,
                seed=cfg.seed,
                name=cfg.name,
                # Without this every head drew the same stream and the layer
                # shipped four copies of one memory. See bhanox.seeding.
                head_index=h,
            )
            for h in range(cfg.n_heads)
        ]

        def q(rows: int, cols: int, site: str) -> NDArray[np.floating]:
            rng = init_rng(cfg.seed, site, cfg.name)
            w = rng.standard_normal((rows, cols)) / np.sqrt(rows)
            # Dequantized, not the raw int8 codes. The codes are +/-127, so a
            # float matmul against them makes the read-out about 500x larger
            # than the residual it is being *added* to, and `x = x +
            # DeltaBank(x)` silently becomes a replacement. layer_norm hides
            # the scale downstream, which is exactly why it went unnoticed.
            return absmax_quantize(w, axis=0).dequantize().astype(np.float32)

        self.W_o = q(cfg.n_heads * cfg.d_v, cfg.d_model, "db.readout")
        self.G = q(cfg.d_model, cfg.d_model, "db.bypass")

    def reset(self) -> None:
        """Reset every head."""
        for head in self.heads:
            head.reset()

    def ensure_batch(self, batch: int) -> None:
        """Grow every head's state to hold ``batch`` independent streams."""
        for head in self.heads:
            head.ensure_batch(batch)

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Run one token through every head, then read out and add the bypass.

        Args:
            x: ``(B, d_model)`` activations, or a single ``(d_model,)`` vector,
                which is treated as one sample.

        Returns:
            ``(B, d_model)`` -- the layer's contribution to the residual stream
            -- or ``(d_model,)`` for a single vector.

        Raises:
            ValueError: If ``x`` has the wrong width.
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.config.d_model:
            raise ValueError(
                f"DeltaBankLayer expected d_model={self.config.d_model}, "
                f"got {arr.shape[-1]}"
            )
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        self.ensure_batch(arr.shape[0])
        reads = np.concatenate(
            [h.forward(arr, self.bank_rates) for h in self.heads], axis=-1
        )
        # `reads @ W_o` is (B, n_heads*d_v) @ (n_heads*d_v, d_model). The bypass
        # is a left multiply for a single vector and a right multiply for a
        # batch, because the row-vector convention flips.
        out = reads @ self.W_o + arr @ self.G.T
        return out[0] if single else out

    @property
    def state_nbytes(self) -> int:
        """Total int8 state bytes for the layer."""
        return int(sum(h.state_nbytes for h in self.heads))

    def param_count(self) -> int:
        """Total stored parameter values in the layer."""
        total = int(self.W_o.size + self.G.size)
        for h in self.heads:
            total += int(
                h.W_k.size + h.W_q.size + h.W_v.size + h.W_r.size + h.bank_logits.size
            )
        return total
