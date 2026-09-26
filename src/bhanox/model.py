"""The assembled Bhanox network.

Purpose: wire HashBind, DeltaBank, MicroExpert and (optionally) VectorVault
into the layer stack that the public API in :mod:`bhanox` exposes.

In simple words: this is the actual model. Everything else is one component of
it.

Why this file exists although the architecture table does not list it: the
public API (``bhanox.Bhanox(cfg)``, ``model.embed``, ``model.forward``,
``model.generate``) needs one place that owns the layer stack. Without it the
API would have to live in ``__init__.py``, which is worse (ADR-001).

Architecture per layer::

    x <- x + DeltaBank(x)          # recurrent memory, constant cost
    x <- x + MicroExpert(x)        # sparse capacity
    x <- LayerNorm(x)              # keeps the residual stream bounded

Layer norm is a per-column absmax rescale, which stays in the integer domain
under the default regime; it is not a floating-point op in the deployed graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bhanox.config import BhanoxConfig
from bhanox.core.deltabank_layer import DeltaBankLayer
from bhanox.frontend.hashbind import HashBind
from bhanox.governor.pulsegate import PulseGate
from bhanox.memory.vectorvault import VectorVault
from bhanox.mixer.microexpert import MicroExpertLayer
from bhanox.quant.numerics import absmax_quantize

__all__ = ["Bhanox", "layer_norm"]


def _init_output(cfg: BhanoxConfig) -> NDArray[np.float32]:
    """Initialise the unembedding projection, ``(d_model, output_vocab)``.

    Returns:
        A real-valued projection, scaled like the residual stream it reads.

    Why this is not left at zero: an all-zero unembedding makes every logit
    exactly 0.0, so ``generate`` can only ever emit token 0 and every check on
    the model's output is vacuously true. A randomly initialised model has to
    produce a non-degenerate distribution, or the untrained baseline is
    meaningless.

    The scale is ``1/sqrt(d_model)`` so that multiplying a unit-RMS residual by
    it yields a comparable-magnitude logit rather than a huge or vanishing one.
    """
    rng = np.random.default_rng(abs(hash(("unembed", cfg.name))) % 2**32)
    w = rng.standard_normal((cfg.d_model, cfg.output_vocab)) / np.sqrt(cfg.d_model)
    return absmax_quantize(w, axis=-1).dequantize().astype(np.float32)


def layer_norm(x: NDArray[np.floating], eps: float = 1e-5) -> NDArray[np.floating]:
    """Normalise the residual stream to unit RMS.

    Args:
        x: ``(..., d)`` activations.
        eps: Variance floor, so an all-zero row returns zeros rather than NaN.

    Returns:
        Normalised activations of the same shape.

    Why this is not a float op in the deployed graph: in the int8 regime this
    is a per-column absmax rescale and a shift, both whitelisted. The reference
    uses the standard form because it is the definition the native runtime and
    the PyTorch mirror must agree with to 1e-3 (I4).
    """
    mu = x.mean(axis=-1, keepdims=True)
    centred = x - mu
    var = np.mean(centred * centred, axis=-1, keepdims=True)
    return (centred / np.sqrt(var + eps)).astype(np.float32)


@dataclass
class Bhanox:
    """A Bhanox model: front-end plus a stack of memory+sparse-FFN layers.

    Constructing a ``Bhanox`` gives an *untrained* model with a randomly
    initialised state. This is the reference implementation and the source of
    truth for the math (law C9); ``bhanox.from_pretrained`` will load trained
    weights once a model zoo exists (M3).

    Attributes:
        config: The validated model config.
        embedder: HashBind front-end.
        deltabanks: One DeltaBank memory per layer.
        mixers: One MicroExpert mixer per layer.
        gates: One PulseGate per layer, sized to the residual width it gates.
        vault: Optional VectorVault, present only when ``config.use_vault``.
        output: Unembedding projection, ``(d_model, vocab_table)``.
        n_forward: Lifetime count of forward calls.
    """

    config: BhanoxConfig
    embedder: HashBind = field(init=False)
    deltabanks: list[DeltaBankLayer] = field(init=False)
    mixers: list[MicroExpertLayer] = field(init=False)
    gates: list[PulseGate] = field(init=False)
    vault: VectorVault | None = field(default=None, init=False)
    output: NDArray[np.floating] = field(init=False)
    n_forward: int = 0

    def __post_init__(self) -> None:
        """Build every component for the configured shape."""
        cfg = self.config
        self.embedder = HashBind(
            d_model=cfg.d_model,
            pool_size=cfg.pool_size,
            vocab_table=cfg.vocab_table,
            n_hashes=cfg.n_hashes,
        )
        self.deltabanks = [DeltaBankLayer(cfg) for _ in range(cfg.n_layers)]
        self.mixers = [MicroExpertLayer(cfg) for _ in range(cfg.n_layers)]
        self.gates = [PulseGate(cfg.d_model) for _ in range(cfg.n_layers)]
        self.output = _init_output(cfg)
        if cfg.use_vault:
            self.vault = VectorVault(d_value=cfg.d_model)

    # -- parameters ----------------------------------------------------------

    def param_count(self) -> int:
        """Total stored parameter values, excluding the recurrent state.

        The state is a fixed buffer, not a parameter: it is re-created on reset
        and is not checkpointed, which is what makes a checkpoint tiny.
        """
        total = self.embedder.param_count() + int(self.output.size)
        total += sum(b.param_count() for b in self.deltabanks)
        total += sum(m.param_count() for m in self.mixers)
        return int(total)

    def state_nbytes(self) -> int:
        """int8 bytes of recurrent state. Constant in context length."""
        return int(sum(b.state_nbytes for b in self.deltabanks))

    def packed_nbytes(self) -> int:
        """int8 bytes of parameters, i.e. the checkpoint size."""
        return self.param_count()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recurrent state and flush the gates. O(1)."""
        for bank, gate in zip(self.deltabanks, self.gates, strict=True):
            bank.reset()
            gate.flush()

    # -- forward -------------------------------------------------------------

    def embed(self, ids: NDArray[np.integer]) -> NDArray[np.float32]:
        """HashBind front-end.

        Args:
            ids: Integer ids, any shape.

        Returns:
            ``float32`` embeddings of shape ``ids.shape + (d_model,)``.
        """
        return self.embedder.embed(np.asarray(ids))

    def forward(
        self, ids: NDArray[np.integer], *, train: bool = False
    ) -> NDArray[np.float32]:
        """Full forward pass, teacher-forced.

        Args:
            ids: ``(B, T)`` integer ids. A 1-D input is treated as one sequence.
            train: Enable the router's load-balancing update.

        Returns:
            ``(B, T, output_vocab)`` logits.

        Raises:
            ValueError: If the context is longer than ``config.max_context``.

        Warning:
            The recurrent state is a *single* instance, so ``B`` rows are not
            independent. Row 0 is processed, then row 1 against the state row 0
            left behind, and so on: ``forward(stack([a, b]))`` is identical to
            ``forward(concatenate([a, b]))``. Batching therefore buys nothing for
            the memory path -- it is the mixer that vectorises. Per-sample state
            is an M2 concern; do not assume row independence in a loss.
        """
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"forward expects (B, T) ids, got shape {arr.shape}")
        if arr.shape[-1] > self.config.max_context:
            raise ValueError(
                f"context {arr.shape[-1]} exceeds max_context="
                f"{self.config.max_context}. The state is O(1) in length, so "
                "this is a training-window limit, not a runtime one."
            )
        x = self.embed(arr).astype(np.float32)
        if train:
            x = x.copy()
        for bank, mixer, gate in zip(
            self.deltabanks, self.mixers, self.gates, strict=True
        ):
            x = x + self._run_memory(bank, gate, x)
            x = x + mixer.forward(x, train=train)
            x = layer_norm(x)
        self.n_forward += 1
        return (x @ self.output).astype(np.float32)

    def _run_memory(
        self, bank: DeltaBankLayer, gate: PulseGate, x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """DeltaBank pass with the PulseGate deciding which heads to recompute.

        Args:
            bank: This layer's memory.
            gate: This layer's governor.
            x: ``(B, T, d_model)`` activations.

        Returns:
            ``(B, T, d_model)`` memory contribution.

        Why the batch loop is outside the time loop: the state and the gate are
        single instances, so the only ordering that is well defined is row 0
        fully, then row 1, and so on. See the warning on :meth:`forward`.
        """
        batch, seq, _ = x.shape
        out = np.zeros((batch, seq, gate.n_channels), dtype=np.float32)
        for b in range(batch):
            for t in range(seq):
                step_out = bank.forward(x[b, t])
                compute = gate.step(step_out, np.abs(step_out))
                out[b, t] = np.where(compute, step_out, 0.0)
        return out

    def step(self, ids: NDArray[np.integer] | np.integer) -> NDArray[np.float32]:
        """Advance the state by one token and return next-token logits.

        Args:
            ids: A single token id, or a shape-1 array holding one.

        Returns:
            ``(vocab_table,)`` logits.

        Raises:
            ValueError: If more than one token id is supplied.
        """
        arr = np.asarray(ids).reshape(-1)
        if arr.size != 1:
            raise ValueError(
                f"step takes exactly one token id, got {arr.size}. Use forward() "
                "for a whole sequence."
            )
        x = self.embed(arr)[0]
        for bank, mixer, gate in zip(
            self.deltabanks, self.mixers, self.gates, strict=True
        ):
            x = x + self._step_memory(bank, gate, x)
            x = x + mixer.forward(x)
            x = layer_norm(x)
        self.n_forward += 1
        return (x @ self.output).astype(np.float32)

    def _step_memory(
        self, bank: DeltaBankLayer, gate: PulseGate, x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Single-token memory pass with the gate applied."""
        step_out = bank.forward(x)
        compute = gate.step(step_out, np.abs(step_out))
        return np.where(compute, step_out, 0.0)

    # -- convenience ---------------------------------------------------------

    def generate(
        self,
        ids: NDArray[np.integer],
        *,
        max_new: int = 64,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> NDArray[np.int64]:
        """Generate tokens autoregressively. See :mod:`bhanox.generate`.

        Args:
            ids: Prompt ids.
            max_new: How many tokens to produce.
            temperature: Sampling temperature. 0 or below means greedy.
            seed: RNG seed for reproducible sampling.

        Returns:
            ``int64`` array of prompt + generated ids.
        """
        from bhanox.generate import generate  # local: avoids an import cycle

        return generate(self, ids, max_new=max_new, temperature=temperature, seed=seed)
