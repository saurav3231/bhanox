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
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from bhanox.budgets import format_bytes
from bhanox.config import BhanoxConfig
from bhanox.core.deltabank_layer import DeltaBankLayer
from bhanox.frontend.hashbind import HashBind
from bhanox.governor.pulsegate import PulseGate
from bhanox.memory.vectorvault import VectorVault
from bhanox.mixer.microexpert import MicroExpertLayer
from bhanox.quant.numerics import absmax_quantize
from bhanox.seeding import init_rng

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
    rng = init_rng(cfg.seed, "unembed", cfg.name)
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


class TempSection(TypedDict):
    """The DeltaBank working state against its budget."""

    used_bytes: int
    budget_bytes: int | None
    used_human: str
    budget_human: str | None
    within_budget: bool
    adjustable: bool


class PermSection(TypedDict):
    """The VectorVault against its budget."""

    used_bytes: int
    budget_bytes: int | None
    used_human: str
    budget_human: str | None
    within_budget: bool
    entries: int
    entry_limit: int
    reserved_bytes: int


class MemoryReport(TypedDict):
    """Both memories' budgets, as `bhanox.memory_report` returns them."""

    config: str
    temp: TempSection
    perm: PermSection | None
    warnings: list[str]


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
        # The index matters: keyed on shape alone, every layer drew the same
        # stream and the stack was n_layers copies of one block. See
        # bhanox.seeding.
        self.mixers = [
            MicroExpertLayer(cfg, layer_index=i) for i in range(cfg.n_layers)
        ]
        self.gates = [PulseGate(cfg.d_model) for _ in range(cfg.n_layers)]
        self.output = _init_output(cfg)
        if cfg.use_vault:
            self.vault = VectorVault(d_value=cfg.d_model)
            # A budget passed to load_config has to reach the vault here, or the
            # config would accept "2GB", turn the vault on, and then run it
            # uncapped -- which is the silent-discard failure D7 forbids.
            if cfg.perm_mem_bytes is not None:
                self.vault.set_budget(cfg.perm_mem_bytes)

    # -- parameters ----------------------------------------------------------

    def param_count(self) -> int:
        """Total learned parameter values, excluding the recurrent state.

        The state is a fixed buffer, not a parameter: it is re-created on reset
        and is not checkpointed, which is what makes a checkpoint tiny.

        The gates count. Their per-channel wake/sleep thresholds are learned and
        are trained like any other weight, so a size claim that excluded them
        would under-report what the optimizer actually updates -- and, during
        training, silently omit 1,536 values from the bill. ``test_checkpoint``
        pins this total against the checkpoint's own tensor walk so a future
        parameter cannot escape both.

        Derived-from-config arrays are excluded: ``DeltaBankLayer.bank_rates``
        is a cached broadcast of ``cfg.decay_rates``, not something learned, and
        counting it would make the number depend on how a constant is cached.
        """
        total = self.embedder.param_count() + int(self.output.size)
        total += sum(b.param_count() for b in self.deltabanks)
        total += sum(m.param_count() for m in self.mixers)
        total += sum(g.param_count() for g in self.gates)
        return int(total)

    def state_nbytes(self) -> int:
        """int8 bytes of recurrent state. Constant in context length."""
        return int(sum(b.state_nbytes for b in self.deltabanks))

    def packed_nbytes(self) -> int:
        """int8 bytes of parameters, i.e. the checkpoint size."""
        return self.param_count()

    # -- memory budgets (spec D7) ---------------------------------------------

    def set_perm_budget(self, budget: str | int | None) -> None:
        """Resize the permanent store's byte budget, live.

        The one Bhanox memory a caller can resize while the model runs. Lowering
        it below what is already stored truncates the vault by importance
        score; it never raises, because running out of room must cost recall
        rather than the process. Raising it lets the vault admit again.

        Args:
            budget: ``"4GB"``, an int, or ``None`` to lift the cap.

        Raises:
            ValueError: If the model has no vault, or the budget cannot be
                parsed or is too small for one entry.
        """
        if self.vault is None:
            raise ValueError(
                f"{self.config.name}: no VectorVault, so there is no permanent "
                "store to budget. Build the model with use_vault=True."
            )
        self.vault.set_budget(budget)
        self.config = self.config.with_memory_budgets(perm_mem=budget)

    def memory_report(self) -> MemoryReport:
        """Bytes held by each memory against its budget, plus the verdict.

        Returns:
            A :class:`MemoryReport`. ``perm`` is ``None`` when the model has no
            vault. ``warnings`` lists any budget that is exceeded.

        Why a typed structure and not a string: a report you cannot assert on is
        decoration. ``bhanox.memory_report`` prints it; tests read the same
        object.
        """
        warnings: list[str] = []
        required = self.config.required_temp_bytes()
        temp_budget = self.config.temp_mem_bytes
        temp_ok = temp_budget is None or required <= temp_budget
        if temp_budget is not None and not temp_ok:
            warnings.append(
                f"temp memory needs {format_bytes(required)} but the budget is "
                f"{format_bytes(temp_budget)}. This is not adjustable: the state "
                "is fixed by the trained weights. Raise temp_mem or train a "
                "narrower model."
            )
        temp: TempSection = {
            "used_bytes": required,
            "budget_bytes": temp_budget,
            "used_human": format_bytes(required),
            "budget_human": (
                None if temp_budget is None else format_bytes(temp_budget)
            ),
            "within_budget": temp_ok,
            "adjustable": False,
        }
        perm: PermSection | None = None
        if self.vault is not None:
            used = self.vault.used_bytes
            budget = self.vault.perm_budget_bytes
            perm_ok = budget is None or used <= budget
            if budget is not None and not perm_ok:
                warnings.append(
                    f"perm memory holds {format_bytes(used)} against a budget of "
                    f"{format_bytes(budget)}."
                )
            perm = {
                "used_bytes": used,
                "budget_bytes": budget,
                "used_human": format_bytes(used),
                "budget_human": None if budget is None else format_bytes(budget),
                "within_budget": perm_ok,
                "entries": self.vault.filled,
                "entry_limit": self.vault.max_admissible_entries(),
                "reserved_bytes": self.vault.nbytes,
            }
        return {
            "config": self.config.name,
            "temp": temp,
            "perm": perm,
            "warnings": warnings,
        }

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

        Note:
            Rows are independent. The DeltaBank state and the PulseGate state
            both carry a sample axis, so sample ``b`` reads only what sample
            ``b`` wrote.

            Two different claims, and the difference matters. The recurrent
            *state* is int32, so ``forward(stack([a, b]))[1]`` and
            ``forward(b)[0]`` leave states that compare equal with ``==``. The
            *logits* are float32 and come out of a matmul over a ``(B, T)``
            block, so they are equal only to rounding -- BLAS sums a batched
            product in a different order than a single-row one. Compare
            logits with ``allclose``; compare state with ``array_equal``.

            This was not true before M2. The state was a single instance, so
            ``forward(stack([a, b]))`` equalled
            ``forward(concatenate([a, b]))`` and a batched loss measured the
            wrong thing. Batch-1 behaviour is unchanged by the refactor --
            measured, not assumed: same int32 state, same gate decisions, and
            float32 logits bit-identical (``max|diff| == 0.0``) on the nano
            config. See ``docs/architecture.md``.
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

        Why the time loop is outside the batch loop's shadow: the state is
        ``(B, d_k, d_v)``, one row per sample, so sample ``b`` only ever reads
        what sample ``b`` wrote. The time loop stays outermost because the
        recurrence is sequential in ``t`` -- there is nothing to vectorise
        across it -- but the batch is now real rather than a throughput knob
        for the mixer alone.
        """
        batch, seq, _ = x.shape
        out = np.zeros((batch, seq, gate.n_channels), dtype=np.float32)
        for t in range(seq):
            step_out = bank.forward(x[:, t])
            compute = gate.step(step_out, np.abs(step_out))
            out[:, t] = np.where(compute, step_out, 0.0)
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
        prompt: bytes,
        *,
        max_new: int = 64,
        temperature: float = 0.8,
        seed: int | None = None,
        reset: bool = True,
    ) -> bytes:
        """Generate bytes after a byte prompt. See :mod:`bhanox.generate`.

        Args:
            prompt: The bytes to continue, at least 4 of them. ``max_new``,
                ``temperature``, ``seed`` and ``reset`` mean what the module
                says they mean.

        Returns:
            ``bytes``: the prompt plus the generated bytes, which are not
            necessarily valid UTF-8.
        """
        from bhanox.generate import generate  # local: avoids an import cycle

        return generate(
            self,
            prompt,
            max_new=max_new,
            temperature=temperature,
            seed=seed,
            reset=reset,
        )

    def generate_ids(
        self,
        ids: NDArray[np.integer],
        *,
        max_new: int = 64,
        temperature: float = 0.8,
        seed: int | None = None,
        reset: bool = True,
    ) -> NDArray[np.uint8]:
        """Generate from 4-gram ids. See :func:`bhanox.generate.generate_ids`.

        Args:
            ids: Prompt context ids, which must be byte 4-grams. The other
                arguments mean what the module says they mean.

        Returns:
            ``uint8`` array of the generated byte values only.
        """
        from bhanox.generate import generate_ids  # local: avoids an import cycle

        return generate_ids(
            self,
            ids,
            max_new=max_new,
            temperature=temperature,
            seed=seed,
            reset=reset,
        )
