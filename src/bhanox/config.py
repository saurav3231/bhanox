"""Configuration: model shapes, the three frozen presets, and invariant I1.

Purpose: one place that knows what "Nano", "Mini" and "Small" mean, and one
place that refuses a config whose recurrence cannot hold its own writes
(head-load, invariant I1).

In simple words: this is the blueprint, plus the check that the blueprint is
strong enough to stand on.

Reference: architecture spec D2 (I1), D4 (presets).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from bhanox.budgets import parse_bytes, temp_state_bytes

__all__ = [
    "L2_BYTES",
    "PRESETS",
    "BhanoxConfig",
    "available_presets",
    "concurrent_writes",
    "load_config",
    "register_preset",
]

#: Conservative L2 cache size assumed by the I2 residency audit (KiB).
#: Override per machine via ``BhanoxConfig(l2_bytes=...)`` -- real silicon
#: ranges from 256 KiB (older Atom) to 2 MiB (Zen 4 / E-cores).
L2_BYTES = 1024 * 1024


@dataclass(frozen=True)
class BhanoxConfig:
    """Frozen model shape. All fields are integers; no runtime toggles.

    Attributes:
        name: Preset identifier, e.g. ``"nano"``.
        d_model: Model width. All projections are d_model-wide.
        n_layers: Number of DeltaBank + MicroExpert blocks.
        n_heads: Independent DeltaBank memories per layer.
        d_k: Key/query width of one head. Also the state's row count.
        d_v: Value width of one head. Also the state's column count.
        n_banks: Number of decay-rate banks B; the state prior is
            ``{1 - 2**-b : b in 1..B}``.
        n_experts: Routed (fine-grained) experts per layer.
        n_shared_experts: Always-on shared experts per layer.
        d_expert: Hidden width inside one expert MLP.
        top_k: Routed experts activated per token.
        use_vault: Enable the VectorVault episodic store.
        ternary: Opt-in ternary {-1,0,+1} weight regime. Mandated OFF below
            Small scale -- full-ternary collapses at small scale (design
            phase: BPC 22.3 at 100 steps vs 3.756 for int8-hybrid).
        vocab_table: Rows in the HashBind learned table (ids < this get a
            dedicated row; everything else is hashed). This is *not* the output
            vocabulary -- see ``output_vocab``.
        output_vocab: Size of the unembedding. The spec's front-end is a byte
            4-gram, so the prediction target is the next *byte*: 256 classes.
        pool_size: Rows in the shared HashBind hash pool.
        n_hashes: Independent hash functions per token.
        max_context: Positional window for the recurrent state. The state does
            not grow with it (O(1) generation, spec A2/goal 3).
        seed: Seed for weight initialisation. Every projection derives its own
            stream from ``(seed, site, name)``, so the streams cannot collide
            and the whole model is reproducible across processes. This is
            recorded in the checkpoint header. It used to be
            ``abs(hash((site, name)))``, and ``hash()`` on a str is randomised
            per process, so the read-out, bypass, unembed and MoE weights were
            silently different on every run.
        l2_bytes: L2 budget used by the I2 audit. Not a model parameter.
        temp_mem_bytes: Spec-D7 budget for the DeltaBank working state. ``None``
            means "whatever the shape needs". Not user-resizable: the state is
            fixed by the trained weights.
        perm_mem_bytes: Spec-D7 budget for the VectorVault. ``None`` means
            uncapped, which is the pre-D7 behaviour. Settable live via
            ``Bhanox.set_perm_budget``.
    """

    name: str = "custom"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_k: int = 16
    d_v: int = 32
    n_banks: int = 8
    n_experts: int = 16
    n_shared_experts: int = 1
    d_expert: int = 32
    top_k: int = 2
    use_vault: bool = False
    ternary: bool = False
    vocab_table: int = 1024
    output_vocab: int = 256
    pool_size: int = 8192
    n_hashes: int = 4
    max_context: int = 4096
    seed: int = 0
    l2_bytes: int = L2_BYTES
    temp_mem_bytes: int | None = None
    perm_mem_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate shape and invariant I1.

        Raises:
            ValueError: On a non-positive field, an inconsistent shape, a
                top_k larger than the expert pool, or an I1 violation.
        """
        positive = {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "n_banks": self.n_banks,
            "d_expert": self.d_expert,
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "n_hashes": self.n_hashes,
            "l2_bytes": self.l2_bytes,
            "vocab_table": self.vocab_table,
            "output_vocab": self.output_vocab,
        }
        for field_name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{self.name}: {field_name} must be > 0, got {value}")
        if self.n_experts < 0 or self.n_shared_experts < 0:
            raise ValueError(f"{self.name}: expert counts must be >= 0")
        if self.n_experts == 0 and self.n_shared_experts == 0:
            raise ValueError(f"{self.name}: need at least one expert")
        if self.top_k > self.n_experts:
            raise ValueError(
                f"{self.name}: top_k={self.top_k} exceeds n_experts={self.n_experts}"
            )
        if self.n_hashes > self.pool_size:
            raise ValueError(
                f"{self.name}: n_hashes={self.n_hashes} exceeds "
                f"pool_size={self.pool_size}"
            )
        if self.vocab_table > self.pool_size:
            raise ValueError(
                f"{self.name}: vocab_table={self.vocab_table} exceeds "
                f"pool_size={self.pool_size}"
            )
        # Ternary is an opt-in regime for >=1B-scale long schedules only.
        # Design phase: full-ternary at small scale collapses during training.
        if self.ternary and self.name in ("nano", "mini"):
            raise ValueError(
                f"{self.name}: ternary is opt-in and NOT permitted below Small "
                "scale (design phase: full-ternary collapses at small scale)"
            )
        self.check_head_load()

    def with_memory_budgets(
        self,
        *,
        temp_mem: str | int | None = None,
        perm_mem: str | int | None = None,
    ) -> BhanoxConfig:
        """Return a copy with the spec-D7 byte budgets applied.

        Frozen dataclass, so this returns a new config rather than mutating.
        Passing ``None`` for both leaves the config untouched, which is why
        ``load_config("nano")`` behaves exactly as it did before budgets existed.

        Args:
            temp_mem: Temporary-state budget, e.g. ``"64MB"``.
            perm_mem: Permanent-store budget, e.g. ``"2GB"``. Implies
                ``use_vault=True``, since a budget for a memory that does not
                exist would otherwise be silently ignored.

        Returns:
            A config carrying the budgets.

        Raises:
            ValueError: If a budget cannot be parsed.
        """
        if temp_mem is None and perm_mem is None:
            return self
        # Asking for a permanent-memory budget implies wanting a permanent
        # memory. Without this, `load_config("mini", perm_mem="2GB")` would
        # accept the budget and then silently ignore it, because every preset
        # ships with the vault off. A budget that is quietly discarded is worse
        # than one that is refused.
        #
        # Only the arguments actually passed are replaced. Passing perm_mem
        # alone must not clear a temp_mem set earlier -- `set_perm_budget` goes
        # through here, and clearing the other budget on every live resize is
        # how a 64MB temp budget silently becomes "uncapped".
        return replace(
            self,
            use_vault=(
                True if perm_mem is not None and not self.use_vault else self.use_vault
            ),
            temp_mem_bytes=(
                self.temp_mem_bytes if temp_mem is None else parse_bytes(temp_mem)
            ),
            perm_mem_bytes=(
                self.perm_mem_bytes if perm_mem is None else parse_bytes(perm_mem)
            ),
        )

    def required_temp_bytes(self) -> int:
        """Temporary-state bytes this shape needs, int8, one byte per cell.

        Fixed by the trained weights, so unlike the permanent store it is not a
        knob: narrowing ``d_k``, ``d_v`` or ``n_heads`` would change the model.
        See :func:`bhanox.budgets.temp_state_bytes` for why ``n_banks`` is absent.
        """
        return temp_state_bytes(self.n_layers, self.n_heads, self.d_k, self.d_v)

    def check_head_load(self) -> None:
        """Enforce invariant I1 (head-load) for this config.

        I1: ``concurrent_writes < d_k`` where
        ``concurrent_writes = write_rate * 1/(1 - lambda_max)``.

        Why: if more associations are in flight than the state has key rows,
        the delta rule has nowhere to put them and later writes evict the
        associations being retrieved. This is a hardware capacity limit, not a
        tuning knob.

        Raises:
            ValueError: If the shallowest decay bank cannot absorb one write
                per layer per token.
        """
        load = concurrent_writes(self.n_layers, self.n_banks)
        if load >= self.d_k:
            raise ValueError(
                f"{self.name}: invariant I1 violated -- concurrent_writes="
                f"{load:.1f} >= d_k={self.d_k}. Each layer writes once per "
                f"token and the shallowest bank (1-2^-1=0.5) halves the "
                f"effective capacity. Raise d_k to >={load + 1}, lower "
                f"n_layers to <={_max_layers(self.d_k, self.n_banks)}, or add "
                "heads."
            )

    @property
    def head_load(self) -> float:
        """Concurrent writes per head-row at steady state (I1 metric)."""
        return concurrent_writes(self.n_layers, self.n_banks)

    @property
    def total_experts(self) -> int:
        """Routed plus shared experts."""
        return self.n_experts + self.n_shared_experts

    @property
    def decay_rates(self) -> list[float]:
        """The frozen bank prior ``[1 - 2**-b for b in 1..n_banks]``."""
        return [1.0 - 2.0**-b for b in range(1, self.n_banks + 1)]

    @property
    def n_state_rows(self) -> int:
        """Total DeltaBank key rows in the model (d_k summed over layers)."""
        return self.n_layers * self.n_heads * self.d_k

    def evolve(self, **changes: Any) -> BhanoxConfig:
        """Return a validated copy with fields replaced.

        Args:
            **changes: Field overrides.

        Returns:
            A new validated :class:`BhanoxConfig`.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for checkpoints and YAML/JSON configs)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BhanoxConfig:
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _max_layers(d_k: int, n_banks: int) -> int:
    """Largest layer count satisfying I1 for a given key width.

    Why: the error message for an I1 violation should offer a number the user
    can paste, not make them solve the inequality themselves.
    """
    slack = d_k - 1
    return max(0, int(slack // concurrent_writes(1, n_banks)))


def concurrent_writes(write_rate: float, n_banks: int) -> float:
    """Expected in-flight writes per head under geometric decay (I1).

    Args:
        write_rate: Writes per token. One per layer, so ``n_layers``.
        n_banks: Number of decay banks. Present for interface stability; the
            binding constraint is the shallowest bank, which every config has.

    Returns:
        ``write_rate * 1/(1 - lambda)`` with ``lambda = 1 - 2**-1 = 0.5``.

    Why the shallowest bank and not the slowest: the delta rule *corrects*
    rather than accumulates, so a write that has decayed to nothing occupies
    no key row. Saturation comes from the fast-decay channels, where several
    recent writes are all still at full strength and competing for the same
    rows. A write of rate ``lambda`` stays materially present for about
    ``1/(1-lambda)`` steps, so at ``lambda = 0.5`` that is 2 steps and the
    in-flight count is twice the write rate.

    Reading the spec's ``lambda_max`` as the *slowest* bank (``1 - 2**-B``)
    instead makes I1 unsatisfiable for every config in the frozen preset table
    (nano would need ``d_k >= 1025``), so the shallowest-bank reading is the one
    the preset table was built against. Recorded in ADR-005.
    """
    del n_banks  # see docstring: the shallowest bank binds, not n_banks
    lambda_shallow = 1.0 - 2.0**-1
    return write_rate / (1.0 - lambda_shallow)


# --- Frozen presets (architecture D4) ----------------------------------------

_NANO = BhanoxConfig(
    name="nano",
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_k=16,
    d_v=32,
    n_banks=8,
    n_experts=16,
    n_shared_experts=1,
    d_expert=32,
    top_k=2,
    use_vault=False,
    ternary=False,
)

_MINI = BhanoxConfig(
    name="mini",
    d_model=256,
    n_layers=8,
    n_heads=8,
    d_k=32,
    d_v=64,
    n_banks=12,
    n_experts=64,
    n_shared_experts=1,
    d_expert=64,
    top_k=2,
    use_vault=False,
    ternary=False,
)

_SMALL = BhanoxConfig(
    name="small",
    d_model=512,
    n_layers=16,
    n_heads=16,
    d_k=64,
    d_v=64,
    n_banks=16,
    n_experts=128,
    n_shared_experts=2,
    d_expert=128,
    top_k=2,
    use_vault=True,
    ternary=False,
)

PRESETS: dict[str, BhanoxConfig] = {c.name: c for c in (_NANO, _MINI, _SMALL)}


def available_presets() -> tuple[str, ...]:
    """Names of the registered presets."""
    return tuple(sorted(PRESETS))


def register_preset(config: BhanoxConfig) -> BhanoxConfig:
    """Add a validated config to the preset registry.

    Args:
        config: Config to register. Already validated at construction.

    Returns:
        The same config, for chaining.
    """
    PRESETS[config.name] = config
    return config


def load_config(
    source: str | Path | BhanoxConfig | dict[str, Any],
    *,
    temp_mem: str | int | None = None,
    perm_mem: str | int | None = None,
) -> BhanoxConfig:
    """Resolve a config from a preset name, a JSON/YAML path, dict, or object.

    In simple words: "give me the Nano shape" -- by any of the usual ways.

    Args:
        source: ``"nano"`` / ``"mini"`` / ``"small"``, a path to a JSON file, a
            mapping of fields, or an existing config.
        temp_mem: Byte budget for the DeltaBank working state, e.g. ``"64MB"``.
            Spec D7. Defaults to the preset's own requirement, so omitting it
            changes nothing.
        perm_mem: Byte budget for the VectorVault, e.g. ``"2GB"``. Spec D7.
            Lowering it below what a loaded model already holds truncates the
            vault by importance score rather than raising.

    Returns:
        A validated :class:`BhanoxConfig`.

    Raises:
        KeyError: If a preset name is unknown (message lists valid names).
        FileNotFoundError: If a path does not exist.
        ValueError: If a file is not valid JSON, a dict is not a valid config,
            or a budget cannot be parsed.
    """
    if isinstance(source, BhanoxConfig):
        config = source
    elif isinstance(source, dict):
        config = BhanoxConfig.from_dict(source)
    else:
        key = str(source)
        if key in PRESETS:
            config = PRESETS[key]
        else:
            path = Path(key)
            if path.suffix == ".json" and path.exists():
                config = BhanoxConfig.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            else:
                raise KeyError(
                    f"unknown config {key!r}. Presets: "
                    f"{', '.join(available_presets())}; or pass a path to a .json "
                    "config."
                )
    return config.with_memory_budgets(temp_mem=temp_mem, perm_mem=perm_mem)


ConfigName = Literal["nano", "mini", "small"]
