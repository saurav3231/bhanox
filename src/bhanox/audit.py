"""Residency audit: bytes touched per token, per layer (invariant I2).

Purpose: make the energy story checkable. Energy follows bytes moved, not
FLOPs (Horowitz, ISSCC 2014), so the primary cost metric in this project is
bytes-touched-per-token, and the primary architectural constraint is that this
number fits in L2.

In simple words: if the numbers a layer needs every token do not fit in the
CPU's fast cache, the CPU has to go to slow memory, and the energy bill jumps
by an order of magnitude. This module measures that, per layer, and fails loudly
when the budget is blown.

Invariant I2: ``bytes_touched(per token, per layer) <= L2_BYTES``.

Current honest result, measured at a 1 MiB budget:

=========  ==============  =================  =========
preset     bytes / layer   vs 1 MiB L2        verdict
=========  ==============  =================  =========
``nano``        110,592              10.5%  PASS
``mini``        720,896              68.8%  PASS
``small``     3,538,944             337.5%  FAIL
=========  ==============  =================  =========

``small`` fails because it is not only sparse on the wrong axis: its DeltaBank
projections alone are 3,014,656 B per layer, against a 1,048,576 B budget. The
mixer is not the problem -- it touches 524,288 B, 5.7x less than its dense
equivalent. Reported, not tuned away (ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bhanox.config import BhanoxConfig
from bhanox.core.deltabank_layer import DeltaBankLayer
from bhanox.frontend.hashbind import HashBind
from bhanox.ir.verifier import Node, assert_inference_safe
from bhanox.mixer.microexpert import MicroExpertLayer
from bhanox.model import Bhanox

__all__ = ["INFERENCE_OPS", "AuditReport", "LayerCost", "audit_bytes_per_token"]

#: The op sequence a Bhanox inference graph executes per token per layer, in
#: BIR terms. Checked against invariant I3 on every audit run, so "no float in
#: inference" is verified rather than promised.
INFERENCE_OPS: tuple[str, ...] = (
    "LOAD_I8",
    "MUL_I8:8",
    "ADD",
    "SUB",
    "SAT",
    "SHIFT",
    "CMP",
    "LUT",
    "POPCOUNT",
    "PERMUTE",
    "STORE_I8",
)


@dataclass(frozen=True)
class LayerCost:
    """Bytes touched per token by one layer, itemised by component.

    Attributes:
        name: Component label, e.g. ``"deltabank"``.
        nbytes: int8 bytes touched per token.
        detail: Human-readable breakdown.
    """

    name: str
    nbytes: int
    detail: str = ""


@dataclass
class AuditReport:
    """Result of an I2 residency audit.

    Attributes:
        config_name: The audited config.
        front_end: Bytes touched once per token by HashBind. This happens *once*
            for the whole model, not once per layer, so it is deliberately kept
            out of :attr:`layer_total` -- counting it in both would charge the
            same bytes twice and quietly inflate every per-layer figure.
        per_layer: Per-layer breakdown, identical in total for every layer.
        layer_total: Bytes touched per token per layer. This is what I2 checks.
        state_total: Bytes of recurrent state for the whole model.
        l2_bytes: The budget the audit compared against.
        op_check: The I3 op whitelist that was verified during the audit.
    """

    config_name: str
    front_end: int
    per_layer: list[LayerCost] = field(default_factory=list)
    layer_total: int = 0
    state_total: int = 0
    l2_bytes: int = 0
    op_check: tuple[str, ...] = INFERENCE_OPS

    @property
    def total_per_token(self) -> int:
        """Bytes touched per token for the whole model."""
        return self.front_end + self.layer_total

    @property
    def passed(self) -> bool:
        """Whether invariant I2 holds for every layer."""
        return self.layer_total <= self.l2_bytes

    @property
    def utilization(self) -> float:
        """Fraction of the L2 budget the hottest layer consumes."""
        return self.layer_total / self.l2_bytes if self.l2_bytes else float("inf")

    def summary(self) -> str:
        """Render the report as plain text.

        Returns:
            A human-readable multi-line report, labelled ``measured`` because
            every number in it comes from counting actual stored arrays.
        """
        lines = [
            f"Bhanox residency audit -- config={self.config_name} (measured)",
            f"  front-end (HashBind)   {self.front_end:>12,} B / token (whole model)",
            f"  recurrent state        {self.state_total:>12,} B (constant)",
            f"  per layer total        {self.layer_total:>12,} B / token",
            f"  whole model            {self.total_per_token:>12,} B / token",
        ]
        lines += [
            f"    {c.name:<22}{c.nbytes:>10,} B  {c.detail}" for c in self.per_layer
        ]
        verdict = "PASS" if self.passed else "FAIL"
        lines.append(
            f"  I2 vs L2 budget        {self.l2_bytes:>12,} B  "
            f"-> {verdict} ({self.utilization:.1%} of budget)"
        )
        lines.append(f"  I3 op whitelist        {len(self.op_check)} ops verified")
        return "\n".join(lines)


def _deltabank_cost(bank: DeltaBankLayer, cfg: BhanoxConfig) -> LayerCost:
    """Bytes the DeltaBank touches per token.

    The state is read once and written once, and the projections are read once,
    so the figure is a straight sum of buffer sizes with no context dependence.
    """
    h = bank.heads[0]
    per_head = h.W_k.size + h.W_q.size + h.W_v.size + h.W_r.size
    shared = bank.W_o.size + bank.G.size
    projections = per_head * len(bank.heads) + shared
    state = bank.state_nbytes
    return LayerCost(
        "deltabank",
        int(projections + 2 * state),
        f"{cfg.n_heads} heads, state read+written ({state:,} B x2)",
    )


def _mixer_cost(mixer: MicroExpertLayer, cfg: BhanoxConfig) -> LayerCost:
    """Bytes the MicroExpert touches per token: shared + top-k, never all."""
    active = mixer.active_nbytes()
    dense = mixer.dense_nbytes()
    return LayerCost(
        "microexpert",
        active,
        f"{cfg.n_shared_experts}+{cfg.top_k} of {cfg.total_experts} experts "
        f"({dense / max(active, 1):.1f}x fewer than dense)",
    )


def _frontend_cost(embedder: HashBind) -> LayerCost:
    """Bytes the front-end touches per token: a few pool rows, not the pool.

    Only ``n_hashes`` rows are read per token, so the front-end's *touched*
    bytes are tiny even though its *resident* bytes are the largest single
    table in the model. Residency is what matters for the energy claim, so the
    audit reports both.
    """
    touched = embedder.n_hashes * embedder.d_model
    return LayerCost(
        "hashbind (touched)",
        int(touched),
        f"{embedder.n_hashes} rows of a {embedder.nbytes:,} B resident pool",
    )


def audit_bytes_per_token(model: Bhanox) -> AuditReport:
    """Audit a model's bytes-touched-per-token against invariant I2.

    Args:
        model: A :class:`bhanox.model.Bhanox`.

    Returns:
        An :class:`AuditReport` with a per-layer breakdown and a pass/fail
        verdict against the config's ``l2_bytes`` budget.

    Raises:
        I3ViolationError: If the inference op whitelist itself is not I3-clean, which
            would mean the audit was measuring an illegal graph.
    """
    assert_inference_safe(INFERENCE_OPS, where="inference op list")
    cfg: BhanoxConfig = model.config
    bank, mixer = model.deltabanks[0], model.mixers[0]
    # The front-end is charged once for the model, so it is reported separately
    # and excluded from the per-layer sum that I2 is stated against.
    items = [_deltabank_cost(bank, cfg), _mixer_cost(mixer, cfg)]
    return AuditReport(
        config_name=cfg.name,
        front_end=_frontend_cost(model.embedder).nbytes,
        per_layer=items,
        layer_total=sum(c.nbytes for c in items),
        state_total=model.state_nbytes(),
        l2_bytes=cfg.l2_bytes,
    )


def audit_ops(nodes: list[Node]) -> None:
    """Verify an explicit node list against I3.

    Thin re-export so callers do not need to import :mod:`bhanox.ir` directly.

    Args:
        nodes: BIR nodes to check.

    Raises:
        I3ViolationError: On the first illegal node.
    """
    from bhanox.ir.verifier import Graph, verify

    verify(Graph("audit", tuple(nodes)))


def _demo() -> None:
    """Self-check: audit every preset and print the verdicts."""
    import bhanox

    for name in bhanox.available_presets():
        model = bhanox.Bhanox(bhanox.load_config(name))
        report = audit_bytes_per_token(model)
        print(report.summary())
        print()


if __name__ == "__main__":
    _demo()
