"""Bhanox -- a CPU-native neural architecture substrate.

Purpose: expose the public API. Import this module and you have a working
Bhanox model on a CPU with no GPU, no data center, and no runtime dependency
beyond numpy.

In simple words: this is the front door. Everything you need is here.

Quickstart::

    import bhanox

    model = bhanox.Bhanox(bhanox.load_config("nano"))
    out = model.generate(b"hello", max_new=32, temperature=0.8)
    print(bhanox.report(model))

Honesty note (project law A3): every number this package reports is either
``measured`` (counted or timed by code in this repo) or ``modeled`` (a component
model). Nothing is estimated silently. See ``docs/benchmarks.md``.
"""

from __future__ import annotations

from pathlib import Path

from bhanox import audit as audit_module
from bhanox import checkpoint, frontend
from bhanox.audit import AuditReport
from bhanox.config import (
    PRESETS,
    BhanoxConfig,
    available_presets,
    load_config,
    register_preset,
)
from bhanox.frontend.hashbind import HashBind, encode_bytes
from bhanox.generate import generate, generate_ids
from bhanox.model import Bhanox, MemoryReport, PermSection, TempSection

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "Bhanox",
    "BhanoxConfig",
    "HashBind",
    "__version__",
    "audit_bytes_per_token",
    "available_presets",
    "checkpoint",
    "encode_bytes",
    "from_pretrained",
    "frontend",
    "generate",
    "generate_ids",
    "load_config",
    "memory_report",
    "register_preset",
    "report",
]


def audit_bytes_per_token(model: Bhanox) -> AuditReport:
    """Audit bytes touched per token against invariant I2.

    Args:
        model: The model to audit.

    Returns:
        A :class:`bhanox.audit.AuditReport`.
    """
    return audit_module.audit_bytes_per_token(model)


def report(model: Bhanox) -> str:
    """Human-readable cost summary for a model.

    Args:
        model: The model to describe.

    Returns:
        A multi-line string with parameter counts, packed size, state size,
        the I2 residency verdict and the measured I3 op count. Every figure is
        counted from real arrays at call time, hence ``measured``.
    """
    cfg = model.config
    residency = audit_bytes_per_token(model)
    params = model.param_count()
    lines = [
        f"Bhanox report -- config={cfg.name} (all figures measured)",
        f"  parameters            {params:>12,} values",
        f"  packed (int8)         {params / 1e6:>11.2f} M  "
        f"({model.packed_nbytes() / 1024:.1f} KiB)",
        f"  recurrent state       {model.state_nbytes():>12,} B "
        f"(constant in context)",
        f"  L2 residency          {residency.passed!s:>12}  "
        f"({residency.utilization:.1%} of {cfg.l2_bytes:,} B)",
        f"  I3 op whitelist       {len(residency.op_check):>12} ops, no float",
        "",
        residency.summary(),
    ]
    return "\n".join(lines)


def memory_report(model: Bhanox, *, as_text: bool = True) -> str | MemoryReport:
    """Print (or return) each memory's bytes against its budget.

    Spec D7. This is the user-facing view of the two byte budgets: the
    DeltaBank's temporary working state and the VectorVault's permanent store.

    Args:
        model: The model to describe.
        as_text: Print a table and return it. Set False to get the typed report,
            which is what the tests assert on.

    Returns:
        The formatted report, or a :class:`bhanox.model.MemoryReport` when
        ``as_text`` is False.
    """
    data = model.memory_report()
    if not as_text:
        return data
    lines = [
        f"Memory report -- config={data['config']} (spec D7 byte budgets)",
        f"  {'memory':<6} {'in use':>12} {'budget':>12}  {'within':<7} note",
    ]

    def row(label: str, section: TempSection | PermSection, note: str) -> str:
        budget = section["budget_human"]
        return (
            f"  {label:<6} {section['used_human']:>12} "
            f"{'uncapped' if budget is None else budget:>12}  "
            f"{section['within_budget']!s:<7} {note}"
        )

    lines.append(row("temp", data["temp"], "fixed by the trained weights"))
    perm = data["perm"]
    if perm is None:
        lines.append("  perm   no vault (use_vault=False)")
    else:
        note = (
            f"{perm['entries']}/{perm['entry_limit']} entries, "
            f"{perm['reserved_bytes']:,} B reserved"
        )
        lines.append(row("perm", perm, note))
    for warning in data["warnings"]:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def from_pretrained(name_or_path: str) -> Bhanox:
    """Load a trained model from a checkpoint.

    Args:
        name_or_path: A path to a ``.npz`` checkpoint written by
            :func:`bhanox.checkpoint.save`.

    Returns:
        A ready-to-use :class:`Bhanox`, built from the config recorded in the
        checkpoint and restored bit-exactly.

    Raises:
        OSError: The file is not a Bhanox checkpoint.
        ValueError: The version is unknown, or a tensor does not fit the
            recorded architecture.

    Note:
        The recurrent state is deliberately not in the file, so the returned
        model starts with an empty state. That is what keeps a checkpoint
        proportional to parameters rather than to context length. A trainer that
        needs a bit-exact mid-sequence resume carries the state itself; see
        :mod:`bhanox.checkpoint`.
    """
    return checkpoint.load(Path(name_or_path))
