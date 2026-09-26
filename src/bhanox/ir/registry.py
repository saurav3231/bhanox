"""BIR (Bhanox Intermediate Representation) op registry.

Purpose: the single place that decides which machine operations a Bhanox
inference graph is allowed to contain. Invariant I3 (architecture D2) is
enforced against :data:`WHITELIST` -- not by convention, but by the verifier in
:mod:`bhanox.ir.verifier`, which refuses to build a graph containing anything
outside it.

In simple words: this is the list of things the CPU is allowed to be asked to
do. Anything not on the list is a bug or a design violation.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["INT_WIDTH_LIMIT", "WHITELIST", "OpSpec", "is_allowed", "op_spec"]

# I3: "any integer multiply wider than 8 bits" is a release blocker. This is
# the constant the verifier checks MUL widths against.
INT_WIDTH_LIMIT = 8


@dataclass(frozen=True)
class OpSpec:
    """Static description of one BIR operation.

    Attributes:
        name: Canonical opcode, e.g. ``"MUL_I8"``.
        cost: Relative integer op cost, used by the auditor to rank hotspots.
        is_float: True if the op performs floating-point arithmetic. I3 forbids
            these in an inference graph; they are declared so the verifier can
            reject them by name rather than by pattern matching.
        max_width: For integer ops, the widest operand permitted in bits.
        signed: Whether operands are two's-complement signed.
    """

    name: str
    cost: int
    is_float: bool
    max_width: int = INT_WIDTH_LIMIT
    signed: bool = True


def _op(
    name: str, cost: int, is_float: bool = False, max_width: int = INT_WIDTH_LIMIT
) -> OpSpec:
    return OpSpec(name, cost, is_float, max_width)


# The frozen whitelist from architecture D2, verbatim. Order is documentation.
WHITELIST: dict[str, OpSpec] = {
    spec.name: spec
    for spec in (
        _op("ADD", 1),
        _op("SUB", 1),
        _op("CMP", 1),
        _op("SHIFT", 1),
        _op("PERMUTE", 1),
        _op("POPCOUNT", 2),
        _op("LUT", 2),
        _op("MUL_I8", 2),
        _op("LOAD_I8", 1),
        _op("STORE_I8", 1),
        _op("SAT", 1),
        # Pseudo-ops: graph structure and constants, not arithmetic.
        _op("CONST", 0),
        _op("INPUT", 0),
        _op("OUTPUT", 0),
    )
}

#: Float ops that the verifier rejects on sight. Declared so the error message
#: can name the offender instead of saying "unknown op".
FLOAT_OPS: frozenset[str] = frozenset(
    {
        "FADD",
        "FSUB",
        "FMUL",
        "FDIV",
        "FDOT",
        "FCMP",
        "EXP",
        "LOG",
        "TANH",
        "SIGMOID",
        "SOFTMAX",
        "GELU",
        "SQRT",
        "POW",
        "SCALE",
        "FMA",
    }
)


def op_spec(name: str) -> OpSpec:
    """Look up an op by name.

    Args:
        name: Canonical opcode.

    Returns:
        The registered :class:`OpSpec`.

    Raises:
        KeyError: If ``name`` is not a whitelisted op.
    """
    try:
        return WHITELIST[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a BIR whitelisted op") from None


def is_allowed(name: str, width_bits: int = INT_WIDTH_LIMIT) -> bool:
    """Report whether an op/width pair satisfies invariant I3.

    Args:
        name: Canonical opcode.
        width_bits: Operand width in bits. Only meaningful for arithmetic ops.

    Returns:
        True if the op is whitelisted and the width is within limit.
    """
    if name in FLOAT_OPS:
        return False
    spec = WHITELIST.get(name)
    if spec is None or spec.is_float:
        return False
    return width_bits <= spec.max_width
