"""BIR graph verifier -- the build-time enforcer of invariant I3.

Purpose: refuse to "build" an inference graph that contains floating-point
arithmetic or an integer multiply wider than 8 bits. Catching this statically is
the whole point: the 95% energy claim and the CPU-native thesis both collapse if
a single fp32 op sneaks into the generation path, and a silent ``float`` array
in a NumPy reference is exactly how that happens.

In simple words: this is the bouncer at the door of the inference graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from bhanox.ir.registry import FLOAT_OPS, INT_WIDTH_LIMIT, WHITELIST, OpSpec

__all__ = ["Graph", "I3ViolationError", "Node", "assert_inference_safe", "verify"]


class I3ViolationError(ValueError):
    """Raised when a graph contains an op that invariant I3 forbids."""


@dataclass(frozen=True)
class Node:
    """One operation in a BIR graph.

    Attributes:
        op: Canonical opcode, must be a key of :data:`WHITELIST`.
        width_bits: Operand width for arithmetic ops. MUL wider than
            :data:`INT_WIDTH_LIMIT` is a violation.
        attrs: Free-form annotations (shapes, dtypes). Never interpreted by the
            verifier -- kept so graphs stay debuggable.
    """

    op: str
    width_bits: int = INT_WIDTH_LIMIT
    attrs: Mapping[str, object] = field(default_factory=dict)

    @property
    def spec(self) -> OpSpec:
        """Return the registered spec, raising I3ViolationError if unregistered."""
        try:
            return WHITELIST[self.op]
        except KeyError:
            raise I3ViolationError(
                f"op {self.op!r} is not in the BIR whitelist "
                f"(I3). Allowed: {', '.join(sorted(WHITELIST))}"
            ) from None


@dataclass(frozen=True)
class Graph:
    """An ordered list of BIR nodes.

    Attributes:
        name: Graph label, used in error messages and audit output.
        nodes: Operations, in execution order.
    """

    name: str
    nodes: tuple[Node, ...] = ()

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)


def verify(graph: Graph) -> Graph:
    """Verify a graph against invariant I3.

    Checks, in order of cheapness: op is registered, op is not a float op, and
    integer arithmetic width is within :data:`INT_WIDTH_LIMIT`.

    Args:
        graph: Graph to check.

    Returns:
        The same graph, unchanged, when it passes -- so this composes as
        ``graph = verify(graph)`` at every construction site.

    Raises:
        I3ViolationError: On the first offending node, naming it and the graph.
    """
    for index, node in enumerate(graph):
        # Float ops are checked first, and separately, so the error can name the
        # real problem ("this is a float op") instead of the generic "not
        # whitelisted", which is also true but tells the reader nothing.
        if node.op in FLOAT_OPS:
            raise I3ViolationError(
                f"{graph.name}[{index}]: floating-point op {node.op!r} is "
                "forbidden in an inference graph (I3)"
            )
        spec = node.spec  # raises I3ViolationError for unregistered ops
        if spec.is_float:
            raise I3ViolationError(
                f"{graph.name}[{index}]: {node.op!r} is a floating-point op and "
                "is forbidden in an inference graph (I3)"
            )
        if node.width_bits > spec.max_width:
            raise I3ViolationError(
                f"{graph.name}[{index}]: {node.op!r} at {node.width_bits} bits "
                f"exceeds the {spec.max_width}-bit limit (I3). Only "
                "MUL_I8-class integer multiply is permitted."
            )
    return graph


def assert_inference_safe(ops: Iterable[str], *, where: str = "graph") -> None:
    """Verify a bare iterable of op names against I3.

    Convenience wrapper for modules that keep a flat op list rather than a
    :class:`Graph` -- e.g. an audit report rendered as text.

    Args:
        ops: Op names, with optional ``"MUL_I8:32"`` width annotations.
        where: Label for the error message.

    Raises:
        I3ViolationError: If any op is not whitelist-clean.
    """
    for raw in ops:
        op, _, width = raw.partition(":")
        graph = Graph(where, (Node(op, int(width) if width else INT_WIDTH_LIMIT),))
        verify(graph)


def _demo() -> None:
    """Self-check: the whitelist accepts integer ops and rejects a wide MUL."""
    verify(Graph("ok", (Node("MUL_I8", 8), Node("POPCOUNT"), Node("LUT"))))
    for bad in (Node("FMUL"), Node("MUL_I8", 32), Node("MATMUL_FP16")):
        try:
            verify(Graph("bad", (bad,)))
        except I3ViolationError:
            continue
        raise AssertionError(f"verifier accepted illegal op {bad.op!r}")
    print("I3 verifier OK: MUL_I8:8 accepted, FMUL / MUL_I8:32 / MATMUL_FP16 rejected")


if __name__ == "__main__":
    _demo()
