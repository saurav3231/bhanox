"""BIR (Bhanox Intermediate Representation) -- the portability contract.

Purpose: hold the op registry and the invariant-I3 verifier. Every backend
(NumPy reference, PyTorch training, future native runtime) must be expressible
as a graph of whitelisted ops, which is what makes "runs on a CPU that does not
exist yet" a checkable claim rather than a slogan.
"""

from __future__ import annotations

from bhanox.ir.registry import INT_WIDTH_LIMIT, WHITELIST, OpSpec, is_allowed, op_spec
from bhanox.ir.verifier import (
    Graph,
    I3ViolationError,
    Node,
    assert_inference_safe,
    verify,
)

__all__ = [
    "INT_WIDTH_LIMIT",
    "WHITELIST",
    "Graph",
    "I3ViolationError",
    "Node",
    "OpSpec",
    "assert_inference_safe",
    "is_allowed",
    "op_spec",
    "verify",
]
