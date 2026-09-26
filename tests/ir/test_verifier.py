"""Tests for the BIR op registry and the invariant-I3 verifier."""

from __future__ import annotations

import pytest

from bhanox.ir import registry, verifier


class TestRegistry:
    def test_whitelist_matches_frozen_spec(self) -> None:
        expected = {
            "ADD",
            "SUB",
            "CMP",
            "SHIFT",
            "PERMUTE",
            "POPCOUNT",
            "LUT",
            "MUL_I8",
            "LOAD_I8",
            "STORE_I8",
            "SAT",
        }
        assert expected <= set(registry.WHITELIST)

    def test_no_whitelisted_op_is_float(self) -> None:
        assert not [s for s in registry.WHITELIST.values() if s.is_float]

    def test_op_spec_raises_on_unknown(self) -> None:
        with pytest.raises(KeyError, match="whitelisted"):
            registry.op_spec("MUL_FP32")

    @pytest.mark.parametrize("op", ["FMUL", "SOFTMAX", "GELU", "EXP"])
    def test_float_ops_are_rejected(self, op: str) -> None:
        assert not registry.is_allowed(op)

    def test_may_not_exceed_eight_bits(self) -> None:
        assert registry.is_allowed("MUL_I8", 8)
        assert not registry.is_allowed("MUL_I8", 16)
        assert not registry.is_allowed("MUL_I8", 32)


class TestVerifier:
    def test_accepts_whitelisted_graph(self) -> None:
        graph = verifier.Graph(
            "delta",
            (
                verifier.Node("LOAD_I8"),
                verifier.Node("MUL_I8", 8),
                verifier.Node("SAT"),
                verifier.Node("ADD"),
                verifier.Node("STORE_I8"),
            ),
        )
        assert verifier.verify(graph) is graph

    def test_rejects_float_op(self) -> None:
        graph = verifier.Graph("bad", (verifier.Node("LOAD_I8"), verifier.Node("FMUL")))
        with pytest.raises(verifier.I3ViolationError, match="floating-point"):
            verifier.verify(graph)

    def test_rejects_wide_multiply(self) -> None:
        graph = verifier.Graph("bad", (verifier.Node("MUL_I8", 32),))
        with pytest.raises(verifier.I3ViolationError, match="exceeds"):
            verifier.verify(graph)

    def test_rejects_unknown_op(self) -> None:
        graph = verifier.Graph("bad", (verifier.Node("ATTENTION"),))
        with pytest.raises(verifier.I3ViolationError, match="not in the BIR whitelist"):
            verifier.verify(graph)

    def test_error_names_the_offending_index(self) -> None:
        graph = verifier.Graph("layer0", (verifier.Node("ADD"), verifier.Node("FMUL")))
        with pytest.raises(verifier.I3ViolationError, match=r"layer0\[1\]"):
            verifier.verify(graph)

    def test_assert_inference_safe_parses_width_annotations(self) -> None:
        verifier.assert_inference_safe(["MUL_I8:8", "POPCOUNT"], where="unit")
        with pytest.raises(verifier.I3ViolationError):
            verifier.assert_inference_safe(["MUL_I8:16"], where="unit")

    def test_empty_graph_is_valid(self) -> None:
        assert len(verifier.verify(verifier.Graph("empty"))) == 0


def test_module_selfcheck_passes() -> None:
    """The spec's runnable self-check must actually pass."""
    verifier._demo()
