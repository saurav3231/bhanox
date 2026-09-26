"""Tests for the I2/I3 audit.

These are the numbers the project publishes, so they are pinned here. If a change
moves one, that is a real change to a public claim, not a test to be quietly
updated.
"""

from __future__ import annotations

import pytest

from bhanox.audit import (
    INFERENCE_OPS,
    audit_bytes_per_token,
    audit_ops,
)
from bhanox.config import load_config
from bhanox.ir.verifier import I3ViolationError, Node, assert_inference_safe
from bhanox.model import Bhanox

# Measured from the arrays at audit time, not targeted. See docs/benchmarks.md.
EXPECTED = {
    "nano": (2_107_460, 8_192, True, 110_592),
    "mini": (24_338_948, 131_072, True, 720_896),
    "small": (324_929_540, 1_048_576, False, 3_538_944),
}
PRESETS = list(EXPECTED)


@pytest.fixture(scope="module")
def models() -> dict[str, Bhanox]:
    """One instance per preset. ``small`` is 325 M parameters, so rebuilding it
    per test dominates the runtime of this file."""
    return {name: Bhanox(load_config(name)) for name in PRESETS}


@pytest.fixture(scope="module")
def reports(models: dict[str, Bhanox]) -> dict[str, object]:
    return {name: audit_bytes_per_token(m) for name, m in models.items()}


class TestReportedFigures:
    @pytest.mark.parametrize("name", PRESETS)
    def test_parameter_count(self, name: str, models: dict[str, Bhanox]) -> None:
        assert models[name].param_count() == EXPECTED[name][0]

    @pytest.mark.parametrize("name", PRESETS)
    def test_state_bytes(self, name: str, reports: dict[str, object]) -> None:
        assert reports[name].state_total == EXPECTED[name][1]

    @pytest.mark.parametrize("name", PRESETS)
    def test_l2_verdict(self, name: str, reports: dict[str, object]) -> None:
        assert reports[name].passed is EXPECTED[name][2]

    @pytest.mark.parametrize("name", PRESETS)
    def test_bytes_per_layer(self, name: str, reports: dict[str, object]) -> None:
        assert reports[name].layer_total == EXPECTED[name][3]

    def test_small_is_the_known_i2_failure(self, reports: dict[str, object]) -> None:
        """Recorded so the failure stays visible rather than being tuned away."""
        assert reports["small"].utilization > 1.0
        assert reports["nano"].utilization < 0.5

    def test_utilization_matches_the_printed_ratio(
        self, reports: dict[str, object]
    ) -> None:
        rep = reports["nano"]
        assert rep.utilization == pytest.approx(rep.layer_total / rep.l2_bytes)

    def test_summary_states_the_verdict_and_ratio(
        self, reports: dict[str, object]
    ) -> None:
        assert "PASS" in reports["nano"].summary()
        assert "FAIL" in reports["small"].summary()
        assert "10.5%" in reports["nano"].summary()

    def test_summary_is_labelled_measured(self, reports: dict[str, object]) -> None:
        assert "measured" in reports["nano"].summary()


class TestLayerAccounting:
    def test_a_layer_is_memory_plus_mixer(self, reports: dict[str, object]) -> None:
        assert [c.name for c in reports["nano"].per_layer] == [
            "deltabank",
            "microexpert",
        ]

    def test_layer_total_is_the_sum_of_its_parts(
        self, reports: dict[str, object]
    ) -> None:
        rep = reports["nano"]
        assert sum(c.nbytes for c in rep.per_layer) == rep.layer_total

    def test_the_front_end_is_charged_once_not_per_layer(
        self, reports: dict[str, object]
    ) -> None:
        """The front-end is charged once for the whole model, so it is reported
        separately and left out of the per-layer sum I2 is stated against.
        Counting it in both is what first made the nano/mini figures look
        better than they were."""
        rep = reports["nano"]
        assert rep.front_end > 0
        assert all(not c.name.startswith("hashbind") for c in rep.per_layer)
        assert rep.total_per_token == rep.front_end + rep.layer_total

    def test_the_front_end_touches_far_less_than_it_resides(
        self, reports: dict[str, object], models: dict[str, Bhanox]
    ) -> None:
        assert reports["nano"].front_end < models["nano"].embedder.nbytes

    def test_memory_dominates_the_layer(self, reports: dict[str, object]) -> None:
        rep = reports["nano"]
        mixer = next(c for c in rep.per_layer if c.name == "microexpert")
        bank = next(c for c in rep.per_layer if c.name == "deltabank")
        assert bank.nbytes > mixer.nbytes

    def test_the_mixer_is_far_smaller_than_dense(
        self, reports: dict[str, object], models: dict[str, Bhanox]
    ) -> None:
        """The sparsity claim, pinned: the mixer must touch the shared plus
        top-k experts, not all of them."""
        rep = reports["nano"]
        mixer = next(c for c in rep.per_layer if c.name == "microexpert")
        assert mixer.nbytes == models["nano"].mixers[0].active_nbytes()
        assert models["nano"].mixers[0].dense_nbytes() > 5 * mixer.nbytes

    def test_bytes_are_whole_numbers(self, reports: dict[str, object]) -> None:
        for name in PRESETS:
            for c in reports[name].per_layer:
                assert isinstance(c.nbytes, int)

    def test_every_item_carries_a_detail(self, reports: dict[str, object]) -> None:
        assert all(c.detail for c in reports["nano"].per_layer)


class TestOpWhitelist:
    def test_no_float_multiply_is_whitelisted(self) -> None:
        banned = {"fmatmul", "dot", "matmul", "float", "fadd", "fmul", "div"}
        assert not banned & set(INFERENCE_OPS)

    def test_the_op_count_is_pinned(self) -> None:
        assert len(INFERENCE_OPS) == 11

    def test_the_report_carries_the_verified_list(
        self, reports: dict[str, object]
    ) -> None:
        assert tuple(reports["nano"].op_check) == INFERENCE_OPS

    def test_every_report_verifies_the_whitelist(
        self, reports: dict[str, object]
    ) -> None:
        for name in PRESETS:
            assert set(reports[name].op_check) <= set(INFERENCE_OPS)

    def test_the_whitelist_is_clean(self) -> None:
        assert_inference_safe(INFERENCE_OPS, where="inference op list")

    def test_a_float_op_is_rejected(self) -> None:
        with pytest.raises(I3ViolationError):
            assert_inference_safe((*INFERENCE_OPS, "FMUL"), where="test")

    def test_a_wide_multiply_is_rejected(self) -> None:
        """The width limit lives on the graph, not on the bare op list, so this
        has to go through a real Node."""
        with pytest.raises(I3ViolationError, match="8-bit limit"):
            audit_ops([Node("MUL_I8", width_bits=32)])

    def test_an_eight_bit_multiply_is_accepted(self) -> None:
        assert audit_ops([Node("MUL_I8", width_bits=8)]) is None

    def test_audit_ops_verifies_a_clean_graph(self) -> None:
        assert audit_ops([Node("LOAD_I8"), Node("ADD"), Node("STORE_I8")]) is None

    def test_audit_ops_rejects_a_float_graph(self) -> None:
        with pytest.raises(I3ViolationError):
            audit_ops([Node("FMUL")])

    def test_audit_ops_rejects_an_unknown_op(self) -> None:
        with pytest.raises(I3ViolationError, match="whitelist"):
            audit_ops([Node("MUL_I32:I8")])
