"""Tests for model configs, presets, and invariant I1 (head-load)."""

from __future__ import annotations

import json

import pytest

from bhanox.config import (
    PRESETS,
    BhanoxConfig,
    available_presets,
    concurrent_writes,
    load_config,
    register_preset,
)


class TestPresets:
    def test_three_presets_exist(self) -> None:
        assert available_presets() == ("mini", "nano", "small")

    @pytest.mark.parametrize(
        ("name", "d_model", "n_layers", "n_heads", "d_k"),
        [("nano", 128, 4, 4, 16), ("mini", 256, 8, 8, 32), ("small", 512, 16, 16, 64)],
    )
    def test_frozen_shapes(
        self, name: str, d_model: int, n_layers: int, n_heads: int, d_k: int
    ) -> None:
        """Architecture D4 freezes these. Changing one is a design change."""
        cfg = load_config(name)
        assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_k) == (
            d_model,
            n_layers,
            n_heads,
            d_k,
        )

    def test_ternary_is_off_below_small(self) -> None:
        assert not PRESETS["nano"].ternary
        assert not PRESETS["mini"].ternary

    def test_vault_off_for_small_configs(self) -> None:
        assert not PRESETS["nano"].use_vault
        assert PRESETS["small"].use_vault

    def test_decay_rates_are_the_frozen_prior(self) -> None:
        cfg = load_config("nano")
        assert cfg.decay_rates[0] == pytest.approx(0.5)
        assert cfg.decay_rates[-1] == pytest.approx(1.0 - 2.0**-cfg.n_banks)
        assert all(0.0 < r < 1.0 for r in cfg.decay_rates)


class TestInvariantI1:
    def test_all_presets_satisfy_i1(self) -> None:
        for name in available_presets():
            cfg = load_config(name)
            assert cfg.head_load < cfg.d_k, name

    def test_presets_leave_exactly_2x_headroom(self) -> None:
        """The frozen table has d_k == 4 * n_layers, i.e. 2x margin (ADR-005)."""
        for name in available_presets():
            cfg = load_config(name)
            assert cfg.d_k == pytest.approx(4 * cfg.n_layers), name

    def test_concurrent_writes_formula(self) -> None:
        """write_rate * 1/(1 - lambda) with the shallowest bank lambda = 0.5."""
        assert concurrent_writes(4, 8) == pytest.approx(8.0)
        assert concurrent_writes(1, 8) == pytest.approx(2.0)

    def test_violation_raises_with_actionable_message(self) -> None:
        with pytest.raises(ValueError) as exc:
            BhanoxConfig(name="bad", d_k=4, n_layers=8, n_heads=2, d_v=8)
        msg = str(exc.value)
        assert "I1" in msg
        assert "concurrent_writes" in msg
        assert "d_k" in msg

    def test_violation_names_a_legal_alternative(self) -> None:
        """The error should hand the user a number they can paste."""
        with pytest.raises(ValueError) as exc:
            BhanoxConfig(name="bad", d_k=4, n_layers=8, n_heads=2, d_v=8)
        assert "Raise d_k" in str(exc.value)

    def test_more_heads_does_not_relieve_i1(self) -> None:
        """I1 is per head, so adding heads is not a fix -- worth pinning down."""
        with pytest.raises(ValueError, match="I1"):
            BhanoxConfig(name="bad", d_k=4, n_layers=8, n_heads=64, d_v=8)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"d_model": 0},
            {"n_layers": -1},
            {"top_k": 0},
            {"d_k": -4},
            {"l2_bytes": 0},
            {"output_vocab": 0},
        ],
    )
    def test_rejects_non_positive_fields(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            BhanoxConfig(name="bad", **kwargs)

    def test_rejects_top_k_above_expert_pool(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            BhanoxConfig(
                name="bad",
                d_model=32,
                n_experts=2,
                top_k=8,
                d_k=8,
                n_layers=1,
                d_v=8,
                d_expert=8,
            )

    def test_rejects_no_experts_at_all(self) -> None:
        with pytest.raises(ValueError, match="at least one expert"):
            BhanoxConfig(
                name="bad",
                d_model=32,
                n_experts=0,
                n_shared_experts=0,
                d_k=8,
                n_layers=1,
                d_v=8,
                d_expert=8,
            )

    def test_rejects_ternary_below_small_scale(self) -> None:
        """Design phase: full-ternary collapses at small scale (BPC 22.3)."""
        cfg = load_config("nano")
        with pytest.raises(ValueError, match="ternary is opt-in"):
            cfg.evolve(ternary=True)

    def test_allows_ternary_at_small(self) -> None:
        assert load_config("small").evolve(ternary=True).ternary

    def test_rejects_hashes_above_pool(self) -> None:
        with pytest.raises(ValueError, match="n_hashes"):
            BhanoxConfig(
                name="bad",
                d_model=32,
                n_hashes=99,
                pool_size=16,
                vocab_table=8,
                d_k=8,
                n_layers=1,
                d_v=8,
                d_expert=8,
            )


class TestSerialisation:
    def test_round_trip(self) -> None:
        cfg = load_config("mini")
        assert BhanoxConfig.from_dict(cfg.to_dict()) == cfg

    def test_from_dict_ignores_unknown_keys(self) -> None:
        data = load_config("nano").to_dict()
        data["some_future_field"] = 1
        assert BhanoxConfig.from_dict(data) == load_config("nano")

    def test_load_from_json_path(self, tmp_path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(load_config("nano").to_dict()), encoding="utf-8")
        assert load_config(path) == load_config("nano")

    def test_unknown_name_lists_presets(self) -> None:
        with pytest.raises(KeyError, match="nano"):
            load_config("gigantic")

    def test_passthrough(self) -> None:
        cfg = load_config("nano")
        assert load_config(cfg) is cfg

    def test_evolve_revalidates(self) -> None:
        with pytest.raises(ValueError, match="I1"):
            load_config("nano").evolve(d_k=4)


def test_register_preset() -> None:
    cfg = BhanoxConfig(
        name="unit-test-tiny",
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_k=8,
        d_v=8,
        d_expert=8,
        n_experts=2,
        top_k=1,
    )
    try:
        register_preset(cfg)
        assert load_config("unit-test-tiny") is cfg
    finally:
        PRESETS.pop("unit-test-tiny", None)
