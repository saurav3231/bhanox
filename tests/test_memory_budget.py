"""Tests for the spec-D7 memory budgets.

The claim being defended: both of Bhanox's memories are explicit, bounded byte
ceilings a caller sets, and the process never exceeds them. A Transformer cannot
cap its KV cache; a Mamba state is fixed at construction. This file is where
that claim is either true or not.

Each budget test has a companion that shows the gate can fail, because a
ceiling that cannot be exceeded is not a ceiling.
"""

from __future__ import annotations

import numpy as np
import pytest

import bhanox
from bhanox.budgets import (
    ENTRY_META_BYTES,
    format_bytes,
    parse_bytes,
    perm_entry_bytes,
    temp_state_bytes,
)
from bhanox.config import load_config
from bhanox.memory.vectorvault import VectorVault
from bhanox.model import Bhanox


def vault_cfg(**over):
    """A nano config with the vault on, so perm memory exists."""
    return load_config({"name": "nano", "use_vault": True, **over})


def rng_vecs(n: int, d: int, seed: int = 0):
    """`n` random cue/value pairs."""
    rng = np.random.default_rng(seed)
    return [
        (
            rng.standard_normal(d).astype(np.float32),
            rng.standard_normal(d).astype(np.float32),
        )
        for _ in range(n)
    ]


class TestParseBytes:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("1024", 1024),
            ("1KB", 1024),
            ("1K", 1024),
            ("64MB", 64 * 1024**2),
            ("2GB", 2 * 1024**3),
            ("1.5MB", int(1.5 * 1024**2)),
            (" 8 mb ", 8 * 1024**2),
        ],
    )
    def test_parses_every_spelling(self, spec, expected) -> None:
        assert parse_bytes(spec) == expected

    def test_int_passes_through(self) -> None:
        assert parse_bytes(4096) == 4096

    def test_none_means_uncapped(self) -> None:
        assert parse_bytes(None) is None

    @pytest.mark.parametrize("bad", ["", "MB", "-1MB", "1XB", "1 2 MB", "many"])
    def test_rejects_nonsense(self, bad) -> None:
        with pytest.raises(ValueError):
            parse_bytes(bad)

    def test_zero_is_rejected(self) -> None:
        """A zero budget cannot hold anything, so it is a mistake, not a setting."""
        with pytest.raises(ValueError, match="> 0"):
            parse_bytes("0MB")


class TestSizing:
    def test_temp_matches_the_measured_state(self) -> None:
        """The formula must agree with the arrays, or the budget is fiction.

        This is the check that caught the design-phase formula: it predicted
        1,024 B for nano where the real state is 2,048 B.
        """
        for name in ("nano", "mini"):
            model = Bhanox(load_config(name))
            assert model.config.required_temp_bytes() == model.state_nbytes()

    def test_n_banks_does_not_change_temp_bytes(self) -> None:
        """The banks are the decay schedule, not memory slots.

        The spec's `L * B * 2d` implied B bought memory. It does not: a wider
        bank mix only changes the decay rates, so this must stay flat. If a
        future change makes B allocate, this test is the thing to revisit.
        """
        narrow = load_config({"name": "custom", "n_banks": 4, "n_layers": 1})
        wide = load_config({"name": "custom", "n_banks": 16, "n_layers": 1})
        assert narrow.required_temp_bytes() == wide.required_temp_bytes()

    def test_entry_meta_matches_the_stored_record(self) -> None:
        """ENTRY_META_BYTES is measured from the dtypes, not guessed."""
        v = VectorVault(n_slots=4, d_value=8, bits=64)
        measured = (
            v.keys[0].nbytes
            + v.values[0].nbytes
            + v.salience[0].nbytes
            + v.age[0].nbytes
        )
        assert perm_entry_bytes(v.bits, v.d_value) == measured
        assert measured - (v.bits // 8 + 4 * v.d_value) == ENTRY_META_BYTES

    def test_the_key_dominates_the_entry(self) -> None:
        """Why the spec's `2d + 16` was 14x low: it forgot the hypervector."""
        key = 8192 // 8
        assert key > 4 * 32
        assert perm_entry_bytes(8192, 32) == 1024 + 128 + ENTRY_META_BYTES

    def test_format_round_trips(self) -> None:
        assert parse_bytes(format_bytes(64 * 1024**2)) == 64 * 1024**2
        assert format_bytes(900) == "900B"
        assert format_bytes(1024) == "1.0KB"


class TestTempBudget:
    def test_default_is_uncapped_and_matches_requirement(self) -> None:
        cfg = load_config("nano")
        assert cfg.temp_mem_bytes is None
        assert cfg.required_temp_bytes() == temp_state_bytes(
            cfg.n_layers, cfg.n_heads, cfg.d_k, cfg.d_v
        )

    def test_load_config_accepts_both_budgets(self) -> None:
        cfg = load_config("mini", temp_mem="64MB", perm_mem="2GB")
        assert cfg.temp_mem_bytes == 64 * 1024**2
        assert cfg.perm_mem_bytes == 2 * 1024**3

    def test_omitting_budgets_leaves_the_preset_untouched(self) -> None:
        """`load_config("nano")` must behave exactly as it did pre-D7."""
        assert load_config("nano").perm_mem_bytes is None
        assert load_config("nano").temp_mem_bytes is None

    def test_oversubscribing_temp_is_legal_but_reported(self) -> None:
        """Oversubscribing is legal. Silently pretending it fits is not."""
        model = Bhanox(load_config("nano", temp_mem="1KB"))
        report = model.memory_report()
        assert report["temp"]["within_budget"] is False
        assert report["warnings"], "an exceeded budget must be reported"
        assert "not adjustable" in report["warnings"][0]

    def test_temp_is_not_adjustable_at_runtime(self) -> None:
        """No `set_temp_budget`: the state is fixed by the trained weights."""
        assert not hasattr(Bhanox, "set_temp_budget")

    def test_a_generous_temp_budget_reports_clean(self) -> None:
        model = Bhanox(load_config("nano", temp_mem="1MB"))
        report = model.memory_report()
        assert report["temp"]["within_budget"] is True
        assert report["warnings"] == []
        assert report["temp"]["adjustable"] is False


class TestPermBudget:
    def test_defaults_to_uncapped(self) -> None:
        v = VectorVault(n_slots=8, d_value=8, bits=64)
        assert v.perm_budget_bytes is None
        assert v.max_admissible_entries() == 8

    def test_usage_stays_within_budget_however_much_is_written(self) -> None:
        """The core D7 claim, checked by brute force."""
        v = VectorVault(n_slots=512, d_value=16, bits=256)
        v.set_budget("8KB")
        for key, val in rng_vecs(400, 16):
            v.write(key, val, surprise=1.0)
            assert v.nbytes <= 8 * 1024, "reservation escaped the budget"
            assert v.used_bytes <= 8 * 1024, "usage escaped the budget"

    def test_the_budget_bounds_bytes_not_merely_admission(self) -> None:
        """A budget that only gated writes would still hold the arrays.

        This is the check that matters: the slot table is allocated up front, so
        admission-only enforcement would report 64KB while spending 1.5MB.
        """
        v = VectorVault(n_slots=1024, d_value=128, bits=8192)
        assert v.nbytes > 1024**2, "precondition: the table starts out huge"
        v.set_budget("64KB")
        assert v.nbytes <= 64 * 1024
        assert v.n_slots == 64 * 1024 // v.entry_bytes

    def test_a_loose_budget_does_not_allocate_it(self) -> None:
        """A budget is a ceiling, not a purchase order.

        `set_budget("4GB")` must not allocate 4GB. The entry cap governs how
        much is actually reserved; an earlier draft resized the table up to the
        budget and tried to allocate 2,084,935 slots for a 4GB ceiling, which is
        precisely the OOM D7 forbids.
        """
        v = VectorVault(n_slots=1024, d_value=128, bits=8192)
        before = v.nbytes
        v.set_budget("4GB")
        assert v.nbytes == before
        assert v.n_slots == 1024
        assert v.entry_cap == 1024

    def test_the_entry_cap_and_the_budget_both_bind(self) -> None:
        """Whichever is tighter wins, and the report has to show it."""
        v = VectorVault(n_slots=1024, d_value=16, bits=256)
        assert v.max_admissible_entries() == 1024
        v.set_budget("8KB")  # far tighter than 1024 entries
        assert v.max_admissible_entries() == 8 * 1024 // v.entry_bytes
        v.set_budget("64MB")  # looser than the cap
        assert v.max_admissible_entries() == 1024

    def test_set_entry_cap_resizes_live(self) -> None:
        """E is the count knob, and it is changeable at runtime."""
        v = VectorVault(n_slots=64, d_value=8, bits=64)
        v.set_entry_cap(16)
        assert v.n_slots == 16
        v.set_entry_cap(128)
        assert v.n_slots == 128
        assert v.max_admissible_entries() == 128

    def test_entry_cap_cannot_drop_stored_entries(self) -> None:
        """Refuse rather than silently discard: truncation is set_budget's job."""
        v = VectorVault(n_slots=64, d_value=8, bits=64)
        for key, val in rng_vecs(20, 8):
            v.write(key, val, surprise=1.0)
        with pytest.raises(ValueError, match="while holding"):
            v.set_entry_cap(4)
        assert v.filled == 20

    def test_growing_the_cap_preserves_stored_entries(self) -> None:
        v = VectorVault(n_slots=64, d_value=4, bits=64)
        v.write(np.ones(4, np.float32), np.full(4, 5.0, np.float32), surprise=2.0)
        v.set_entry_cap(256)
        assert v.filled == 1
        assert np.array_equal(v.values[0], np.full(4, 5.0, np.float32))

    def test_a_temp_budget_survives_a_live_perm_resize(self) -> None:
        """Regression: the two budgets are independent.

        `set_perm_budget` goes through `with_memory_budgets`, which used to
        rebuild both fields from its arguments and so reset `temp_mem_bytes` to
        None -- silently turning a 64MB temp ceiling into "uncapped" on the
        first live resize.
        """
        model = Bhanox(load_config("nano", temp_mem="64KB", perm_mem="1MB"))
        assert model.config.temp_mem_bytes == 64 * 1024
        model.set_perm_budget("2MB")
        assert model.config.temp_mem_bytes == 64 * 1024, "temp budget was clobbered"
        assert model.config.perm_mem_bytes == 2 * 1024**2
        assert model.memory_report()["temp"]["budget_human"] == "64.0KB"

    def test_perm_mem_implies_a_vault(self) -> None:
        """Asking for a permanent budget must not be silently ignored.

        Every preset ships with the vault off, so `load_config("mini",
        perm_mem="2GB")` would otherwise accept the budget and drop it, and
        `set_perm_budget` would then raise on a model with nothing to budget.
        """
        cfg = load_config("mini", perm_mem="2GB")
        assert cfg.use_vault is True
        model = Bhanox(cfg)
        model.set_perm_budget("1MB")
        assert model.vault is not None
        assert model.vault.nbytes <= 1024**2

    def test_the_config_budget_reaches_the_vault(self) -> None:
        """Regression: the config accepted "2GB" and the vault ignored it.

        The model built the vault and never handed it `cfg.perm_mem_bytes`, so
        the report showed `budget_bytes: null` for a model configured with 2GB.
        A budget that survives parsing but not construction is worse than one
        that was refused.
        """
        model = Bhanox(load_config("nano", perm_mem="256KB"))
        assert model.vault is not None
        assert model.vault.perm_budget_bytes == 256 * 1024
        report = model.memory_report()
        assert report["perm"]["budget_bytes"] == 256 * 1024
        assert report["perm"]["entry_limit"] == 256 * 1024 // model.vault.entry_bytes

    def test_a_config_budget_tighter_than_one_entry_is_honoured(self) -> None:
        """Construction must not blow up on a budget the vault cannot use."""
        model = Bhanox(load_config("nano", perm_mem="8B"))
        assert model.vault is not None
        assert model.vault.perm_budget_bytes == 8
        assert model.vault.max_admissible_entries() == 0

    def test_a_budget_below_one_entry_truncates_to_nothing(self) -> None:
        """Undersubscribing must never raise.

        D7 says running out of room costs recall, not the process. A budget
        smaller than a single entry used to raise ValueError, which handed the
        caller an exception where the whole point of a memory is to be the thing
        that gives way.
        """
        v = VectorVault(n_slots=32, d_value=8, bits=64)
        for key, val in rng_vecs(8, 8):
            v.write(key, val, surprise=1.0)
        assert v.filled == 8
        v.set_budget(16)  # far below entry_bytes
        assert v.filled == 0
        assert v.nbytes <= 16
        assert v.max_admissible_entries() == 0

    def test_a_zero_budget_keeps_the_vault_writable(self) -> None:
        """A zero ceiling means nothing is retained, not that writes crash."""
        v = VectorVault(n_slots=8, d_value=4, bits=64)
        v.set_budget(0)
        key = np.ones(4, np.float32)
        v.write(key, np.ones(4, np.float32), surprise=1.0)
        assert v.filled == 0
        assert v.query(key)[0].size == 0
        assert v.retrieve(key) is None
        v.set_budget(None)  # lifting restores the cap
        assert v.n_slots == 8

    def test_temp_mem_alone_does_not_enable_the_vault(self) -> None:
        """Only the permanent budget implies a permanent memory."""
        cfg = load_config("nano", temp_mem="64KB")
        assert cfg.use_vault is False
        assert cfg.perm_mem_bytes is None

    def test_lowering_truncates_instead_of_raising(self) -> None:
        """Running out of room must cost recall, not the process."""
        v = VectorVault(n_slots=256, d_value=16, bits=256)
        for key, val in rng_vecs(60, 16):
            v.write(key, val, surprise=1.0)
        assert v.filled == 60
        v.set_budget("2KB")
        assert v.filled < 60
        assert v.used_bytes <= 2 * 1024

    def test_truncation_keeps_the_most_important(self) -> None:
        """Truncation uses the importance score, not slot order.

        Slot order would throw away whatever happened to sit at a high index,
        which after a fresh build is arbitrary.
        """
        v = VectorVault(n_slots=64, d_value=4, bits=64)
        v.write(np.ones(4, np.float32), np.full(4, 7.0, np.float32), surprise=100.0)
        for i in range(20):
            v.write(
                np.full(4, float(i + 2), np.float32),
                np.full(4, float(i), np.float32),
                surprise=0.01,
            )
        v.set_budget(perm_entry_bytes(v.bits, v.d_value) * 3)
        kept = [row[0] for row in v.values[: v.filled]]
        assert 7.0 in kept, "the high-salience entry was truncated first"

    def test_writes_stop_admitting_once_full(self) -> None:
        v = VectorVault(n_slots=64, d_value=8, bits=64)
        v.set_budget(perm_entry_bytes(v.bits, v.d_value) * 4)
        for key, val in rng_vecs(20, 8):
            v.write(key, val, surprise=1.0)
        assert v.filled == 4
        assert (
            v.write(np.ones(8, np.float32), np.ones(8, np.float32), surprise=9.0)
            is False
        )

    def test_set_budget_is_idempotent(self) -> None:
        """A Kaggle session can die and be re-run; re-applying must be safe."""
        v = VectorVault(n_slots=64, d_value=8, bits=64)
        for key, val in rng_vecs(20, 8):
            v.write(key, val, surprise=1.0)
        v.set_budget("1KB")
        first = (v.n_slots, v.filled, v.nbytes)
        for _ in range(3):
            v.set_budget("1KB")
        assert (v.n_slots, v.filled, v.nbytes) == first

    def test_lifting_the_cap_restores_slot_capacity(self) -> None:
        v = VectorVault(n_slots=64, d_value=8, bits=64)
        v.set_budget("1KB")
        v.set_budget(None)
        assert v.perm_budget_bytes is None
        assert v.max_admissible_entries() == v.n_slots

    def test_an_uncapped_vault_still_evicts(self) -> None:
        """Budgets must not have broken the pre-D7 eviction path."""
        v = VectorVault(n_slots=2, d_value=4, bits=64)
        for i in range(5):
            v.write(
                np.full(4, float(i + 1), np.float32), np.full(4, float(i), np.float32)
            )
        assert v.filled == 2


class TestPublicSurface:
    def test_set_perm_budget_on_a_model_with_a_vault(self) -> None:
        model = Bhanox(vault_cfg(perm_mem="64KB"))
        model.set_perm_budget("32KB")
        assert model.vault.nbytes <= 32 * 1024
        assert model.config.perm_mem_bytes == 32 * 1024

    def test_set_perm_budget_without_a_vault_says_so(self) -> None:
        model = Bhanox(load_config("nano"))
        assert model.vault is None
        with pytest.raises(ValueError, match="use_vault"):
            model.set_perm_budget("1MB")

    def test_memory_report_is_exported(self) -> None:
        assert "memory_report" in bhanox.__all__
        assert callable(bhanox.memory_report)

    def test_memory_report_text_names_both_memories(self) -> None:
        model = Bhanox(vault_cfg(perm_mem="64KB"))
        text = bhanox.memory_report(model)
        assert "temp" in text and "perm" in text
        assert "8.0KB" in text

    def test_memory_report_handles_a_model_without_a_vault(self) -> None:
        text = bhanox.memory_report(Bhanox(load_config("nano")))
        assert "no vault" in text

    def test_memory_report_returns_the_same_dict_it_prints(self) -> None:
        """One source of truth: tests assert the dict, users read the text."""
        model = Bhanox(vault_cfg(perm_mem="64KB"))
        data = bhanox.memory_report(model, as_text=False)
        assert data == model.memory_report()
        assert data["perm"]["within_budget"] is True
        assert data["temp"]["within_budget"] is True


class TestGenerationMemoryIsFlat:
    def test_generation_adds_no_permanent_memory(self) -> None:
        """The D6/D7 tie-in: generation must not grow either memory.

        The temporary state is fixed by shape. The vault only grows if something
        salient is written, so feed a prompt and confirm the delta is zero --
        which is the claim that distinguishes this from attention's cache.
        """
        cfg = vault_cfg()
        model = Bhanox(cfg)
        before_temp = model.state_nbytes()
        before_perm = model.vault.nbytes
        model.generate(np.arange(8, dtype=np.int64), max_new=32)
        assert model.state_nbytes() == before_temp
        assert model.vault.nbytes == before_perm

    def test_state_is_flat_across_context_length(self) -> None:
        """A longer prompt must not cost more state."""
        model = Bhanox(load_config("nano"))
        short = model.state_nbytes()
        model.generate(np.arange(4, dtype=np.int64), max_new=16)
        model.reset()
        model.generate(np.arange(512, dtype=np.int64), max_new=16)
        assert model.state_nbytes() == short
