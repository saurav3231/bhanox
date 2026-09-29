"""VectorVault state through a checkpoint save/load round trip.

The bug this file exists for: the inventory walk carries arrays only, so a
vault's slot tables were restored while ``filled`` and ``clock`` came back at
their initial 0. The arrays were intact and the vault was useless -- ``query``
short-circuited on ``filled == 0`` and answered nothing, and ``write`` handed
out slot 0 on every call, overwriting one recovered entry per write while the
rest sat orphaned. Nothing raised anywhere along that path. Writing the scalars
without restoring them would reproduce the same silence one layer down, so
every test here checks that the values come *back*, not merely that they were
written.

Fixtures use a vault-enabled ``nano``: about 10 MB per checkpoint against
1.3 GB for ``small``, through the identical inventory and restore path.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bhanox import checkpoint as ck
from bhanox.checkpoint import (
    META_KEY,
    RUNTIME_KEY,
    RUNTIME_VERSION,
    load,
    read_meta,
    save,
)
from bhanox.config import load_config
from bhanox.model import Bhanox

#: Distinct surprises, so every slot's ``salience`` differs and a slot swap
#: cannot pass by coincidence. All above THETA_S, so all writes are admitted.
SURPRISES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]

#: Deliberately not the 0.5 default, and not equal to any surprise above, so a
#: restore that silently kept the constructor value would be caught.
THETA_S = 0.375

#: Comfortably above ``n_slots * entry_bytes`` (1024 * 1548 = 1_585_152), so
#: ``set_budget`` records a real ceiling without resizing the tables -- a
#: resize would change their shapes and make them unfittable in a fresh vault.
PERM_BUDGET = 4 * 1024 * 1024

#: Above every surprise in SURPRISES, for continuation writes that must land.
SOMEONE = 2.0


def vaulted(n_slots: int | None = None) -> Bhanox:
    """A nano carrying a real VectorVault, built through the config's switch.

    ``use_vault`` is the flag ``Bhanox.__post_init__`` reads, so this exercises
    the production construction path rather than grafting a vault on afterwards.
    """
    model = Bhanox(dataclasses.replace(load_config("nano"), use_vault=True))
    if n_slots is not None:
        model.vault.set_entry_cap(n_slots)
    return model


def fill(vault: Any, n: int = len(SURPRISES), seed: int = 0) -> list[np.ndarray]:
    """Write ``n`` known entries and return their keys, in write order."""
    rng = np.random.default_rng(seed)
    keys: list[np.ndarray] = []
    for i in range(n):
        key = rng.standard_normal(vault.d_value).astype(np.float32)
        value = rng.standard_normal(vault.d_value).astype(np.float32)
        vault.write(key, value, surprise=SURPRISES[i])
        keys.append(key)
    return keys


def forge(source: Path, dest: Path, mutate: Any) -> Path:
    """Copy a checkpoint, running ``mutate`` over its metadata first.

    The same forging trick ``test_checkpoint.py`` uses to provoke refusals, so
    the malformed cases below are real files on disk rather than mocked
    internals. The source is never modified, so a module-scoped fixture stays
    valid whichever test fails.
    """
    with np.load(source, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    meta = json.loads(bytes(payload[META_KEY]).decode("utf-8"))
    mutate(meta)
    payload[META_KEY] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez(dest, **payload)
    return dest


@pytest.fixture(scope="module")
def populated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint of a 12-entry vault with meaningful scalars, saved once."""
    path = tmp_path_factory.mktemp("vault") / "ck.npz"
    source = vaulted()
    fill(source.vault)
    source.vault.theta_s = THETA_S
    source.vault.set_budget(PERM_BUDGET)
    save(source, path, step=11)
    return path


class TestEmptyVault:
    def test_round_trip_keeps_the_empty_state(self, tmp_path: Path) -> None:
        """An empty vault must not come back populated, and a populated one
        must not come back empty: the scalars are restored either way."""
        path = tmp_path / "ck.npz"
        source = vaulted()
        assert source.vault.filled == 0 and source.vault.clock == 0
        save(source, path, step=0)

        fresh = vaulted()
        load(path, fresh)

        assert fresh.vault.filled == 0
        assert fresh.vault.clock == 0
        assert fresh.vault.theta_s == source.vault.theta_s
        assert fresh.vault.perm_budget_bytes is None
        assert not fresh.vault.keys.any()
        assert not fresh.vault.values.any()
        assert fresh.vault.salience.sum() == 0
        assert fresh.vault.age.sum() == 0

    def test_the_block_is_written_for_an_empty_vault(self, tmp_path: Path) -> None:
        """Recorded unconditionally, so "empty" is a saved fact rather than an
        absence the loader has to guess at."""
        path = tmp_path / "ck.npz"
        save(vaulted(), path)
        block = read_meta(path)[RUNTIME_KEY]["vectorvault"]
        assert block["filled"] == 0
        assert block["clock"] == 0
        assert block["version"] == RUNTIME_VERSION


class TestPopulatedVault:
    def test_every_array_and_scalar_is_restored_exactly(self, populated: Path) -> None:
        """The core claim. Arrays by exact equality, scalars by equality -- and
        the target is proven different from the source first, so a load that
        did nothing fails instead of passing."""
        source = load(populated)
        assert source.vault.filled == len(SURPRISES)

        fresh = vaulted()
        assert fresh.vault.filled == 0
        assert fresh.vault.clock == 0
        assert fresh.vault.theta_s != THETA_S
        assert fresh.vault.perm_budget_bytes is None

        load(populated, fresh)

        restored = dict(ck.inventory(fresh))
        for name, arr in ck.inventory(source):
            assert np.array_equal(arr, restored[name]), name
        assert fresh.vault.filled == len(SURPRISES)
        assert fresh.vault.clock == len(SURPRISES)
        assert fresh.vault.theta_s == THETA_S
        assert fresh.vault.perm_budget_bytes == PERM_BUDGET

    def test_slots_hold_distinct_keys_values_ages_and_salience(
        self, populated: Path
    ) -> None:
        """Distinctness, so the assertions cannot be satisfied by one repeated
        entry or by a uniform fill."""
        fresh = vaulted()
        load(populated, fresh)
        v = fresh.vault
        n = len(SURPRISES)
        assert len({v.keys[i].tobytes() for i in range(n)}) == n
        assert len({v.values[i].tobytes() for i in range(n)}) == n
        assert len(set(v.salience[:n].tolist())) == n
        # A write stamps age with the clock it just advanced, so ages are 1..N.
        assert v.age[:n].tolist() == list(range(1, n + 1))
        assert v.age[:n].max() <= v.clock

    def test_query_still_finds_each_known_entry(self, populated: Path) -> None:
        """The behaviour the bug destroyed: a restored vault answers queries,
        and answers them with the right entry rather than merely some entry."""
        fresh = vaulted()
        load(populated, fresh)
        for i, key in enumerate(_keys_of()):
            idx, similarity = fresh.vault.query(key, top_k=1)
            assert idx.size == 1, f"entry {i} became unreachable after a load"
            assert int(idx[0]) == i, f"entry {i} resolved to slot {int(idx[0])}"
            assert similarity[0] == pytest.approx(1.0)
            assert fresh.vault.retrieve(key) is not None

    def test_numpy_scalars_are_converted_for_json(self, tmp_path: Path) -> None:
        """``json.dumps`` raises on ``np.int64``, so the block must hold plain
        Python types whatever the field happens to contain."""
        path = tmp_path / "ck.npz"
        source = vaulted()
        fill(source.vault, 3)
        source.vault.filled = np.int64(source.vault.filled)
        source.vault.clock = np.int64(source.vault.clock)
        source.vault.theta_s = np.float32(0.25)
        source.vault.perm_budget_bytes = np.int64(PERM_BUDGET)
        save(source, path)

        block = read_meta(path)[RUNTIME_KEY]["vectorvault"]
        for key in ("filled", "clock", "n_slots"):
            assert type(block[key]) is int
        assert type(block["theta_s"]) is float
        assert type(block["perm_budget_bytes"]) is int

        fresh = vaulted()
        load(path, fresh)
        assert type(fresh.vault.filled) is int
        assert type(fresh.vault.clock) is int
        assert fresh.vault.filled == 3
        assert fresh.vault.theta_s == pytest.approx(0.25)

    def test_the_build_from_config_path_yields_a_usable_vault(
        self, populated: Path
    ) -> None:
        """``load`` with no model rebuilds from the recorded config, so the
        vault has to be reconstructed with it and still come back populated."""
        built = load(populated)
        assert built.vault is not None
        assert built.vault.filled == len(SURPRISES)
        assert built.vault.clock == len(SURPRISES)
        assert built.vault.theta_s == THETA_S
        assert built.vault.perm_budget_bytes == PERM_BUDGET
        assert built.vault.retrieve(_keys_of()[0]) is not None

    def test_a_vault_resized_the_same_way_on_both_sides_round_trips(
        self, tmp_path: Path
    ) -> None:
        """``set_entry_cap`` shrinks the tables. Both sides carrying the same
        cap is the supported way to checkpoint such a vault."""
        path = tmp_path / "ck.npz"
        source = vaulted(n_slots=64)
        fill(source.vault, 8)
        save(source, path)

        fresh = vaulted(n_slots=64)
        load(path, fresh)
        assert fresh.vault.filled == 8
        assert fresh.vault.n_slots == 64
        assert np.array_equal(source.vault.keys, fresh.vault.keys)
        assert np.array_equal(source.vault.values, fresh.vault.values)


def _keys_of() -> list[np.ndarray]:
    """The keys written into the module fixture, regenerated from its seed.

    ``fill`` is deterministic, so the fixture's probes can be rebuilt without
    the fixture having to hand them out. A bare vault suffices; the probe
    hypervectors depend only on ``d_value`` and the seed.
    """
    from bhanox.memory.vectorvault import VectorVault

    return fill(VectorVault(d_value=load_config("nano").d_model))


class TestContinuation:
    def test_two_loads_of_one_checkpoint_continue_in_step(
        self, populated: Path
    ) -> None:
        """The strongest statement available without a trainer.

        ``TestPopulatedVault`` already pins that a load reproduces the source
        exactly, so two models that differ only in how many times they were
        loaded agreeing on a continued run means the round trip is transparent
        to whatever runs next.

        This is what the bug broke most sharply: with ``filled`` back at 0 the
        restored vault picked slot 0, overwrote a recovered entry, and stayed a
        write behind forever.
        """
        reference = load(populated)
        restored = vaulted()
        load(populated, restored)

        rng = np.random.default_rng(99)
        key = rng.standard_normal(reference.vault.d_value).astype(np.float32)
        value = rng.standard_normal(reference.vault.d_value).astype(np.float32)
        written = reference.vault.filled

        assert reference.vault.write(key, value, surprise=SOMEONE) is True
        assert restored.vault.write(key, value, surprise=SOMEONE) is True

        assert reference.vault.filled == restored.vault.filled == written + 1
        assert reference.vault.clock == restored.vault.clock
        # The new entry landed in the same slot, not in a recycled low slot.
        assert np.array_equal(
            reference.vault.keys[written], restored.vault.keys[written]
        )
        assert np.array_equal(
            reference.vault.values[written], restored.vault.values[written]
        )
        assert reference.vault.age[written] == restored.vault.age[written]
        for name in ("keys", "values", "salience", "age"):
            assert np.array_equal(
                getattr(reference.vault, name), getattr(restored.vault, name)
            ), name

    def test_repeated_writes_stay_in_step(self, populated: Path) -> None:
        """Not just the first write: ten of them, since a wrong slot only shows
        up once the tables start being reused."""
        reference = load(populated)
        restored = vaulted()
        load(populated, restored)
        rng = np.random.default_rng(7)
        for _ in range(10):
            key = rng.standard_normal(reference.vault.d_value).astype(np.float32)
            value = rng.standard_normal(reference.vault.d_value).astype(np.float32)
            assert reference.vault.write(key, value, surprise=SOMEONE) is True
            assert restored.vault.write(key, value, surprise=SOMEONE) is True
            assert (reference.vault.filled, reference.vault.clock) == (
                restored.vault.filled,
                restored.vault.clock,
            )
        for name in ("keys", "values", "salience", "age"):
            assert np.array_equal(
                getattr(reference.vault, name), getattr(restored.vault, name)
            ), name

    def test_theta_s_still_governs_admission(self, populated: Path) -> None:
        """Restoring ``theta_s`` is not bookkeeping: a write the saved vault
        would have refused must still be refused after the load."""
        fresh = vaulted()
        load(populated, fresh)
        rng = np.random.default_rng(3)
        key = rng.standard_normal(fresh.vault.d_value).astype(np.float32)
        value = rng.standard_normal(fresh.vault.d_value).astype(np.float32)
        before = fresh.vault.filled
        assert fresh.vault.write(key, value, surprise=THETA_S - 0.1) is False
        assert fresh.vault.filled == before
        assert fresh.vault.write(key, value, surprise=THETA_S + 0.1) is True
        assert fresh.vault.filled == before + 1

    def test_perm_budget_still_governs_admission(self, populated: Path) -> None:
        """A restored budget must still cap writes, not merely be recorded."""
        fresh = vaulted()
        load(populated, fresh)
        assert fresh.vault.perm_budget_bytes == PERM_BUDGET
        allowed = fresh.vault.budget_entries()
        assert allowed is not None
        assert allowed * fresh.vault.entry_bytes <= PERM_BUDGET
        assert allowed >= fresh.vault.n_slots  # set_budget did not resize
        fresh.vault.set_budget(fresh.vault.entry_bytes)
        assert fresh.vault.budget_entries() == 1


class TestMetadataSafety:
    def test_caller_extra_survives_alongside_the_vault_block(
        self, tmp_path: Path
    ) -> None:
        """Unrelated metadata is the caller's, and the new block must not
        displace it."""
        path = tmp_path / "ck.npz"
        source = vaulted()
        fill(source.vault, 4)
        extra = {"tokens": 1234, "corpus": "demo", "nested": {"a": [1, 2]}}
        save(source, path, step=2, extra=extra)

        meta = read_meta(path)
        assert meta["extra"] == extra
        assert meta["step"] == 2
        assert meta[RUNTIME_KEY]["vectorvault"]["filled"] == 4

        fresh = vaulted()
        load(path, fresh)
        assert read_meta(path)["extra"] == extra
        assert fresh.vault.filled == 4

    def test_extra_may_not_shadow_the_runtime_block(self, tmp_path: Path) -> None:
        """The existing no-shadowing rule, extended to the new key."""
        with pytest.raises(ValueError, match="shadows"):
            save(vaulted(), tmp_path / "ck.npz", extra={RUNTIME_KEY: {"nope": 1}})

    def test_a_vaultless_checkpoint_still_loads(self, tmp_path: Path) -> None:
        """No vault, no block, no behaviour change: nano and mini checkpoints
        are unaffected, and keep reading the same way."""
        path = tmp_path / "ck.npz"
        source = Bhanox(load_config("nano"))
        save(source, path)
        assert RUNTIME_KEY not in read_meta(path)
        fresh = Bhanox(load_config("nano"))
        load(path, fresh)
        assert np.array_equal(source.output, fresh.output)

    def test_vault_arrays_without_state_are_refused(
        self, populated: Path, tmp_path: Path
    ) -> None:
        """The compatibility policy, pinned.

        A checkpoint written before vault state was recorded holds the arrays
        and nothing else, and restoring it *is* the bug: a populated vault
        reporting ``filled=0``. It is refused instead. Asserting the refusal is
        what stops the policy drifting back into silent acceptance.
        """
        legacy = forge(populated, tmp_path / "legacy.npz", lambda m: m.pop(RUNTIME_KEY))
        fresh = vaulted()
        with pytest.raises(ValueError, match="no VectorVault scalar state"):
            load(legacy, fresh)
        assert fresh.vault.filled == 0, "a refused load must leave the target alone"

    def test_state_without_a_vault_on_the_model_is_refused(
        self, populated: Path
    ) -> None:
        """A checkpoint with vault state cannot go into a vaultless model, which
        would drop the store on the floor."""
        with pytest.raises(ValueError, match="no vault"):
            load(populated, Bhanox(load_config("nano")))

    def test_a_vault_resized_on_one_side_only_is_refused(self, tmp_path: Path) -> None:
        """``set_budget``/``set_entry_cap`` move the table size. A 64-slot
        checkpoint cannot be restored into a 1024-slot vault, and the error says
        which knob caused it instead of surfacing as a bare shape mismatch."""
        path = tmp_path / "ck.npz"
        source = vaulted(n_slots=64)
        fill(source.vault, 4)
        save(source, path)
        with pytest.raises(ValueError, match="64-slot"):
            load(path, vaulted())

    @pytest.mark.parametrize("field", ck.VAULT_STATE)
    def test_a_missing_field_is_refused(
        self, populated: Path, tmp_path: Path, field: str
    ) -> None:
        """Each scalar in turn. A partially restored vault is the failure this
        whole change exists to prevent, so an incomplete block is not a degraded
        load -- it is a refusal."""
        path = forge(
            populated,
            tmp_path / f"missing_{field}.npz",
            lambda m: m[RUNTIME_KEY]["vectorvault"].pop(field),
        )
        with pytest.raises(ValueError, match="missing"):
            load(path, vaulted())

    def test_a_non_object_block_is_refused(
        self, populated: Path, tmp_path: Path
    ) -> None:
        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"] = [1, 2, 3]

        path = forge(populated, tmp_path / "not_object.npz", mutate)
        with pytest.raises(ValueError, match="must be a JSON object"):
            load(path, vaulted())

    def test_an_unknown_block_version_is_refused(
        self, populated: Path, tmp_path: Path
    ) -> None:
        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"]["version"] = RUNTIME_VERSION + 99

        path = forge(populated, tmp_path / "bad_version.npz", mutate)
        with pytest.raises(ValueError, match="version"):
            load(path, vaulted())

    def test_unknown_fields_are_refused(self, populated: Path, tmp_path: Path) -> None:
        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"]["mystery"] = 1

        path = forge(populated, tmp_path / "unknown.npz", mutate)
        with pytest.raises(ValueError, match="unknown fields"):
            load(path, vaulted())

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("filled", "12", "must be a JSON integer"),
            ("filled", 12.0, "must be a JSON integer"),
            ("filled", True, "must be a JSON integer"),
            ("clock", -1, "clock=-1 is negative"),
            ("filled", 99_999, "outside"),
            ("theta_s", "low", "must be a JSON number"),
            ("theta_s", True, "must be a JSON number"),
            ("perm_budget_bytes", "big", "integer or null"),
            ("perm_budget_bytes", True, "integer or null"),
            ("perm_budget_bytes", -5, "perm_budget_bytes=-5 is negative"),
        ],
    )
    def test_malformed_values_are_refused_not_coerced(
        self,
        populated: Path,
        tmp_path: Path,
        field: str,
        value: Any,
        match: str,
    ) -> None:
        """Refused, never coerced. ``"12"`` becoming ``12`` is how a resume ends
        up half-restored, and a bool slipping through as an int is the same
        class of quiet wrongness."""
        path = forge(
            populated,
            tmp_path / f"bad_{field}_{type(value).__name__}.npz",
            lambda m: m[RUNTIME_KEY]["vectorvault"].update({field: value}),
        )
        with pytest.raises(ValueError, match=match):
            load(path, vaulted())

    def test_a_non_finite_theta_is_refused(
        self, populated: Path, tmp_path: Path
    ) -> None:
        """``NaN`` makes every ``surprise < theta_s`` comparison false, so every
        write would be admitted forever."""

        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"]["theta_s"] = float("nan")

        path = forge(populated, tmp_path / "nan_theta.npz", mutate)
        with pytest.raises(ValueError, match="not finite"):
            load(path, vaulted())

    def test_a_clock_behind_the_stored_ages_is_refused(
        self, populated: Path, tmp_path: Path
    ) -> None:
        """The arrays and the scalars must agree. A ``clock`` older than a
        stored entry's ``age`` means the block and the tables came from different
        places, and applying it would corrupt recency-based eviction with
        nothing failing."""

        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"]["clock"] = 1

        path = forge(populated, tmp_path / "stale_clock.npz", mutate)
        with pytest.raises(ValueError, match="trails a stored age"):
            load(path, vaulted())

    def test_a_refused_load_leaves_the_target_untouched(
        self, populated: Path, tmp_path: Path
    ) -> None:
        """The metadata is checked before any array is written, so a refusal
        cannot leave a half-overwritten model behind."""

        def mutate(meta: dict[str, Any]) -> None:
            meta[RUNTIME_KEY]["vectorvault"]["clock"] = 1

        path = forge(populated, tmp_path / "refused.npz", mutate)
        target = vaulted()
        # Move every stored array off its initial value, so a load that wrote
        # some arrays before failing would show up. Cast through the array's own
        # dtype: the inventory spans float32 and uint8.
        rng = np.random.default_rng(0)
        for _, arr in ck.inventory(target):
            arr += (rng.standard_normal(arr.shape) * 0.5).astype(arr.dtype)
        before = {name: arr.copy() for name, arr in ck.inventory(target)}
        with pytest.raises(ValueError):
            load(path, target)
        for name, arr in ck.inventory(target):
            assert np.array_equal(arr, before[name]), name
        assert target.vault.filled == 0
