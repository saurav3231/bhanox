"""Checkpoint round-trip, atomicity, and versioning.

The property under test throughout is that a checkpoint is either exactly
right or loudly refused. A checkpoint that loads "mostly" is worse than no
checkpoint, because training resumes from it and nobody notices until quality
stalls weeks later.

Every restore test starts by proving the target differs from the source, so a
``load`` that silently did nothing fails instead of passing. Two models built
from the same seed are already identical; comparing them proves nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bhanox
from bhanox.checkpoint import (
    FORMAT,
    META_KEY,
    VERSION,
    _segments,
    inventory,
    load,
    read_meta,
    save,
)
from bhanox.config import load_config
from bhanox.model import Bhanox

IDS = np.arange(1, 24, dtype=np.int64)


def nano() -> Bhanox:
    return Bhanox(load_config("nano"))


def scramble(model: Bhanox, seed: int = 0) -> None:
    """Move every stored weight far from its current value.

    Used to prove a restore did something. Comparing two same-seed models would
    pass even with a no-op ``load``.
    """
    rng = np.random.default_rng(seed)
    for _, arr in inventory(model):
        arr += (rng.standard_normal(arr.shape) * 0.5).astype(arr.dtype)


def worst_diff(a: Bhanox, b: Bhanox) -> float:
    return max(
        float(np.abs(x - y).max())
        for (_, x), (_, y) in zip(inventory(a), inventory(b), strict=True)
    )


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    model = nano()
    for token in IDS:
        model.step(int(token))
    path = tmp_path / "ck.npz"
    save(model, path, step=7)
    return path


class TestInventory:
    def test_covers_every_parameter(self) -> None:
        """The inventory is the whole point: a hand-written field list is where
        a silently dropped weight would come from."""
        names = [n for n, _ in inventory(nano())]
        assert "embedder.table" in names
        assert "output" in names
        assert "deltabanks[0].heads[0].W_k" in names
        assert "deltabanks[3].heads[3].bank_logits" in names
        assert "mixers[0].W1" in names
        assert "gates[0].tau_hi" in names
        assert len(names) == len(set(names)), "duplicate names in the inventory"

    def test_excludes_recurrent_state(self) -> None:
        """State is rebuilt by reset(); saving it would also make a file's
        contents depend on the batch size of the run that wrote it."""
        names = [n for n, _ in inventory(nano())]
        for banned in (".state", ".cached", ".awake", "._quiet", "._has_run", ".loads"):
            assert not any(banned in n for n in names), f"{banned} should not be saved"

    def test_gate_thresholds_are_saved(self) -> None:
        """The gates' learned thresholds are parameters, not runtime state, and
        a checkpoint without them is a different model."""
        names = {n for n, _ in inventory(nano())}
        for i in range(load_config("nano").n_layers):
            assert f"gates[{i}].tau_hi" in names
            assert f"gates[{i}].tau_lo" in names
            assert f"gates[{i}].salience" in names

    def test_order_is_stable(self) -> None:
        """A stable order makes two checkpoints of the same model diffable."""
        assert [n for n, _ in inventory(nano())] == [n for n, _ in inventory(nano())]

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("output", ["output"]),
            ("mixers[2].E", ["mixers", "[2]", "E"]),
            (
                "deltabanks[0].heads[1].W_k",
                ["deltabanks", "[0]", "heads", "[1]", "W_k"],
            ),
        ],
    )
    def test_path_grammar(self, name: str, expected: list[str]) -> None:
        assert _segments(name) == expected

    def test_rejects_unparseable_path(self) -> None:
        with pytest.raises(ValueError, match="unparseable"):
            _segments("not a name[0]")


class TestRoundTrip:
    def test_restores_every_weight_bit_exactly(self, saved: Path) -> None:
        source = nano()
        target = nano()
        scramble(target)
        assert worst_diff(source, target) > 0, "scramble did nothing, test is vacuous"

        load(saved, target)
        assert worst_diff(source, target) == 0.0

    def test_produces_identical_logits(self, saved: Path) -> None:
        source, target = nano(), nano()
        load(saved, target)
        source.reset()
        target.reset()
        assert np.array_equal(source.forward(IDS), target.forward(IDS))

    def test_builds_a_model_from_the_recorded_config(self, saved: Path) -> None:
        """The caller should not have to remember the architecture."""
        source = nano()
        source.reset()
        built = load(saved)
        assert isinstance(built, Bhanox)
        assert built.config.name == source.config.name
        assert np.array_equal(source.forward(IDS), built.forward(IDS))

    def test_restores_runtime_state_to_its_initial_value(self, saved: Path) -> None:
        """Runtime buffers come back as a new model's, not as zeros: an
        untrained PulseGate starts with every channel awake, so all-true is
        correct and all-false would be a different gate."""
        fresh, pristine = nano(), nano()
        load(saved, fresh)
        assert np.array_equal(
            fresh.deltabanks[0].heads[0].state, pristine.deltabanks[0].heads[0].state
        )
        assert np.array_equal(fresh.gates[0].awake, pristine.gates[0].awake)
        assert np.array_equal(fresh.gates[0].cached, pristine.gates[0].cached)

    def test_carrying_the_state_across_a_resume_is_exact(self, saved: Path) -> None:
        """The checkpoint holds weights, not state -- but ``forward`` continues
        from existing state rather than resetting, so a caller that carries the
        recurrent buffers alongside the checkpoint gets a bit-exact resume. This
        is why the state does not have to live in the file, and it is the
        contract the trainer will rely on.
        """
        head, tail = IDS[:10], IDS[10:]
        whole = nano()
        expected = whole.forward(IDS)[:, len(head) :]

        resumed = nano()
        load(saved, resumed)
        for t in head:
            resumed.step(int(t))
        assert np.array_equal(resumed.forward(tail), expected)

    def test_dropping_the_state_changes_the_result(self, saved: Path) -> None:
        """The converse, so the test above cannot pass for the wrong reason: a
        resume that loses the recurrent buffer is a different computation, and
        silently so.
        """
        head, tail = IDS[:10], IDS[10:]
        whole = nano()
        expected = whole.forward(IDS)[:, len(head) :]

        lost = nano()
        load(saved, lost)  # weights restored, recurrent state left empty
        assert not np.allclose(lost.forward(tail), expected)

    def test_does_not_alias_the_saved_arrays(self, saved: Path) -> None:
        """Loading must copy, not rebind: writing to the model after a load has
        to leave the file on disk untouched."""
        first = nano()
        load(saved, first)
        scramble(first, seed=3)

        second = nano()
        load(saved, second)
        assert worst_diff(first, second) > 0, "load rebound arrays instead of copying"


class TestMetadata:
    def test_records_format_version_and_step(self, saved: Path) -> None:
        meta = read_meta(saved)
        assert meta["format"] == FORMAT
        assert meta["version"] == VERSION
        assert meta["step"] == 7

    def test_records_every_shape_and_dtype(self, saved: Path) -> None:
        meta = read_meta(saved)
        for name, arr in inventory(nano()):
            assert meta["shapes"][name] == list(arr.shape)
            assert meta["dtypes"][name] == str(arr.dtype)

    def test_records_the_config(self, saved: Path) -> None:
        assert read_meta(saved)["config"]["name"] == "nano"

    def test_extra_metadata_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.npz"
        save(nano(), path, step=3, extra={"tokens": 1234, "corpus": "demo"})
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data[META_KEY]).decode("utf-8"))
        assert meta["extra"] == {"tokens": 1234, "corpus": "demo"}
        assert meta["step"] == 3

    def test_extra_may_not_shadow_a_tensor(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="shadows"):
            save(nano(), tmp_path / "ck.npz", extra={"output": 1})


class TestRefusal:
    def test_rejects_a_non_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.npz"
        np.savez(path, something=np.zeros(3))
        with pytest.raises(OSError, match="not a Bhanox checkpoint"):
            read_meta(path)

    def test_rejects_an_unknown_version(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.npz"
        save(nano(), path)
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        meta = json.loads(bytes(payload[META_KEY]).decode("utf-8"))
        meta["version"] = VERSION + 99
        payload[META_KEY] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="version"):
            read_meta(path)

    def test_rejects_a_foreign_format(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.npz"
        save(nano(), path)
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        meta = json.loads(bytes(payload[META_KEY]).decode("utf-8"))
        meta["format"] = "something-else"
        payload[META_KEY] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="not a Bhanox checkpoint"):
            read_meta(path)

    def test_rejects_a_shape_mismatch(self, saved: Path) -> None:
        """Loading nano weights into a mini model must fail loudly. Silently
        skipping the mismatched layers would give a half-initialised model."""
        with pytest.raises(ValueError, match="shape"):
            load(saved, Bhanox(load_config("mini")))

    def test_rejects_an_unknown_field(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.npz"
        save(nano(), path)
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        meta = json.loads(bytes(payload[META_KEY]).decode("utf-8"))
        meta["tensors"] = [*meta["tensors"], "no_such_field"]
        payload[META_KEY] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="missing tensors"):
            load(path, nano())


class TestPublicApi:
    def test_from_pretrained_loads_a_checkpoint(self, saved: Path) -> None:
        """``from_pretrained`` was declared as a frozen API hook that raised
        until the M2 checkpoint format existed. It must now return the model
        rather than raise, because a caller written against the frozen surface
        expects a working function."""
        source = nano()
        source.reset()
        restored = bhanox.from_pretrained(str(saved))
        assert isinstance(restored, Bhanox)
        assert np.array_equal(source.forward(IDS), restored.forward(IDS))

    def test_from_pretrained_refuses_a_non_checkpoint(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.npz"
        np.savez(junk, something=np.zeros(3))
        with pytest.raises(OSError):
            bhanox.from_pretrained(str(junk))


class TestAtomicity:
    def test_leaves_no_temp_file(self, saved: Path, tmp_path: Path) -> None:
        assert list(tmp_path.iterdir()) == [saved]

    def test_a_failed_write_keeps_the_previous_checkpoint(
        self, saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole reason for the temp-file dance: a crash mid-save must not
        destroy the checkpoint that was already there."""
        before = saved.read_bytes()

        def explode(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("bhanox.checkpoint.np.savez", explode)
        with pytest.raises(OSError, match="disk full"):
            save(nano(), saved, step=8)

        assert saved.read_bytes() == before, "failed save damaged the old checkpoint"
        assert [p.name for p in tmp_path.iterdir()] == [
            saved.name
        ], "temp file left behind"
        assert read_meta(saved)["step"] == 7

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "ck.npz"
        save(nano(), path)
        assert path.exists()
        assert load(path).config.name == "nano"
