"""The data pipeline's contract: examples, boundaries, offsets, determinism.

Four things can be silently wrong in a next-byte pipeline, and none of them
raises:

1. **The off-by-one.** A document of ``L`` bytes has ``L - 3`` 4-grams but only
   ``L - 4`` targets. Pairing the whole encoder output with the targets would
   invent a target for the final context.
2. **Cross-document grams.** Concatenating a corpus before encoding forms 4-grams
   across document seams that never existed in any document. They are valid ids,
   so nothing fails -- the model just trains on a fiction.
3. **Offsets.** Chunking that forgets which document a span came from, or which
   byte it started at, produces a stream that cannot be resumed or reported.
4. **Nondeterminism.** An order that reads a global RNG is unreproducible, so a
   resumed run silently sees documents in a different sequence.

The 4-gram encoding itself is not re-tested here; ``tests/test_gram_contract.py``
owns it, and this module reuses :func:`encode_bytes` unchanged. What is pinned
here is the supervised half layered on top of it.
"""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.data import (
    BYTE_GRAM_N,
    LOOK_AHEAD_BYTES,
    MIN_USEFUL_BYTES,
    describe_documents,
    document_chunks,
    document_order,
    examples_for,
    iter_examples,
    targets_for,
)
from bhanox.frontend.hashbind import encode_bytes
from bhanox.model import Bhanox

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_PKG = SRC / "bhanox" / "data"


def ungram(gram_id: int) -> bytes:
    """Inverse of :func:`encode_bytes` for a single id."""
    return int(gram_id).to_bytes(BYTE_GRAM_N, "big")


def np_global_state() -> tuple:
    """A comparable snapshot of the global numpy RNG."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return kind, keys.tobytes(), pos, has_gauss, cached


class TestKnownExamples:
    def test_abcdef_gives_the_documented_pairs(self) -> None:
        """ "abcdef" is the worked example the data contract is stated in:
        abcd -> e and bcde -> f.
        """
        inputs, targets = examples_for(b"abcdef")
        assert [ungram(g) for g in inputs] == [b"abcd", b"bcde"]
        assert targets.tolist() == [ord("e"), ord("f")]

    def test_a_pair_count_is_n_minus_four(self) -> None:
        for length in range(MIN_USEFUL_BYTES, 40):
            raw = bytes(range(65, 65 + length))
            inputs, targets = examples_for(raw)
            assert len(inputs) == len(targets) == length - 4

    def test_the_final_four_byte_window_is_not_an_example(self) -> None:
        """ "abcde" has two 4-grams and one target. "bcde" is the final window and
        has no following byte, so it must not become an input.
        """
        raw = b"abcde"
        assert len(encode_bytes(raw)) == 2
        inputs, targets = examples_for(raw)
        assert len(inputs) == 1
        assert ungram(inputs[0]) == b"abcd"
        assert b"bcde" not in [ungram(g) for g in inputs]
        assert targets.tolist() == [ord("e")]

    def test_targets_are_the_shift_by_four(self) -> None:
        raw = b"the quick brown fox"
        inputs, targets = examples_for(raw)
        assert targets.tolist() == list(raw[BYTE_GRAM_N:])
        for i in range(len(inputs)):
            assert targets[i] == raw[i + BYTE_GRAM_N]
            assert ungram(inputs[i]) == raw[i : i + BYTE_GRAM_N]

    def test_targets_for_is_the_public_half_of_the_pinned_helper(self) -> None:
        """``tests/test_gram_contract.py`` pins ``targets_for`` locally and says
        it has to become public API. It must not have drifted.
        """
        for raw in (b"abcde", b"hello world", b"\x00\xff\x80abc"):
            assert targets_for(raw).tolist() == list(raw[BYTE_GRAM_N:])

    def test_never_more_inputs_than_the_encoder_produced(self) -> None:
        for raw in (b"", b"a", b"abc", b"abcd", b"abcde", b"abcdef"):
            inputs, targets = examples_for(raw)
            assert len(inputs) <= len(encode_bytes(raw, n=BYTE_GRAM_N))
            assert len(inputs) == len(targets)


class TestDocumentsDoNotJoin:
    def test_no_gram_is_formed_across_a_seam(self) -> None:
        """ "abc" + "defg" joined would give three examples, including the gram
        "cdef" that spans the seam. Encoded per document, both are too short and
        the collection yields nothing.
        """
        documents = [b"abc", b"defg"]
        chunks = list(
            iter_examples(documents, max_examples=64, seed=0, epoch=0, name="t")
        )
        assert chunks == []
        joined_inputs, joined_targets = examples_for(b"abc" + b"defg")
        assert len(joined_inputs) == 3
        assert 0x63646566 in joined_inputs  # "cdef", the seam gram
        assert joined_targets.tolist() == [ord("e"), ord("f"), ord("g")]

    def test_documents_are_encoded_independently(self) -> None:
        documents = [b"abcdef", b"ghijkl"]
        chunks = list(
            iter_examples(documents, max_examples=64, seed=0, epoch=0, name="t")
        )
        assert [ungram(g) for c in chunks for g in c.inputs] == [
            b"abcd",
            b"bcde",
            b"ghij",
            b"hijk",
        ]
        assert all(b"efgh" not in [ungram(g) for g in c.inputs] for c in chunks)

    def test_the_summary_does_not_count_joined_examples(self) -> None:
        documents = [b"abc", b"defg"]
        summary = describe_documents(documents, max_examples=64)
        assert summary.total_examples == 0
        assert summary.n_short == 2


class TestDeterministicOrder:
    def test_the_same_seed_and_epoch_give_the_same_order(self) -> None:
        a = document_order(16, seed=7, epoch=3, name="corpus")
        b = document_order(16, seed=7, epoch=3, name="corpus")
        assert a.order == b.order

    def test_a_different_epoch_gives_a_different_order(self) -> None:
        orders = {
            document_order(12, seed=7, epoch=epoch, name="corpus").order
            for epoch in range(4)
        }
        assert len(orders) > 1

    def test_a_different_seed_gives_a_different_order(self) -> None:
        orders = {
            document_order(12, seed=seed, epoch=0, name="corpus").order
            for seed in range(4)
        }
        assert len(orders) > 1

    def test_a_different_name_gives_a_different_order(self) -> None:
        a = document_order(12, seed=1, epoch=0, name="one")
        b = document_order(12, seed=1, epoch=0, name="two")
        assert a.order != b.order

    def test_the_order_is_a_permutation(self) -> None:
        order = document_order(11, seed=3, epoch=1, name="corpus")
        assert sorted(order.order) == list(range(11))
        assert len(order) == 11

    def test_an_empty_collection_orders_to_nothing(self) -> None:
        order = document_order(0, seed=0, epoch=0, name="corpus")
        assert order.order == ()
        assert len(order) == 0

    def test_position_and_document_index_round_trip(self) -> None:
        order = document_order(9, seed=2, epoch=5, name="corpus")
        for position, doc_index in enumerate(order.order):
            assert order.position_of(doc_index) == position
            assert order.doc_index_at(position) == doc_index

    def test_the_order_object_carries_the_progress_identity(self) -> None:
        order = document_order(4, seed=11, epoch=2, name="corpus")
        assert (order.seed, order.epoch, order.name) == (11, 2, "corpus")

    def test_a_precomputed_order_is_reused_as_is(self) -> None:
        documents = [b"abcdef", b"ghijkl", b"mnopqr"]
        order = document_order(3, seed=5, epoch=1, name="corpus")
        reused = list(iter_examples(documents, max_examples=64, order=order))
        fresh = list(
            iter_examples(documents, max_examples=64, seed=5, epoch=1, name="corpus")
        )
        assert [c.doc_index for c in reused] == [c.doc_index for c in fresh]
        assert [c.doc_index for c in reused] == list(order.order)


class TestShortDocuments:
    def test_the_minimum_usable_document_is_five_bytes(self) -> None:
        assert MIN_USEFUL_BYTES == BYTE_GRAM_N + 1 == 5
        inputs, targets = examples_for(b"abcd")
        assert len(inputs) == len(targets) == 0
        inputs, targets = examples_for(b"abcde")
        assert len(inputs) == len(targets) == 1

    @pytest.mark.parametrize("raw", [b"", b"a", b"ab", b"abc", b"abcd"])
    def test_short_documents_yield_nothing_and_are_not_padded(self, raw: bytes) -> None:
        inputs, targets = examples_for(raw)
        assert inputs.shape == (0,)
        assert targets.shape == (0,)

    def test_a_four_byte_document_is_not_padded_into_an_example(self) -> None:
        """Four bytes is a complete context with no following byte. Padding it
        would invent a target, which is the failure this policy exists to stop.
        """
        documents = [b"abcd"]
        assert list(iter_examples(documents, max_examples=64, seed=0, name="t")) == []
        summary = describe_documents(documents, max_examples=64)
        assert summary.total_examples == 0
        assert summary.n_short == 1

    def test_the_summary_reports_every_short_document(self) -> None:
        documents = [b"", b"a", b"abcde", b"xy", b"abcdefgh"]
        summary = describe_documents(documents, max_examples=64)
        assert [(d.index, d.n_bytes) for d in summary.short_documents] == [
            (0, 0),
            (1, 1),
            (3, 2),
        ]
        assert summary.total_bytes == 0 + 1 + 5 + 2 + 8
        assert summary.total_examples == 1 + 4

    def test_text_is_encoded_as_utf8_like_the_encoder(self) -> None:
        text = "héllo"
        from_text = examples_for(text)
        from_bytes = examples_for(text.encode("utf-8"))
        assert np.array_equal(from_text[0], from_bytes[0])
        assert np.array_equal(from_text[1], from_bytes[1])
        assert targets_for(text).tolist() == list(text.encode("utf-8")[BYTE_GRAM_N:])

    def test_a_negative_count_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            document_order(-1, seed=0, epoch=0, name="t")


class TestProgressAndOffsets:
    def test_a_chunk_carries_its_document_index_and_span(self) -> None:
        documents = [b"abcdefgh", b"ijklmnop"]
        chunks = list(
            iter_examples(documents, max_examples=64, seed=0, epoch=0, name="t")
        )
        for chunk in chunks:
            assert chunk.doc_index in (0, 1)
            assert 0 <= chunk.byte_start < chunk.byte_end
            assert chunk.byte_end - chunk.byte_start == len(documents[chunk.doc_index])
            assert len(chunk) == chunk.byte_end - chunk.byte_start - BYTE_GRAM_N

    def test_every_target_lands_on_the_byte_its_offset_names(self) -> None:
        documents = [b"the quick brown fox jumps"]
        for chunk in iter_examples(
            documents, max_examples=8, seed=0, epoch=0, name="t"
        ):
            raw = documents[chunk.doc_index]
            assert chunk.first_target_offset == chunk.byte_start + BYTE_GRAM_N
            for position, target in enumerate(chunk.targets):
                assert int(target) == raw[chunk.target_offset_at(position)]

    def test_every_input_is_the_encoder_at_its_own_offset(self) -> None:
        """The strongest offset check: rebuilding each id straight from
        ``encode_bytes`` at the recorded byte offset.
        """
        documents = [b"abcdefghijklmnopqrstuvwxyz"]
        for chunk in iter_examples(
            documents, max_examples=7, seed=0, epoch=0, name="t"
        ):
            raw = documents[chunk.doc_index]
            for position, gram in enumerate(chunk.inputs):
                start = chunk.byte_start + position
                assert int(gram) == int(
                    encode_bytes(raw[start : start + BYTE_GRAM_N])[0]
                )

    def test_a_target_at_a_chunk_edge_is_not_invented(self) -> None:
        documents = [b"abcdefghij"]
        for chunk in iter_examples(
            documents, max_examples=5, seed=0, epoch=0, name="t"
        ):
            raw = documents[chunk.doc_index]
            assert chunk.targets.tolist() == list(
                raw[chunk.first_target_offset : chunk.byte_end]
            )
            assert chunk.target_offset_at(len(chunk)) == chunk.byte_end


def _document(n_bytes: int, *, seed: int = 0) -> bytes:
    """Deterministic bytes with no short period.

    The generator must not be a modular formula: a periodic byte string repeats
    4-gram *values*, which is valid data but makes any "no duplicate ids"
    assertion fail for the wrong reason. Sampling from a seeded RNG keeps the
    test reproducible and the ids distinct, so an id collision means a real bug.
    """
    return random.Random(seed).randbytes(n_bytes)


class TestChunkBoundaries:
    """A chunk limit counts emitted input positions, not raw bytes."""

    def test_a_long_document_is_split_into_runs_of_contexts(self) -> None:
        raw = bytes(range(97, 97 + 15))
        chunks = list(document_chunks(0, raw, max_examples=5))
        assert [c.byte_start for c in chunks] == [0, 5, 10]
        assert [c.byte_end for c in chunks] == [9, 14, 15]
        assert [len(c) for c in chunks] == [5, 5, 1]

    def test_a_chunk_boundary_is_not_a_document_boundary(self) -> None:
        """Every chunk of one document keeps the same index, so a chunk is an
        addressing convenience, not a document.
        """
        raw = bytes(range(97, 97 + 15))
        chunks = list(document_chunks(3, raw, max_examples=5))
        assert {c.doc_index for c in chunks} == {3}

    def test_no_example_is_dropped_at_a_chunk_edge(self) -> None:
        raw = bytes(range(97, 97 + 20))
        chunks = list(document_chunks(0, raw, max_examples=5))
        whole, whole_targets = examples_for(raw)
        joined = [g for c in chunks for g in c.inputs]
        joined_targets = [t for c in chunks for t in c.targets]
        assert len(whole) == 16
        assert joined == list(whole)
        assert joined_targets == list(whole_targets)

    def test_the_loss_counter_is_zero_under_this_chunking(self) -> None:
        raw = bytes(range(97, 97 + 20))
        summary = describe_documents([raw], max_examples=5)
        chunks = list(document_chunks(0, raw, max_examples=5))
        assert summary.boundary_examples_dropped == 0
        assert summary.total_examples == sum(len(c) for c in chunks) == 16
        assert summary.total_examples == len(examples_for(raw)[0])

    def test_no_loss_when_the_document_fits_in_one_chunk(self) -> None:
        raw = bytes(range(97, 97 + 20))
        summary = describe_documents([raw], max_examples=64)
        assert summary.boundary_examples_dropped == 0
        assert summary.total_examples == len(examples_for(raw)[0])

    def test_a_chunk_limit_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            list(document_chunks(0, b"abcdefgh", max_examples=0))
        with pytest.raises(ValueError):
            describe_documents([b"abcdefgh"], max_examples=0)

    def test_a_chunk_limit_of_one_is_accepted(self) -> None:
        chunks = list(document_chunks(0, b"abcdefgh", max_examples=1))
        assert [len(c) for c in chunks] == [1, 1, 1, 1]

    def test_the_summary_agrees_with_iteration_at_every_chunk_size(self) -> None:
        for length in range(0, 60):
            raw = bytes((i * 3 + 1) % 256 for i in range(length))
            for max_examples in (1, 2, 5, 16, 64):
                emitted = sum(
                    len(c) for c in document_chunks(0, raw, max_examples=max_examples)
                )
                summary = describe_documents([raw], max_examples=max_examples)
                assert summary.total_examples == emitted
                assert summary.boundary_examples_dropped == 0
                assert emitted == max(length - BYTE_GRAM_N, 0)


class TestLookAhead:
    """Four look-ahead bytes past the contexts a chunk owns."""

    def test_seventeen_bytes_emits_all_thirteen_examples_once_each(self) -> None:
        raw = bytes(range(97, 97 + 17))
        chunks = list(document_chunks(0, raw, max_examples=5))
        assert [c.byte_start for c in chunks] == [0, 5, 10]
        assert [c.byte_end for c in chunks] == [9, 14, 17]
        assert [c.owned_end for c in chunks] == [5, 10, 13]
        assert [len(c) for c in chunks] == [5, 5, 3]
        assert sum(len(c) for c in chunks) == 13
        offsets = [c.byte_start + p for c in chunks for p in range(len(c))]
        assert offsets == list(range(13))
        assert len(set(offsets)) == 13

    def test_eight_thousand_bytes_at_the_context_limit(self) -> None:
        raw = bytes((i * 7 + 3) % 256 for i in range(8192))
        chunks = list(document_chunks(0, raw, max_examples=4096))
        assert [c.byte_start for c in chunks] == [0, 4096]
        assert [c.byte_end for c in chunks] == [4100, 8192]
        assert [len(c) for c in chunks] == [4096, 4092]
        assert sum(len(c) for c in chunks) == 8188
        assert describe_documents([raw], max_examples=4096).total_examples == 8188

    def test_every_chunk_reads_exactly_four_look_ahead_bytes(self) -> None:
        for length in (5, 6, 17, 20, 100, 8192):
            raw = _document(length)
            for max_examples in (1, 5, 16, 4096):
                for chunk in document_chunks(0, raw, max_examples=max_examples):
                    assert chunk.look_ahead_bytes == LOOK_AHEAD_BYTES
                    assert chunk.byte_end == chunk.owned_end + LOOK_AHEAD_BYTES
                    assert chunk.owned_end - chunk.byte_start == len(chunk)

    def test_look_ahead_bytes_belong_to_the_document_not_the_chunk(self) -> None:
        """The final target of a chunk sits in its look-ahead, so the bytes a
        chunk reads past ``owned_end`` must be real document bytes.
        """
        raw = _document(20, seed=1)
        for chunk in document_chunks(0, raw, max_examples=5):
            for position in range(len(chunk)):
                offset = chunk.target_offset_at(position)
                assert chunk.byte_start <= offset < chunk.byte_end
                assert offset < len(raw)
                assert int(chunk.targets[position]) == raw[offset]

    def test_concatenated_chunks_match_unchunked_processing(self) -> None:
        for length in (5, 17, 20, 100, 257, 1000):
            raw = _document(length, seed=length)
            whole_inputs, whole_targets = examples_for(raw)
            for max_examples in (1, 2, 7, 64, 4096):
                chunks = list(document_chunks(0, raw, max_examples=max_examples))
                assert [g for c in chunks for g in c.inputs] == list(whole_inputs)
                assert [t for c in chunks for t in c.targets] == list(whole_targets)

    def test_no_input_context_id_is_duplicated_across_chunks(self) -> None:
        """Every owned context is emitted once, and these ids are distinct, so no
        id repeats.

        The id-set assertion is only meaningful because ``_document`` is not
        periodic. On periodic data equal ids are correct -- the contract is that
        each context *offset* is emitted once, which
        :meth:`test_no_context_offset_is_missing` pins.
        """
        raw = _document(300, seed=2)
        chunks = list(document_chunks(0, raw, max_examples=13))
        ids = [g for c in chunks for g in c.inputs]
        assert len(ids) == 296
        assert len(set(ids)) == 296

    def test_no_context_offset_is_missing(self) -> None:
        raw = _document(300, seed=3)
        chunks = list(document_chunks(0, raw, max_examples=13))
        seen = [c.byte_start + p for c in chunks for p in range(len(c))]
        assert seen == list(range(296))

    def test_owned_offsets_are_disjoint_across_chunks(self) -> None:
        raw = _document(1000, seed=4)
        for max_examples in (1, 3, 16, 64, 997):
            owned: list[int] = []
            for chunk in document_chunks(0, raw, max_examples=max_examples):
                owned.extend(chunk.byte_start + p for p in range(len(chunk)))
            assert owned == sorted(owned)
            assert len(set(owned)) == len(owned)
            assert owned == list(range(len(raw) - BYTE_GRAM_N))

    def test_a_context_limit_fits_the_configured_model_window(self) -> None:
        max_context = load_config("nano").max_context
        raw = _document(20000, seed=5)
        chunks = list(document_chunks(0, raw, max_examples=max_context))
        assert max(len(c) for c in chunks) <= max_context
        assert sum(len(c) for c in chunks) == 20000 - BYTE_GRAM_N
        assert describe_documents([raw], max_examples=max_context).total_examples == (
            20000 - BYTE_GRAM_N
        )

    def test_short_documents_still_yield_nothing(self) -> None:
        for raw in (b"", b"a", b"abcd"):
            assert list(document_chunks(0, raw, max_examples=1)) == []
        chunks = list(document_chunks(0, b"abcde", max_examples=1))
        assert [(c.byte_start, c.byte_end, len(c)) for c in chunks] == [(0, 5, 1)]


class TestCarryStateMatchesContinuousPass:
    """A chunk boundary is a scheduling seam, not a state boundary.

    Forward only. No training, no gradients, no optimiser: this asserts that
    feeding a document's chunks in order, carrying model state, reproduces one
    continuous pass over the same document. The carry is the whole point -- no
    ``model.reset()`` between chunks.
    """

    def test_carrying_state_across_chunks_matches_one_continuous_pass(self) -> None:
        raw = _document(64, seed=6)
        chunks = list(document_chunks(0, raw, max_examples=16))
        assert [len(c) for c in chunks] == [16, 16, 16, 12]
        chunk_ids = [np.array(c.inputs) for c in chunks]

        model = Bhanox(load_config("nano"))
        model.reset()
        continuous = model.forward(np.concatenate(chunk_ids)[None, :])

        model.reset()
        last = None
        for part in chunk_ids:
            last = model.forward(part[None, :])
        assert last is not None
        assert np.array_equal(last[0, -1], continuous[0, -1])

    def test_resetting_between_chunks_does_not_match(self) -> None:
        """The contrast that makes the previous test meaningful: a reset at every
        chunk boundary is a different computation.
        """
        raw = _document(64, seed=6)
        chunks = list(document_chunks(0, raw, max_examples=16))
        chunk_ids = [np.array(c.inputs) for c in chunks]

        model = Bhanox(load_config("nano"))
        model.reset()
        continuous = model.forward(np.concatenate(chunk_ids)[None, :])

        model.reset()
        last = None
        for part in chunk_ids:
            last = model.forward(part[None, :])
            model.reset()
        assert last is not None
        assert not np.array_equal(last[0, -1], continuous[0, -1])


class TestNoTorchAndNoGlobalRng:
    def test_no_data_module_references_torch(self) -> None:
        for path in sorted(DATA_PKG.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(a.name.split(".")[0] != "torch" for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] != "torch"

    def test_importing_the_data_package_does_not_import_torch(self) -> None:
        """Static checks cannot see a transitive import, so import the package in
        a clean process and look at ``sys.modules``.
        """
        code = (
            "import sys, bhanox.data;"
            "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'torch');"
            "assert not leaked, leaked"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        assert result.returncode == 0, result.stderr

    def test_ordering_does_not_touch_either_global_rng(self) -> None:
        py_state = random.getstate()
        np_state = np_global_state()
        document_order(20, seed=1, epoch=1, name="corpus")
        assert random.getstate() == py_state
        assert np_global_state() == np_state

    def test_building_examples_does_not_touch_either_global_rng(self) -> None:
        py_state = random.getstate()
        np_state = np_global_state()
        documents = [b"abcdefgh", b"ij", b"klmnopqrstuv"]
        list(iter_examples(documents, max_examples=6, seed=2, epoch=0, name="corpus"))
        describe_documents(documents, max_examples=6)
        assert random.getstate() == py_state
        assert np_global_state() == np_state

    def test_a_draw_is_unaffected_by_intervening_global_randomness(self) -> None:
        first = document_order(20, seed=4, epoch=0, name="corpus").order
        random.random()
        np.random.random()
        second = document_order(20, seed=4, epoch=0, name="corpus").order
        assert first == second
