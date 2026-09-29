"""Supervised next-byte examples from in-memory byte documents.

:func:`~bhanox.frontend.hashbind.encode_bytes` produces the *input* stream for the
frozen contract: a document becomes a stream of packed big-endian byte 4-gram
ids. It deliberately has no targets, because the loss and the windowing
decision were not yet fixed. This module supplies the target half and the
offsets, and changes nothing about the encoder.

The contract, unchanged from :mod:`bhanox.generate` and pinned by
``tests/test_gram_contract.py``:

- one model input is one byte 4-gram, one output class is one byte;
- the gram at byte offset ``i`` predicts the byte at ``i + 4``;
- a document of ``L`` bytes therefore has ``L - 4 + 1`` gram ids but only
  ``L - 4`` targets, and the final 4-gram has no following byte inside its own
  buffer.

The off-by-one is a property of next-byte prediction, not a bug to paper over.
The last context is dropped, never padded: there is no fake target, and no
vocabulary, tokenizer or UNK token anywhere in this path.

Two boundaries matter, and they are not the same boundary:

- **Document boundaries are real.** A 4-gram is never formed across two
  documents. Encoding per document is what guarantees it; there is no option to
  concatenate.
- **Chunk boundaries are not document boundaries.** Cutting a long document to
  the model's context window produces chunks that are still one document and
  keep their identity and byte offsets, so progress stays addressable.

A chunk owns a disjoint run of *contexts*, not a disjoint run of bytes, and that
distinction is the whole design. The context at offset ``i`` needs the byte at
``i + BYTE_GRAM_N``, which is very often the first byte of the next chunk. So
each chunk reads :data:`LOOK_AHEAD_BYTES` bytes past the contexts it owns, and
only those bytes. Ownership stays a clean partition of
``0 .. L - BYTE_GRAM_N - 1``, so every supervised example is emitted exactly
once in document order, and concatenating a document's chunks reproduces its
unchunked example stream. Nothing is lost at an edge and nothing is repeated.

**A chunk is not a recurrent-state boundary.** The model carries DeltaBank and
PulseGate state across the positions of a forward pass, and there is no
positional encoding, so a chunk boundary is a scheduling seam and nothing more.
A future trainer should carry model state across the chunks of one document and
reset only at document boundaries. That trainer does not exist yet, and this
module does not touch model state.

There is no mid-document resume here. The smallest unit this module addresses is
a document, and a chunk is an addressing convenience within a document, not a
resume point.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bhanox.data.order import DocumentOrder, document_order
from bhanox.frontend.hashbind import BYTE_GRAM_N, encode_bytes

__all__ = [
    "BYTE_GRAM_N",
    "LOOK_AHEAD_BYTES",
    "MIN_USEFUL_BYTES",
    "Chunk",
    "DocumentSummary",
    "ShortDocument",
    "describe_documents",
    "document_chunks",
    "examples_for",
    "iter_examples",
    "targets_for",
]

#: A document needs one 4-byte context *and* one following byte to yield a
#: single example, so five bytes is the shortest document with any use. This is
#: the whole short-document policy: below this, a document yields nothing and is
#: reported, never padded.
MIN_USEFUL_BYTES = BYTE_GRAM_N + 1

#: Bytes a chunk reads past the contexts it owns, so the last owned context's
#: target is inside the same chunk. This is the whole of the right-extension:
#: four bytes, and nothing else.
LOOK_AHEAD_BYTES = BYTE_GRAM_N


def _as_bytes(raw: bytes | str) -> bytes:
    """Normalise to bytes, without copying a ``bytes`` input."""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return bytes(raw)


def targets_for(raw: bytes | str) -> NDArray[np.int64]:
    """The byte that follows each 4-gram in ``raw``.

    This is the target half of the contract that :func:`encode_bytes` leaves
    open. It matches the ``targets_for`` helper pinned in
    ``tests/test_gram_contract.py`` exactly; that test file notes it needs to
    become public API, which is what this is.

    A gram at offset ``i`` covers bytes ``i .. i + 3`` and predicts byte
    ``i + 4``, so the targets are ``raw[BYTE_GRAM_N:]``.

    Args:
        raw: Bytes, or text, which is encoded as UTF-8 exactly as
            :func:`encode_bytes` does.

    Returns:
        ``int64`` array of ``max(0, len(raw) - BYTE_GRAM_N)`` byte values, each
        in ``0..255``. Empty when the document cannot supply a following byte.
    """
    raw = _as_bytes(raw)
    return np.frombuffer(raw, dtype=np.uint8)[BYTE_GRAM_N:].astype(np.int64)


def examples_for(raw: bytes | str) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Supervised ``(inputs, targets)`` pairs for one whole document.

    Args:
        raw: Bytes, or text. Encoded as UTF-8, matching :func:`encode_bytes`.

    Returns:
        ``(inputs, targets)``, two ``int64`` arrays of equal length, at most
        ``len(raw) - BYTE_GRAM_N`` long. ``inputs`` are packed 4-gram ids from
        :func:`encode_bytes`; ``targets`` are the following byte values.
    """
    raw = _as_bytes(raw)
    ids = encode_bytes(raw, n=BYTE_GRAM_N)
    targets = targets_for(raw)
    # ``encode_bytes`` yields len(raw) - BYTE_GRAM_N + 1 ids while there are only
    # len(raw) - BYTE_GRAM_N targets, so the last id is a context with no byte
    # after it. Pairing stops at the shorter of the two instead of padding the
    # target stream: truncating can only ever lose an example, never invent one.
    return ids[: len(targets)], targets


@dataclass(frozen=True, eq=False)
class Chunk:
    """One contiguous run of one document's contexts, with its offsets.

    A chunk *owns* a disjoint run of supervised contexts and *reads* those
    contexts' bytes plus :data:`LOOK_AHEAD_BYTES` more. Those are different
    extents, and the difference is the point of the chunk, so both are explicit:
    :attr:`byte_start` and :attr:`byte_end` bound the bytes actually read, and
    :attr:`owned_end` bounds the contexts this chunk is responsible for. The
    bytes between them are look-ahead -- read to supply a target, owned by
    nobody.

    ``byte_start`` is both the first raw byte read and the document offset of
    the chunk's first owned context. The two coincide because a span never
    begins before the context it owns.

    Identity equality is deliberate: the arrays would make field-wise equality
    ambiguous, and these are stream records rather than values to compare.
    """

    doc_index: int
    byte_start: int
    byte_end: int
    inputs: NDArray[np.int64]
    targets: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.inputs)

    @property
    def owned_end(self) -> int:
        """Document offset one past this chunk's last owned context.

        Owned contexts start at ``byte_start`` and run through ``owned_end - 1``,
        so ``owned_end - byte_start == len(self)``.
        """
        return self.byte_start + len(self.inputs)

    @property
    def look_ahead_bytes(self) -> int:
        """Bytes read past :attr:`owned_end` to supply the final target.

        This is normally :data:`LOOK_AHEAD_BYTES`; the last context of a
        document is owned by whichever chunk reaches it, and its target is still
        inside the document, so no chunk needs less. It exists to be asserted
        rather than computed by every caller.
        """
        return self.byte_end - self.owned_end

    @property
    def first_target_offset(self) -> int:
        """Byte offset of this chunk's first target, ``byte_start + 4``.

        Together with :meth:`target_offset_at`, a caller can turn a position
        within this chunk into a document byte offset.
        """
        return self.byte_start + BYTE_GRAM_N

    def target_offset_at(self, position: int) -> int:
        """Byte offset of the target for ``position`` within this chunk."""
        return self.first_target_offset + position


@dataclass(frozen=True)
class ShortDocument:
    """A document that yielded no example, and why: it is too short."""

    index: int
    n_bytes: int


@dataclass(frozen=True)
class DocumentSummary:
    """Arithmetic accounting for a collection, with no data encoded.

    Every field is computable from document lengths alone, so a caller can size
    a run, or report what was dropped, before touching any bytes.
    """

    n_documents: int
    max_examples: int
    total_bytes: int
    total_examples: int
    short_documents: tuple[ShortDocument, ...]
    boundary_examples_dropped: int

    @property
    def n_short(self) -> int:
        return len(self.short_documents)


def _check_max_examples(max_examples: int) -> int:
    """Validate a chunk size, returning it.

    The limit counts emitted input positions, so the only unusable value is zero
    or below: a chunk permitted to emit no examples would make
    :func:`document_chunks` non-terminating. This is a configuration mistake
    rather than a data property, so it raises.
    """
    if max_examples < 1:
        raise ValueError(f"max_examples must be at least 1, got {max_examples}")
    return max_examples


def document_chunks(
    doc_index: int, raw: bytes | str, *, max_examples: int
) -> Iterator[Chunk]:
    """Split one document into runs of at most ``max_examples`` contexts.

    ``max_examples`` counts emitted 4-gram input positions, not raw bytes. Each
    chunk owns the next ``max_examples`` supervised contexts and reads
    :data:`LOOK_AHEAD_BYTES` bytes past them, which is exactly the room the last
    owned context's target needs.

    Args:
        doc_index: Index of this document in the caller's collection, carried
            through to every chunk so progress stays addressable.
        raw: The document.
        max_examples: Maximum emitted input positions per chunk, normally
            ``config.max_context``.

    Yields:
        A :class:`Chunk` per run of contexts, in document order.

    Note:
        Ownership is a clean partition of ``0 .. len(raw) - BYTE_GRAM_N - 1``, so
        the yielded contexts are disjoint, ascending, and together exactly the
        ``max(len(raw) - BYTE_GRAM_N, 0)`` supervised examples of the document.
        Concatenating a document's chunks reproduces its unchunked example
        stream exactly. Nothing is dropped at an edge and nothing is repeated.

        This is a scheduling seam, not a state boundary. The model carries
        DeltaBank and PulseGate state across the positions of a forward pass, so
        a future trainer should carry that state across the chunks of one
        document and reset only between documents. This module does not touch
        model state.
    """
    max_examples = _check_max_examples(max_examples)
    raw = _as_bytes(raw)
    n_owned = max(len(raw) - BYTE_GRAM_N, 0)
    for start in range(0, n_owned, max_examples):
        owned = min(max_examples, n_owned - start)
        end = min(start + owned + LOOK_AHEAD_BYTES, len(raw))
        inputs, targets = examples_for(raw[start:end])
        yield Chunk(
            doc_index=doc_index,
            byte_start=start,
            byte_end=end,
            inputs=inputs,
            targets=targets,
        )


def iter_examples(
    documents: Sequence[bytes | str],
    *,
    max_examples: int,
    seed: int = 0,
    epoch: int = 0,
    name: str = "",
    order: DocumentOrder | None = None,
) -> Iterator[Chunk]:
    """Stream every example of every document, in this epoch's order.

    Documents are visited in the order from :func:`document_order`, and documents
    too short to yield an example are skipped. Pass ``order`` to reuse a
    previously computed permutation instead of redrawing it.

    Args:
        documents: In-memory documents. No corpus is read, downloaded or
            embedded here; the caller supplies the bytes.
        max_examples: Maximum emitted input positions per chunk, normally
            ``config.max_context``. A chunk reads up to four bytes beyond that,
            as look-ahead for its final target.
        seed: Run seed, forwarded to the ordering.
        epoch: Epoch whose permutation to use, forwarded to the ordering.
        name: Corpus or dataset label, forwarded to the ordering.
        order: An existing :class:`~bhanox.data.order.DocumentOrder` to use
            as-is. If omitted, one is computed from ``seed``/``epoch``/``name``.

    Yields:
        :class:`Chunk` records. The caller's progress coordinate is
        ``(epoch, doc_index, byte_start)``; ``epoch`` comes from the order
        object and the rest from the chunk. A caller that batches chunks should
        carry model state across the chunks of one document and reset at
        document boundaries; this iterator does neither, because it does not
        touch model state.
    """
    if order is None:
        order = document_order(len(documents), seed=seed, epoch=epoch, name=name)
    for doc_index in order.order:
        raw = _as_bytes(documents[doc_index])
        if len(raw) < MIN_USEFUL_BYTES:
            continue
        yield from document_chunks(doc_index, raw, max_examples=max_examples)


def describe_documents(
    documents: Sequence[bytes | str], *, max_examples: int
) -> DocumentSummary:
    """Count examples and short documents from lengths alone.

    The result must agree with :func:`iter_examples`: the same
    ``total_examples`` is the sum of the chunk lengths, and the same
    ``short_documents`` are the documents that yield nothing. Nothing is
    encoded, so this is cheap enough to call for reporting.

    ``boundary_examples_dropped`` is kept as a machine-checkable invariant and
    is always zero. A chunk reads look-ahead rather than cutting contexts short,
    so a document contributes all of its ``len(raw) - BYTE_GRAM_N`` examples at
    any chunk size.
    """
    max_examples = _check_max_examples(max_examples)
    lengths = [len(_as_bytes(doc)) for doc in documents]
    total_examples = 0
    short: list[ShortDocument] = []
    for index, length in enumerate(lengths):
        if length < MIN_USEFUL_BYTES:
            short.append(ShortDocument(index=index, n_bytes=length))
            continue
        total_examples += length - BYTE_GRAM_N
    return DocumentSummary(
        n_documents=len(lengths),
        max_examples=max_examples,
        total_bytes=sum(lengths),
        total_examples=total_examples,
        short_documents=tuple(short),
        boundary_examples_dropped=0,
    )
