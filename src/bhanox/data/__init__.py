"""Data loading and tokenisation helpers. NumPy only; no corpus is bundled.

The corpus itself is deliberately absent. Corpus source and license are the
project owner's decision, not this package's, so nothing here downloads, reads
from disk, or embeds a dataset. What is here is the NumPy-only, in-memory
foundation the rest of that decision will sit on:

- :mod:`bhanox.data.order` -- deterministic document ordering as a pure
  function of ``(seed, epoch, name)``, with no global RNG touched.
- :mod:`bhanox.data.examples` -- supervised next-byte examples over an
  in-memory collection of byte documents, with document and byte-offset
  progress.

The input side of the tokenisation contract belongs to
:func:`bhanox.frontend.hashbind.encode_bytes` and is reused unchanged, so a
4-gram here means exactly what it means in :mod:`bhanox.generate`. There is no
tokenizer, no vocabulary and no UNK token: text is bytes, and a context is a
byte 4-gram.

Chunking is by *emitted input positions*, not by raw bytes. A chunk owns a
disjoint run of contexts and reads four look-ahead bytes past them for its
final target, so every supervised example of a document is emitted exactly once
in order and no example is lost at a chunk edge. A chunk is a scheduling seam,
not a recurrent-state boundary: a future trainer should carry model state across
the chunks of one document and reset at document boundaries.

No training loop, loss, optimizer, scheduler or checkpoint integration lives
here, and per law C9 nothing in this package may import torch. The first resume
policy is document-boundary-only; the offsets exposed by
:class:`~bhanox.data.examples.Chunk` address progress within a document but do
not imply mid-document resume.
"""

from __future__ import annotations

from bhanox.data.examples import (
    BYTE_GRAM_N,
    LOOK_AHEAD_BYTES,
    MIN_USEFUL_BYTES,
    Chunk,
    DocumentSummary,
    ShortDocument,
    describe_documents,
    document_chunks,
    examples_for,
    iter_examples,
    targets_for,
)
from bhanox.data.order import DocumentOrder, document_order

__all__ = [
    "BYTE_GRAM_N",
    "LOOK_AHEAD_BYTES",
    "MIN_USEFUL_BYTES",
    "Chunk",
    "DocumentOrder",
    "DocumentSummary",
    "ShortDocument",
    "describe_documents",
    "document_chunks",
    "document_order",
    "examples_for",
    "iter_examples",
    "targets_for",
]
