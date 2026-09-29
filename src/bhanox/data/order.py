"""Deterministic document ordering, one stream per (seed, epoch).

Ordering has to be a pure function of the seed and the epoch, or a resumed run
cannot reproduce the sequence of documents it already trained on. Three
properties follow, and all three are load-bearing:

- **Replayable.** ``(seed, epoch, name)`` gives the same order in any process,
  on any machine, forever, because the stream is keyed by BLAKE2b rather than
  by :func:`hash`. A run that starts twice from the same seed sees the same
  documents in the same order.
- **Per-epoch.** The epoch is the stream ``index``, so each epoch gets its own
  permutation from the same seed without any epoch-to-epoch chaining. There is
  no shuffle state to carry, and therefore none to checkpoint: the position
  inside an epoch is the document's rank in :attr:`DocumentOrder.order`.
- **Isolated.** A fresh :class:`numpy.random.Generator` is returned per call, so
  nothing here reads or mutates the global ``random`` or ``numpy.random`` state.
  Two calls with the same arguments are independent and equally reproducible.

The order is over *all* documents, including any too short to yield an example.
Document identity and rank stay well defined independently of the short-document
policy in :mod:`bhanox.data.examples`, so a corpus edit does not silently
reassign ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

from bhanox.seeding import init_rng

__all__ = ["DocumentOrder", "document_order"]

#: Stream tag, kept distinct from every weight-init site so a shuffle can never
#: draw from, or be confused with, a projection's initialisation stream.
SHUFFLE_SITE = "shuffle"


@dataclass(frozen=True)
class DocumentOrder:
    """The document permutation for one (seed, epoch).

    Equality is by identity, not field-wise. The order is a plain tuple of
    ints, so comparing two orders field-wise would be correct, but the type is
    meant to be compared only for identity in any case.
    """

    seed: int
    epoch: int
    name: str
    order: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self):
        return iter(self.order)

    def position_of(self, doc_index: int) -> int:
        """Rank of ``doc_index`` within this epoch's order.

        This is the progress coordinate inside an epoch, alongside the epoch
        itself.
        """
        return self.order.index(doc_index)

    def doc_index_at(self, position: int) -> int:
        """Document index visited at ``position`` in this epoch's order."""
        return self.order[position]


def document_order(count: int, *, seed: int, epoch: int, name: str) -> DocumentOrder:
    """Permute ``range(count)`` deterministically for one epoch.

    Args:
        count: Number of documents. ``0`` gives an empty order.
        seed: The run's seed. Same seed, same permutation.
        epoch: Which epoch's permutation to draw. Streams are independent per
            epoch, so ``epoch=1`` does not depend on ``epoch=0`` having run.
        name: Corpus or dataset name, so two datasets of equal length under one
            seed do not share a permutation.

    Returns:
        A :class:`DocumentOrder` whose ``order`` is a permutation of
        ``range(count)``.

    Note:
        ``name`` is the config ``name`` field used by weight init. A corpus with
        no name yet should pass its own label, and must then keep it: changing
        the label changes the order for a given seed.
    """
    if count < 0:
        raise ValueError(f"document count must be non-negative, got {count}")
    order = init_rng(seed, SHUFFLE_SITE, name, epoch).permutation(count)
    return DocumentOrder(
        seed=seed, epoch=epoch, name=name, order=tuple(int(i) for i in order)
    )
