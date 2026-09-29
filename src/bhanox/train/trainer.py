"""The minimal training loop: one chunk, and a document stream, for the torch mirror.

Scope, deliberately
-------------------
This is a prototype, not a training program. It exists to answer one question --
*can a gradient flow through the whole mirror, and does the loss fall on data
with a learnable structure?* -- and it uses synthetic in-memory bytes only. It
does not read a corpus, checkpoint, resume, or schedule anything. A real training
run is a later milestone and will differ in every respect that matters for
quality: data, schedule, optimizer settings, and the cross-implementation
agreement that ``model_mirror.py`` documents as not holding after weight updates.

The schedule, and what it is not
--------------------------------
**One optimizer step per chunk**, with the returned shadows detached before the
next chunk. That is truncated backpropagation at the chunk seam, and it is not
equivalent to one optimizer update over a whole document. Three things differ:

1. Gradients are truncated at every seam, so a chunk is optimised using only
   evidence from itself.
2. The optimizer steps once per chunk rather than once per document, so a
   4096-example document at ``max_context=4096`` produces ten updates.
3. The weights change between chunks, so the trajectory after the first update
   is a different function than a single update would have walked.

What is preserved across the seam is the *recurrent state*: ``state_int`` is
mutated in place by every ``step`` and carries forward untouched. So the model
still remembers the document; it just cannot be credited for having done so more
than a chunk back. The memory this buys is the point -- at nano the shadow graph
is 32 KiB per token, so a full document would hold ~128 MiB live.

A short final chunk
-------------------
``max_examples`` bounds emitted input positions, so a document of ``N`` bytes
yields ``ceil((N - 4) / max_examples)`` chunks and the last is usually short.
That chunk gets a **full optimizer step** on a noisy gradient estimate. This is
known and not fixed here: the loss is already divided by that chunk's own token
count, and dividing again would not help, because AdamW's update is
approximately invariant to a uniform scaling of the loss (numerator and
denominator in ``m_hat / sqrt(v_hat)`` scale together). The exposure is bounded at
one short step per document, and :func:`run_documents` reports the token-count
histogram so it is visible rather than assumed away.

``b`` and ``loads``
-------------------
``train=True`` on the loss-producing forward only. That is the design the owner
chose for the prototype, and it means ``b`` receives two writes per step: the
optimizer's gradient, and the no-gradient load-balancing nudge from
:meth:`~bhanox.train.mixer_mirror.MicroExpertLayerMirror.update_load_bias`. The
nudge bypasses AdamW entirely -- not its moments, not weight decay, not any
schedule -- so ``b``'s movement is not attributable to the learning rate alone.
Any future diagnostic on ``b`` has to separate the two.

Evaluation goes through :func:`evaluate_chunk` with ``train=False``, which
leaves ``b``, ``loads`` and ``last_entropy`` bit-identical. That is a property of
the call site, not of ``train=True``: ``train=True`` is a state mutation and
does change all three.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from bhanox.data import Chunk, iter_examples
from bhanox.train.model_mirror import BhanoxMirror
from bhanox.train.objective import next_byte_loss, token_mean, trainable_parameters
from bhanox.train.reset import clear_shadows, reset_mirror

__all__ = [
    "ChunkReport",
    "DocumentReport",
    "build_optimizer",
    "evaluate_chunk",
    "run_documents",
    "train_chunk",
]


@dataclass(frozen=True)
class ChunkReport:
    """What one chunk contributed, for the caller to aggregate.

    Attributes:
        doc_index: Which document this chunk belonged to.
        byte_start: Document offset of the chunk's first owned context.
        n_tokens: Emitted input positions, i.e. supervised examples.
        loss: Token-mean loss for this chunk.
    """

    doc_index: int
    byte_start: int
    n_tokens: int
    loss: float


@dataclass(frozen=True)
class DocumentReport:
    """Per-chunk losses and counts for one document.

    Attributes:
        doc_index: The document these chunks belonged to.
        chunks: One entry per chunk, in document order.
    """

    doc_index: int
    chunks: tuple[ChunkReport, ...] = field(default=())

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def n_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chunks)

    @property
    def mean_loss(self) -> float:
        """Token-weighted mean over the document.

        Weighted rather than a mean of per-chunk means, because the last chunk
        is usually short and a mean of means would give it the same vote as a
        full chunk. Note this weighting applies to *reporting* only; the
        per-chunk optimizer steps are unaffected by it.
        """
        total = sum(c.n_tokens for c in self.chunks)
        if total == 0:
            return 0.0
        return sum(c.loss * c.n_tokens for c in self.chunks) / total

    @property
    def token_counts(self) -> tuple[int, ...]:
        return tuple(c.n_tokens for c in self.chunks)


def build_optimizer(
    mirror: BhanoxMirror,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """An AdamW over the trainable parameters, excluding ``salience``.

    Args:
        mirror: The mirror to optimise.
        lr: Learning rate. The prototype default; a real schedule is a later
            decision.
        weight_decay: Decoupled weight decay, applied by AdamW and not to
            ``b``'s load-balancing nudge.

    Returns:
        A configured ``AdamW``.

    Note:
        The parameter list comes from
        :func:`~bhanox.train.objective.trainable_parameters`, not
        ``mirror.parameters()``. The difference is the inert ``salience``, which
        would otherwise sit in an optimizer group forever receiving a ``None``
        gradient.
    """
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    params = trainable_parameters(mirror)
    if not params:
        raise ValueError("no trainable parameters; check the model was built")
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
    )


def train_chunk(
    mirror: BhanoxMirror,
    optimizer: torch.optim.Optimizer,
    chunk: Chunk,
    shadows: list[list[Tensor]] | None = None,
) -> tuple[ChunkReport, list[list[Tensor]]]:
    """One optimizer step on one chunk, and the detached shadows to continue with.

    Args:
        mirror: The mirror to train. Its integer state advances.
        optimizer: The optimizer to step.
        chunk: One chunk from :func:`bhanox.data.iter_examples`.
        shadows: Detached shadows from the previous chunk of the *same*
            document, or ``None`` to start fresh.

    Returns:
        ``(report, next_shadows)`` where ``next_shadows`` are already detached
        and safe to feed into the following chunk of the same document.

    Note:
        ``train=True`` here and only here: this is the loss-producing forward.
        The shadows come back detached, which preserves the recurrent value for
        the next chunk while truncating gradient flow across the seam. See the
        module docstring for what that costs.
    """
    ids = chunk.inputs.reshape(1, -1)
    optimizer.zero_grad(set_to_none=True)
    logits, next_shadows = mirror.step(ids, shadows=shadows, train=True)
    # Built after the forward and moved onto ``logits``' device. ``from_numpy``
    # always lands on the *default* device, and ``cross_entropy`` requires its
    # input and target to match, so a mirror living on CUDA raises
    # "Tensor on device cpu is not on the expected device cuda:0" here unless
    # the targets follow it. The move is a no-op on CPU, where the two devices
    # already agree.
    targets = torch.from_numpy(np.asarray(chunk.targets, dtype=np.int64)).to(
        logits.device
    )
    loss_sum = next_byte_loss(logits, targets)
    loss_sum.backward()
    optimizer.step()
    return (
        ChunkReport(
            doc_index=chunk.doc_index,
            byte_start=chunk.byte_start,
            n_tokens=len(chunk),
            loss=float(token_mean(loss_sum.detach(), len(chunk))),
        ),
        clear_shadows(next_shadows) or [],
    )


def evaluate_chunk(
    mirror: BhanoxMirror,
    chunk: Chunk,
    shadows: list[list[Tensor]] | None = None,
) -> tuple[float, list[list[Tensor]]]:
    """Loss on a chunk with no side effects.

    Args:
        mirror: The mirror. Its integer state still advances -- that is the
            recurrence, not a side effect of routing.
        chunk: The chunk to score.
        shadows: Detached shadows from the previous chunk of the same document.

    Returns:
        ``(token_mean_loss, next_shadows)``, the shadows detached.

    Note:
        ``train=False`` is the point of this function. It leaves the mixers'
        ``b``, ``loads`` and ``last_entropy`` bit-identical, so an evaluation
        pass cannot contaminate the next training step's load-balancing nudge or
        its entropy diagnostic. ``train=True`` would change all three.
    """
    ids = chunk.inputs.reshape(1, -1)
    with torch.no_grad():
        logits, next_shadows = mirror.step(ids, shadows=shadows, train=False)
        # Same device rule as ``train_chunk``; see the comment there.
        targets = torch.from_numpy(np.asarray(chunk.targets, dtype=np.int64)).to(
            logits.device
        )
        loss_sum = next_byte_loss(logits, targets)
        loss = float(token_mean(loss_sum, len(chunk)))
    return loss, clear_shadows(next_shadows) or []


def run_documents(
    mirror: BhanoxMirror,
    optimizer: torch.optim.Optimizer,
    documents: Sequence[bytes],
    *,
    max_examples: int,
    seed: int = 0,
    epoch: int = 0,
    name: str = "trainer",
    reset: Callable[[BhanoxMirror], None] = reset_mirror,
) -> Iterator[DocumentReport]:
    """Train over synthetic documents, yielding one report per document.

    The recurrence rule, unchanged from the data pipeline's design: state carries
    across the chunks of one document, and resets only when ``doc_index`` changes.

    Args:
        mirror: The mirror to train.
        optimizer: The optimizer to step once per chunk.
        documents: In-memory byte documents. Nothing is read from disk.
        max_examples: Emitted input positions per chunk, normally
            ``config.max_context``.
        seed: Forwarded to the document ordering.
        epoch: Forwarded to the document ordering.
        name: Corpus label, forwarded to the ordering. A label for synthetic
            bytes, not a dataset.
        reset: The document-boundary reset, injectable for tests.

    Yields:
        A :class:`DocumentReport` per document that produced at least one chunk.

    Note:
        A document shorter than one example is skipped by the underlying
        iterator and produces no report, so a caller cannot tell an empty
        document from an absent one. That matches
        :func:`bhanox.data.iter_examples`.
    """
    shadows: list[list[Tensor]] | None = None
    current: int | None = None
    pending: list[ChunkReport] = []
    for chunk in iter_examples(
        documents, max_examples=max_examples, seed=seed, epoch=epoch, name=name
    ):
        if chunk.doc_index != current:
            if current is not None:
                yield DocumentReport(doc_index=current, chunks=tuple(pending))
            reset(mirror)
            shadows = None
            current = chunk.doc_index
            pending = []
        report, shadows = train_chunk(mirror, optimizer, chunk, shadows)
        pending.append(report)
    if current is not None:
        yield DocumentReport(doc_index=current, chunks=tuple(pending))
