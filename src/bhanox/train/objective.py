"""Next-byte loss and optimizer parameter selection, for training only (law C9).

Position-wise, not shifted
--------------------------
A chunk's ``inputs[j]`` and ``targets[j]`` are the *same* supervised example:
``inputs[j]`` is the byte 4-gram at document offset ``i+j`` and ``targets[j]`` is
the byte at ``i+j+4``. That alignment is the data pipeline's contract
(:mod:`bhanox.data.examples`), established by Stage 1 and not negotiable here.

So the loss pairs them index for index. There is no ``[:, 1:]``/``[:, :-1]``
shift, and adding one would be a silent correctness bug rather than an obvious
one: the shifted version still trains, still reports a falling loss, and
predicts the byte two positions ahead of where it should. A token language model
*is* shifted, which is exactly why the shift is a habit worth checking here.
``test_the_loss_is_position_wise`` in ``tests/test_trainer.py`` pins it against a
hand-computed value.

Reduction, and what it does not fix
-----------------------------------
:func:`next_byte_loss` reduces with ``sum``, and the caller divides by a token
count. For a single chunk that is the chunk's own mean, which is correct for
that chunk.

It does **not** make a short final chunk harmless under a per-chunk optimizer
schedule. A chunk of 1 example still triggers a full optimizer step, and its
gradient estimate is far noisier than a 4096-example chunk's. That is a property
of the *number of updates*, not of the averaging, and no amount of loss scaling
fixes it: AdamW's update is roughly ``lr * m_hat / (sqrt(v_hat) + eps)``, in
which numerator and denominator both scale with the loss, so multiplying a
chunk's loss by any constant leaves the step size nearly unchanged. The exposure
is bounded -- at most one short step per document -- and
``test_the_smoke_run_reports_chunk_token_counts`` reports the histogram so it is
visible rather than assumed away.

Inert parameters
----------------
:func:`trainable_parameters` excludes ``salience``. It is an ``nn.Parameter``
only so that the parameter count and the checkpoint inventory match the
reference's, and it is never read by the gate
(:mod:`bhanox.train.governor_mirror`), so it can never receive a gradient. An
optimizer group holding a permanently-``None`` gradient is a step that silently
does nothing to that slice of the model -- and a slice that does nothing is
indistinguishable, in a loss curve, from a slice that is learning.

The exclusion is by name and is deliberately narrow. :func:`unknown_parameter_names`
exists so a *new* parameter cannot slip past the filter unnoticed, borrowing the
discipline of ``_group_of`` in ``tests/test_model_gradients.py``, which raises
rather than passing when it meets a name it does not recognise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bhanox.train.model_mirror import BhanoxMirror

__all__ = [
    "INERT_NAME",
    "next_byte_loss",
    "token_mean",
    "trainable_parameters",
    "unknown_parameter_names",
]

#: Parameter-name fragment marking a parameter that is deliberately inert.
#: The gate's ``salience`` is the only one, and
#: ``test_the_inert_group_is_exactly_the_documented_one`` in
#: ``tests/test_model_gradients.py`` pins that it is the *only* one.
INERT_NAME = "salience"

#: Every trainable parameter name, grouped by which part of the mirror owns it.
#: Used only to fail loudly on an unrecognised name -- see
#: :func:`unknown_parameter_names`.
_KNOWN_PREFIXES = (
    "embed.",
    "unembed.",
    "mixers.",
    "banks.",
    "gates.",
)


def next_byte_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Next-byte cross-entropy, summed over positions.

    Args:
        logits: ``(B, T, output_vocab)`` from
            :meth:`~bhanox.train.model_mirror.BhanoxMirror.step`.
        targets: ``(T,)`` or ``(B, T)`` int64 byte values from a
            :class:`~bhanox.data.examples.Chunk`. Paired position-wise with
            ``logits``; nothing is shifted.

    Returns:
        A scalar tensor: the summed loss over every position.

    Note:
        Returns a **sum**, not a mean. Divide by the token count at the call
        site, where the count is known, so that the reduction cannot
        accidentally average across chunks of unequal size.

    Raises:
        ValueError: On a shape mismatch, or if ``targets`` is not integral.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, T, vocab), got {tuple(logits.shape)}")
    if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[0] == 1):
        flat_targets = targets.reshape(-1)
    else:
        raise ValueError(
            f"targets must be (T,) or (1, T), got {tuple(targets.shape)}; a "
            "(B, T) batch of documents is not supported because each row would "
            "need its own recurrent state carried in lockstep"
        )
    n_positions = int(logits.shape[0]) * int(logits.shape[1])
    if int(flat_targets.numel()) != n_positions:
        raise ValueError(
            f"targets has {int(flat_targets.numel())} entries but logits has "
            f"{n_positions} positions; the two must correspond one for one"
        )
    if flat_targets.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"targets must be an integer tensor, got {flat_targets.dtype}")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(logits.shape[-1])),
        flat_targets.to(torch.int64),
        reduction="sum",
    )


def token_mean(loss_sum: Tensor, n_tokens: int) -> Tensor:
    """Average a summed loss by token count.

    Args:
        loss_sum: Summed loss over the positions in this step.
        n_tokens: How many positions it covers.

    Returns:
        The scalar mean.

    Raises:
        ValueError: If ``n_tokens`` is not positive, which would be a caller
            bug rather than a property of the data.
    """
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be at least 1, got {n_tokens}")
    return loss_sum / float(n_tokens)


def trainable_parameters(mirror: BhanoxMirror) -> list[Tensor]:
    """Every parameter the optimizer should own.

    Args:
        mirror: The mirror whose parameters to collect.

    Returns:
        Parameters in ``named_parameters()`` order, excluding the inert
        ``salience``.

    Note:
        Use this rather than ``mirror.parameters()``. The difference is the
        512-value ``salience`` slice per gate, which is an ``nn.Parameter`` for
        checkpoint parity and is never read.
    """
    return [p for name, p in mirror.named_parameters() if INERT_NAME not in name]


def unknown_parameter_names(mirror: BhanoxMirror) -> list[str]:
    """Trainable parameter names that no group recognises.

    Args:
        mirror: The mirror to inspect.

    Returns:
        Names that neither carry a known mirror prefix nor are the documented
        inert one. Empty in the current design.

    Note:
        This is a guard, not a filter. An empty result today is a fact about
        today; a non-empty one means a new parameter was added without anyone
        deciding whether it trains. ``test_an_unrecognised_parameter_is_reported``
        checks the guard fires, because a check that has never failed is not
        known to work.
    """
    return [
        name
        for name, _ in mirror.named_parameters()
        if not name.startswith(_KNOWN_PREFIXES) and INERT_NAME not in name
    ]
