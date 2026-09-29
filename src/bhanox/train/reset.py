"""Document-boundary reset for the torch mirror, for training only (law C9).

Why this exists
---------------
The NumPy reference has :meth:`bhanox.model.Bhanox.reset`, which does two
things: it zeroes each DeltaBank's integer state and it flushes each PulseGate.
The mirror has no equivalent. :class:`~bhanox.train.governor_mirror.PulseGateMirror`
has ``flush`` and :class:`~bhanox.train.mirror.DeltaBankHeadMirror` has no reset
at all, so there was nothing for a trainer to call. This module supplies the
operation without adding a method to :class:`~bhanox.train.model_mirror.BhanoxMirror`,
which is already close to the 500-line module ceiling.

What a reset clears
-------------------
- **DeltaBank integer state.** ``state_int`` is the authoritative recurrent
  trajectory, and zeroing it is what makes the next document a genuinely new
  stream rather than a continuation of the previous one. Rows added later by
  ``ensure_batch`` already start at zero, so zeroing the rows that exist is
  sufficient and O(1) in sequence length.
- **PulseGate hysteresis.** ``cached``, ``_quiet`` and ``_has_run`` are zeroed
  and ``awake`` is set True, by ``flush``.

``_has_run`` is the one that is easy to miss. It exists to force the first
computation on a never-run channel, so a gate that carried it across a reset
would never wake again and would silently train against a dead channel.
``flush`` clears it; that is why this module calls ``flush`` rather than
reimplementing the state machine.

What a reset must not clear
---------------------------
Everything that is a property of the *run* rather than of the current document:

- every ``nn.Parameter``, including the trained weights and the inert
  ``salience``;
- ``bank_rates``, the per-layer decay rates;
- the mixers' ``loads`` counters and ``last_entropy``;
- the GELU lookup table;
- optimizer state, which is not reachable from the mirror and is untouched.

``loads`` in particular is deliberately lifetime. ``load_entropy()`` is a
run-level diagnostic -- it answers "is routing balanced across the whole run so
far", and resetting it per document would reduce it to a single-document number
that says nothing about whether experts are starving over training. The reference
accumulates it for the same reason
(:mod:`bhanox.checkpoint_inventory` lists it under ``SKIP`` as "a per-call MoE
routing counter by design").

When to call it
---------------
**At document boundaries only.** Within one document, a chunk is a scheduling
seam and its recurrent state must carry across it; the chunk's shadows are
detached and re-fed by
:func:`bhanox.train.trainer.clear_shadows`, which is a different operation on a
different tensor. Resetting per chunk would be wrong in a way no test would
catch on a single-document corpus: it would simply produce a model with no
memory across the sequence it is supposed to remember.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from bhanox.train.governor_mirror import PulseGateMirror
from bhanox.train.mirror import DeltaBankHeadMirror
from bhanox.train.model_mirror import BhanoxMirror, DeltaBankLayerMirror

__all__ = ["clear_shadows", "reset_mirror"]


def reset_mirror(mirror: BhanoxMirror) -> None:
    """Clear recurrent state, as a document boundary requires.

    Zeroes every DeltaBank head's integer state and flushes every PulseGate,
    mirroring :meth:`bhanox.model.Bhanox.reset`. Parameters, bank rates, the
    GELU table, ``loads`` and ``last_entropy`` are all left alone: they are
    run-level, not document-level.

    This is O(1) in sequence length -- it touches the state arrays, not the
    tokens that produced them -- and is safe to call between any two forwards.

    Args:
        mirror: The mirror to reset. Mutated in place.
    """
    banks = cast("list[DeltaBankLayerMirror]", list(mirror.banks))
    gates = cast("list[PulseGateMirror]", list(mirror.gates))
    for bank in banks:
        for head in cast("list[DeltaBankHeadMirror]", list(bank.heads)):
            with torch.no_grad():
                # ``state_int`` is the authoritative trajectory, advanced in
                # place by every ``step``. It is an int32 buffer, so this is a
                # genuine zero and not a float approximation of one.
                head.state_int.zero_()
    for gate in gates:
        gate.flush()


def clear_shadows(shadows: list[list[Tensor]] | None) -> list[list[Tensor]] | None:
    """Detach every shadow, keeping the values.

    This is the per-chunk counterpart to :func:`reset_mirror`, and the two must
    not be confused. Within a document the integer state carries forward on its
    own -- ``state_int`` is mutated in place by every ``step`` -- while the
    shadows are the differentiable mirror of that trajectory. Detaching them
    drops the autograd history and keeps the numbers, so the next chunk's
    forward still reads the state the previous chunk actually walked.

    **The trade-off, stated plainly:** this is truncated backpropagation at the
    chunk seam. No later chunk can influence an earlier chunk's parameter
    update, because the graph that would carry that credit is gone. At nano the
    shadow graph is 4 layers x 4 heads x 16 x 32 x 4 bytes = 32 KiB per token,
    so a 4096-token document would hold ~128 MiB of it live; that is the memory
    this avoids, and the long-range gradient credit is what it costs.

    Consequently a per-chunk optimizer schedule is **not** equivalent to one
    optimizer update over a whole document. Three things differ: gradients are
    truncated at every seam, the optimizer steps once per chunk rather than once
    per document, and the weights change between chunks, so the trajectory after
    the first update is a different function than the one a single update would
    have walked. Nothing here claims the two are interchangeable.

    Args:
        shadows: ``next_shadows`` from a previous
            :meth:`~bhanox.train.model_mirror.BhanoxMirror.step`, or ``None`` to
            start a fresh sequence.

    Returns:
        The same structure with every tensor detached, or ``None`` for ``None``.
    """
    if shadows is None:
        return None
    return [[t.detach() for t in layer] for layer in shadows]
