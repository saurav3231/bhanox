"""Which arrays a checkpoint carries, and how a name addresses one.

Split out of ``bhanox.checkpoint`` for law C5 (one concern per module). The
archive is the *format*; this is the *addressing* -- the set of tensors that
persist, and the grammar that maps a name like ``deltabanks[0].heads[1].W_k`` to
a place in a live object graph. The archive in ``bhanox.checkpoint`` imports
these two functions from here, so the dependency runs one way and neither
concern has to know about the other's half of the job.

Why inventory rather than a hand-written field list:

- Every array on the model is discovered by walking the dataclass fields, so a
  new parameter cannot be silently left out of the checkpoint. A hand-maintained
  list looks tidier and is exactly where this kind of bug hides: someone adds a
  tensor, forgets the list, and training resumes from a checkpoint that quietly
  dropped it. ``test_inventory_covers_every_array`` is the guard.
- The walk in :func:`inventory` and the write-back in :func:`_assign` share one
  path grammar, so the two cannot drift. Every refusal is raised, never skipped:
  a checkpoint that loads "mostly" is the failure this guards against.
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Field names never persisted, at any depth.
#:
#: These are all *runtime* state rather than parameters. The reason to keep the
#: list short is that a checkpoint silently missing a weight is a different
#: model; the reason to include these is that they are not weights.
#:
#: ``state``, ``cached``, ``awake``, ``_quiet``, ``_has_run`` are the recurrent
#: buffers, rebuilt by ``reset()``/``flush()``. Saving them would also make a
#: checkpoint's contents depend on the batch size of whichever run wrote it --
#: these carry a sample axis, so a batch-4 save would not equal a batch-1 save
#: of the same weights. Parameters-only keeps the file predictable, and a
#: stateful mid-sequence resume is the trainer's to own.
#:
#: ``loads`` is a per-call MoE routing counter by design, and ``decay``/``decay_q``
#: are derived from the bank rates.
SKIP = frozenset(
    {
        "state",
        "cached",
        "awake",
        "_quiet",
        "_has_run",
        "loads",
        "decay_q",
        "decay",
    }
)


def _walk(obj: Any, prefix: str) -> list[tuple[str, NDArray[Any]]]:
    """Every array reachable from ``obj``, as ``(name, array)`` pairs.

    Walks dataclass fields in declaration order, so the inventory is stable
    across runs and diffable between two checkpoints of the same model.
    Recursion is depth-first and containers (list of layers) are indexed.
    """
    found: list[tuple[str, NDArray[Any]]] = []
    if isinstance(obj, np.ndarray):
        return [(prefix, obj)]
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            if f.name in SKIP:
                continue
            value = getattr(obj, f.name)
            if isinstance(value, np.ndarray):
                found.append((f"{prefix}.{f.name}", value))
            elif is_dataclass(value) and not isinstance(value, type):
                found.extend(_walk(value, f"{prefix}.{f.name}"))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, np.ndarray):
                        found.append((f"{prefix}.{f.name}[{i}]", item))
                    elif is_dataclass(item) and not isinstance(item, type):
                        found.extend(_walk(item, f"{prefix}.{f.name}[{i}]"))
    return found


def inventory(model: Any) -> list[tuple[str, NDArray[Any]]]:
    """Name and array for every persisted tensor on ``model``.

    The recurrent state and the volatile per-step counters are excluded via
    :data:`SKIP`: the state is rebuilt by ``reset()`` and the counters are
    derived. Everything else is a weight or a learned table, and a checkpoint
    missing a weight is silently a different model.
    """
    found: list[tuple[str, NDArray[Any]]] = []
    for f in fields(model):
        if f.name in SKIP:
            continue
        value = getattr(model, f.name)
        if isinstance(value, np.ndarray):
            found.append((f.name, value))
        elif is_dataclass(value) and not isinstance(value, type):
            found.extend(_walk(value, f.name))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, np.ndarray):
                    found.append((f"{f.name}[{i}]", item))
                elif is_dataclass(item) and not isinstance(item, type):
                    found.extend(_walk(item, f"{f.name}[{i}]"))
    return found


def _assign(model: Any, name: str, array: NDArray[Any]) -> None:
    """Write ``array`` back into ``model`` at inventory ``name``.

    Inverts :func:`inventory` by walking the same path grammar, so the two
    cannot drift: ``deltabanks[0].heads[1].W_k`` steps through a field, an
    index, a field, an index, and a leaf. A path that does not resolve is a
    version error and is raised, never skipped -- a checkpoint that loads
    "mostly" is the failure this guards against.
    """
    target: Any = model
    for i, seg in enumerate(_segments(name)):
        last = i == len(_segments(name)) - 1
        if seg.startswith("["):
            idx = int(seg[1:-1])
            if not isinstance(target, (list, tuple)) or idx >= len(target):
                raise ValueError(f"checkpoint path {name!r} does not fit this model")
            target = target[idx]
            continue
        if not hasattr(target, seg):
            raise ValueError(f"checkpoint has no field {seg!r} in path {name!r}")
        target = getattr(target, seg)
        if last:
            if not isinstance(target, np.ndarray):
                raise ValueError(f"checkpoint field {name!r} is not an array")
            if target.shape != array.shape:
                raise ValueError(
                    f"checkpoint field {name!r} has shape {array.shape}, "
                    f"model expects {target.shape}"
                )
            target[...] = array


def _segments(name: str) -> list[str]:
    """Split an inventory name into field and index steps.

    ``"mixers[2].E"`` -> ``["mixers", "[2]", "E"]``.
    """
    out: list[str] = []
    for part in name.split("."):
        match = re.match(r"^([A-Za-z_]\w*)((?:\[\d+\])*)$", part)
        if not match:
            raise ValueError(f"unparseable checkpoint path segment: {part!r}")
        out.append(match.group(1))
        for idx in re.findall(r"\[\d+\]", match.group(2)):
            out.append(idx)
    return out
