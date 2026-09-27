"""Checkpoint save and load: versioned, atomic, and NumPy-only.

Why this shape:

- **Inventory, not a hand-written field list.** Every array on the model is
  discovered by walking the dataclass fields, so a new parameter cannot be
  silently left out of the checkpoint. A hand-maintained list looks tidier and
  is exactly where this kind of bug hides: someone adds a tensor, forgets this
  file, and training resumes from a checkpoint that quietly dropped it.
  ``test_inventory_covers_every_array`` is the guard.
- **Atomic.** A run that dies mid-write must not leave a half-written file that
  loads as a valid checkpoint. Write to a sibling temp file, fsync, then
  ``os.replace``, which is atomic on both POSIX and Windows. A reader sees
  either the old file or the new one, never a truncated mix.
- **Versioned.** ``format`` and ``version`` are written into the archive and
  checked on load, so an old file is refused with a clear error instead of
  being misread as a newer layout.
- **Bit-exact.** Weights round-trip as raw arrays with no cast and no
  requantisation. A checkpoint that reloads to slightly different numbers is
  worse than no checkpoint: it looks like it worked.

Recurrent state is deliberately *not* saved. It is a fixed buffer rebuilt by
``reset()``, which is what keeps a checkpoint proportional to parameters rather
than to context length. The trainer owns the decision to carry a sequence
across a resume, and that is a training concern, not a model one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bhanox.config import BhanoxConfig

#: Written into every archive and checked on load.
FORMAT = "bhanox-checkpoint"

#: Bump when the layout changes incompatibly. 1 is the initial flat inventory.
VERSION = 1

#: Keys used for scalars in the archive. Prefixed to avoid colliding with an
#: inventory name.
META_KEY = "__meta__"
STEP_KEY = "__step__"

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


def _atomic_write(path: Path, write: Any) -> None:
    """Run ``write(tmp)`` then atomically move it onto ``path``.

    The temp file is a sibling so the final move stays on one filesystem,
    which is the condition for ``os.replace`` being atomic. A half-written
    checkpoint is never visible under the real name.
    """
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save(
    model: Any,
    path: str | Path,
    *,
    step: int = 0,
    config: BhanoxConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint of ``model`` to ``path`` atomically.

    Args:
        model: The model to persist. Every array in :func:`inventory` is
            written, bit-exact.
        path: Destination file. Written atomically, so a crash mid-save leaves
            the previous checkpoint intact.
        step: Training step to record, for the trainer's resume logic.
        config: Config to record for validation on load. Defaults to the
            model's own ``config`` attribute.
        extra: Additional JSON-serialisable metadata. Must not shadow
            inventory keys.

    Returns:
        The path written.
    """
    path = Path(path)
    if config is None:
        config = getattr(model, "config", None)
    tensors = inventory(model)

    meta: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "step": int(step),
        "tensors": [name for name, _ in tensors],
        "dtypes": {name: str(arr.dtype) for name, arr in tensors},
        "shapes": {name: list(arr.shape) for name, arr in tensors},
    }
    if config is not None:
        meta["config"] = config.to_dict()
    if extra:
        clash = set(extra) & set(meta["tensors"])
        if clash:
            raise ValueError(f"extra metadata shadows tensor names: {sorted(clash)}")
        meta["extra"] = extra

    payload: dict[str, Any] = {name: np.asarray(arr) for name, arr in tensors}
    payload[META_KEY] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    payload[STEP_KEY] = np.asarray(int(step), dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda fh: np.savez(fh, **payload))
    return path


def read_meta(path: str | Path) -> dict[str, Any]:
    """Load and validate the metadata of a checkpoint without touching a model.

    Raises:
        OSError: The file is not a Bhanox checkpoint at all.
        ValueError: The format or version is not one this build understands.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        if META_KEY not in data:
            raise OSError(f"{path} is not a Bhanox checkpoint")
        meta = json.loads(bytes(data[META_KEY]).decode("utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Bhanox checkpoint")
    if meta.get("version") != VERSION:
        raise ValueError(
            f"{path} is checkpoint version {meta.get('version')}, "
            f"this build reads version {VERSION}"
        )
    return meta


def load(path: str | Path, model: Any | None = None) -> Any:
    """Restore a model from ``path``.

    Args:
        path: Checkpoint to read.
        model: Target to overwrite. If ``None``, a model is built from the
            config recorded in the checkpoint, so the returned model has the
            right architecture without the caller having to remember it.

    Returns:
        The model, restored in place if one was passed, or newly built.

    Raises:
        OSError: The file is not a Bhanox checkpoint.
        ValueError: The version is unknown, or a tensor does not fit the
            target model's shapes.
    """
    meta = read_meta(path)

    if model is None:
        cfg = meta.get("config")
        if cfg is None:
            raise ValueError(
                f"{path} has no recorded config; pass a model to load into"
            )
        model = _build_from_config(cfg)

    tensors = meta["tensors"]
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [n for n in tensors if n not in data.files]
        if missing:
            raise ValueError(f"{path} is missing tensors: {missing[:4]}")
        for name in tensors:
            _assign(model, name, data[name])
    return model


def _build_from_config(cfg: dict[str, Any]) -> Any:
    from bhanox.model import Bhanox

    return Bhanox(BhanoxConfig.from_dict(cfg))
