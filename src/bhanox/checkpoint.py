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
- **Scalars are not silently dropped.** The inventory walk can only carry
  arrays, so a dataclass holding live state in plain ints and floats would
  restore as an empty husk with its tables still full. The VectorVault is the
  live case: ``filled`` and ``clock`` decide whether the recovered entries can
  be read at all, so they travel in a namespaced, versioned metadata block
  (:data:`RUNTIME_KEY`), and a checkpoint carrying the arrays without that
  block is refused rather than half-restored.

Recurrent state is deliberately *not* saved. It is a fixed buffer rebuilt by
``reset()``, which is what keeps a checkpoint proportional to parameters rather
than to context length. The trainer owns the decision to carry a sequence
across a resume, and that is a training concern, not a model one.

Module layout (law C5, one concern per module). This module is the *archive*:
the file format, the atomic write, and ``save``/``load``. Its two supporting
concerns live next door and are re-exported here, so every name below keeps
importing from ``bhanox.checkpoint`` exactly as before:

- :mod:`bhanox.checkpoint_inventory` -- which arrays persist, and the path
  grammar that maps an inventory name onto a live object graph.
- :mod:`bhanox.checkpoint_vault` -- the VectorVault scalar runtime state the
  array walk structurally cannot carry.

The dependency runs one way (this module imports from both; neither imports
back), so neither concern has to know about the other's half of the job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from bhanox.checkpoint_inventory import (
    SKIP,
    _assign,
    _segments,
    _walk,
    inventory,
)
from bhanox.checkpoint_vault import (
    RUNTIME_KEY,
    RUNTIME_VERSION,
    VAULT_STATE,
    _apply_vault_state,
    _check_vault_state,
    _int_field,
    _vault_state,
)
from bhanox.config import BhanoxConfig

__all__ = [
    "FORMAT",
    "META_KEY",
    "RUNTIME_KEY",
    "RUNTIME_VERSION",
    "SKIP",
    "STEP_KEY",
    "VAULT_STATE",
    "VERSION",
    "_apply_vault_state",
    "_assign",
    "_check_vault_state",
    "_int_field",
    "_segments",
    "_vault_state",
    "_walk",
    "inventory",
    "load",
    "read_meta",
    "save",
]

#: Written into every archive and checked on load.
FORMAT = "bhanox-checkpoint"

#: Bump when the layout changes incompatibly. 1 is the initial flat inventory.
VERSION = 1

#: Keys used for scalars in the archive. Prefixed to avoid colliding with an
#: inventory name.
META_KEY = "__meta__"
STEP_KEY = "__step__"


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
    # Recorded here rather than left to the caller: the walk cannot see these
    # fields, and a caller who has to remember them is a caller who forgets.
    vault = getattr(model, "vault", None)
    if vault is not None:
        meta[RUNTIME_KEY] = {"vectorvault": _vault_state(vault)}
    if extra:
        clash = set(extra) & (set(meta["tensors"]) | {RUNTIME_KEY})
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
        The model, restored in place if one was passed, or newly built. A
        VectorVault's slot tables *and* its scalar state come back together, so
        the returned vault is the one that was saved rather than an empty
        husk holding recovered entries.

    Raises:
        OSError: The file is not a Bhanox checkpoint.
        ValueError: The version is unknown, a tensor does not fit the target
            model's shapes, or the VectorVault state is absent, incomplete, or
            disagrees with the slot tables beside it.
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
    # Checked before a single array is written, so a refused load leaves the
    # target untouched rather than half-overwritten.
    block = (meta.get(RUNTIME_KEY) or {}).get("vectorvault")
    if block is None and any(n.startswith("vault.") for n in tensors):
        raise ValueError(
            f"{path} stores VectorVault arrays but no VectorVault scalar state: "
            "it predates vault state being checkpointed, and restoring it would "
            "rebuild a populated vault reporting filled=0 that answers no query. "
            "Re-save it from a build that records vault state."
        )
    if block is not None and getattr(model, "vault", None) is None:
        raise ValueError(
            f"{path} holds VectorVault state but the target model has no vault"
        )
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [n for n in tensors if n not in data.files]
        if missing:
            raise ValueError(f"{path} is missing tensors: {missing[:4]}")
        # Checked here, before the assign loop, so a bad block refuses the load
        # outright instead of half-overwriting the target first.
        state = None
        if block is not None:
            state = _check_vault_state(
                model.vault,
                block,
                path,
                data["vault.age"] if "vault.age" in data.files else None,
            )
        for name in tensors:
            _assign(model, name, data[name])
    if state is not None:
        _apply_vault_state(model.vault, state)
    return model


def _build_from_config(cfg: dict[str, Any]) -> Any:
    from bhanox.model import Bhanox

    return Bhanox(BhanoxConfig.from_dict(cfg))
