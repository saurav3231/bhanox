"""The VectorVault scalar state that the array walk structurally cannot carry.

Split out of ``bhanox.checkpoint`` for law C5 (one concern per module). The
archive in ``bhanox.checkpoint`` owns the file; this owns the one component's
runtime metadata that the format cannot express as an array.

Why the scalars need their own versioned block:

- The inventory walk in :mod:`bhanox.checkpoint_inventory` can only carry
  arrays, so a dataclass holding live state in plain ints and floats would
  restore as an empty husk with its tables still full. The VectorVault is the
  live case: ``filled`` and ``clock`` decide whether the recovered entries can
  be read at all, so they travel in a namespaced, versioned metadata block
  (:data:`RUNTIME_KEY`), and a checkpoint carrying the arrays without that
  block is refused rather than half-restored.
- ``theta_s`` and ``perm_budget_bytes`` decide which writes are admitted at
  all, so a resume that lost them would quietly change what the model keeps.
- ``n_slots`` is recorded as a cross-check rather than restored. The current
  table size is movable at runtime via ``set_budget``/``set_entry_cap``, so a
  checkpoint from a resized vault has array shapes a fresh vault cannot
  accept, and naming that beats the generic shape error.

The validate/apply split (:func:`_check_vault_state` then
:func:`_apply_vault_state`) is load-bearing rather than stylistic: validation
runs *before* any array is written, so a refusal leaves the target model
untouched instead of half-overwritten.
"""

from __future__ import annotations

import math
from typing import Any

from numpy.typing import NDArray

#: Top-level metadata key for state the array walk cannot express. Namespaced so
#: a component's scalars can never collide with a tensor name or with a caller's
#: ``extra`` field, both of which ``save`` already guards.
RUNTIME_KEY = "runtime"

#: Version of the runtime block itself, checked on load so a block written by a
#: future layout is refused instead of half-understood.
RUNTIME_VERSION = 1

#: The VectorVault scalars the inventory structurally cannot carry.
#:
#: ``filled`` and ``clock`` are what turn a restored vault from *populated*
#: into *usable*. With ``filled`` back at its initial 0, ``query`` short-circuits
#: and reports an empty store while the recovered entries sit untouched in the
#: slot tables, and ``write`` hands out slot 0 again and again, overwriting one
#: recovered entry per write and orphaning the rest. Nothing raises along that
#: path.
VAULT_STATE = ("filled", "clock", "theta_s", "perm_budget_bytes", "n_slots")


def _vault_state(vault: Any) -> dict[str, Any]:
    """The vault's scalar runtime state as JSON-safe values.

    ``int``/``float`` rather than whatever NumPy scalar a field happens to hold,
    because ``json.dumps`` raises on ``np.int64`` and a checkpoint that cannot be
    written is not a checkpoint.
    """
    budget = vault.perm_budget_bytes
    return {
        "version": RUNTIME_VERSION,
        "filled": int(vault.filled),
        "clock": int(vault.clock),
        "theta_s": float(vault.theta_s),
        "perm_budget_bytes": None if budget is None else int(budget),
        "n_slots": int(vault.n_slots),
    }


def _int_field(block: dict[str, Any], key: str) -> int:
    """Read ``key`` as a plain int, refusing bools and floats.

    Strictly typed on purpose: a block hand-edited to ``"filled": "12"`` is
    refused rather than coerced, because coercion is how a resume ends up
    half-restored. ``bool`` is excluded explicitly since it subclasses ``int``.
    """
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"VectorVault {key} must be a JSON integer, got {value!r} of type "
            f"{type(value).__name__}"
        )
    return value


def _check_vault_state(
    vault: Any, block: Any, path: Any, age: NDArray[Any] | None
) -> dict[str, Any]:
    """Validate the recorded vault scalars and return them ready to apply.

    Split from the apply step so it can run *before* any array is written. A
    refusal therefore leaves the target model untouched, and a checkpoint whose
    vault was resized by ``set_budget``/``set_entry_cap`` is named as such
    rather than surfacing as a bare shape mismatch from the array walk.

    ``age`` is the ``vault.age`` array as stored in the checkpoint, or ``None``
    when the archive has none. The age/clock cross-check reads it from the
    archive rather than from ``vault`` so that it too happens before the write;
    it is checking the data about to be restored, which is the stricter and
    more useful reading.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"VectorVault runtime state in {path} must be a JSON object, got "
            f"{type(block).__name__}"
        )
    version = block.get("version")
    if version != RUNTIME_VERSION:
        raise ValueError(
            f"VectorVault runtime state in {path} is version {version!r}, this "
            f"build reads version {RUNTIME_VERSION}"
        )
    missing = [key for key in VAULT_STATE if key not in block]
    if missing:
        raise ValueError(
            f"VectorVault runtime state in {path} is missing {missing}; refusing "
            "to restore a vault whose scalars are incomplete"
        )
    unknown = sorted(set(block) - set(VAULT_STATE) - {"version"})
    if unknown:
        raise ValueError(
            f"VectorVault runtime state in {path} has unknown fields {unknown}"
        )

    n_slots = _int_field(block, "n_slots")
    if n_slots != int(vault.n_slots):
        raise ValueError(
            f"checkpoint holds a {n_slots}-slot VectorVault but the model has "
            f"{int(vault.n_slots)}; a vault resized by set_budget/set_entry_cap "
            "cannot be restored into a differently sized one"
        )
    filled = _int_field(block, "filled")
    if not 0 <= filled <= n_slots:
        raise ValueError(f"VectorVault filled={filled} is outside [0, {n_slots}]")
    clock = _int_field(block, "clock")
    if clock < 0:
        raise ValueError(f"VectorVault clock={clock} is negative")

    raw_theta = block["theta_s"]
    if isinstance(raw_theta, bool) or not isinstance(raw_theta, (int, float)):
        raise ValueError(
            f"VectorVault theta_s must be a JSON number, got {raw_theta!r} of "
            f"type {type(raw_theta).__name__}"
        )
    theta_s = float(raw_theta)
    if not math.isfinite(theta_s):
        raise ValueError(f"VectorVault theta_s={theta_s!r} is not finite")

    budget = block["perm_budget_bytes"]
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError(
                "VectorVault perm_budget_bytes must be a JSON integer or null, "
                f"got {budget!r} of type {type(budget).__name__}"
            )
        if budget < 0:
            raise ValueError(f"VectorVault perm_budget_bytes={budget} is negative")

    # A write stamps ``age`` with the clock value it just advanced, so the
    # recorded clock bounds every stored age. Truncation lowers ``filled``
    # without rewinding ``clock``, hence ``<=`` rather than ``==``. A clock
    # older than a stored age means the block and the tables came from
    # different places, and applying it would corrupt recency-based eviction
    # with nothing failing.
    if filled and age is not None and age.size:
        newest = int(age[:filled].max())
        if newest > clock:
            raise ValueError(
                f"VectorVault clock={clock} trails a stored age of {newest}; the "
                "recorded scalars disagree with the stored slot tables"
            )

    return {
        "filled": filled,
        "clock": clock,
        "theta_s": theta_s,
        "perm_budget_bytes": budget,
    }


def _apply_vault_state(vault: Any, state: dict[str, Any]) -> None:
    """Write scalars already cleared by :func:`_check_vault_state`.

    Every refusal has happened by the time this runs, so the arrays and the
    scalars that describe them land together or not at all.
    """
    vault.filled = state["filled"]
    vault.clock = state["clock"]
    vault.theta_s = state["theta_s"]
    vault.perm_budget_bytes = state["perm_budget_bytes"]
