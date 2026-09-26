"""Memory budgets: the user-facing byte ceilings from spec D7.

Purpose: let a caller cap Bhanox's two memories in *bytes they choose*, which is
the one thing a Transformer cannot do to its KV cache and Mamba cannot do to its
state. A Transformer grows with context; a Mamba state is fixed at construction;
Bhanox's two memories are explicit, bounded, and settable.

In simple words: you say how much memory you can afford, and Bhanox stays inside
it instead of asking you to buy more.

Both budgets default from the reference configs, so leaving them alone changes no
existing behaviour. Oversubscribing is legal and warns once. Undersubscribing
never raises: the vault truncates by importance score, because running out of
budget must degrade recall, not crash the process.

Sizing, corrected to the built layout (see ``docs/architecture.md``):

    TEMP (DeltaBank working state) = n_layers * n_heads * d_k * d_v
    PERM (VectorVault)             = n_slots * (bits // 8 + 4 * d_value + META)

The design-phase spec wrote these as ``L * B * 2d`` and ``E * (2d + 16)``. Both
undercount the built architecture, and the second by 14x, because they assume a
key is ``d`` wide. Here a key is a hypervector of ``bits`` bits, and the
temporary state is ``d_k`` rows by ``d_v`` columns per head -- not a slot count
times a width. A budget check that undercounts is worse than no budget, so the
formulas here are the ones the code actually measures.
"""

from __future__ import annotations

import re

__all__ = [
    "ENTRY_META_BYTES",
    "format_bytes",
    "parse_bytes",
    "perm_entry_bytes",
    "temp_state_bytes",
]

# Per-entry bookkeeping the vault keeps alongside the key and value: a float32
# salience and an int64 write-clock stamp, which is what the importance score
# reads. Measured against the real arrays, not guessed -- see
# ``test_entry_meta_matches_the_stored_record``, which recomputes it from the
# dtypes. Getting this wrong by a few bytes per entry is what made an earlier
# draft report more *used* than *reserved*.
ENTRY_META_BYTES = 12

_UNITS = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2}
_UNITS["G"] = 1024**3
_UNITS["GB"] = 1024**3
_UNITS["T"] = 1024**4
_UNITS["TB"] = 1024**4

_SPEC = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")


def parse_bytes(spec: str | int | None) -> int | None:
    """Parse a human byte budget such as ``"64MB"`` into an integer.

    In simple words: turn "2GB" into the number of bytes you actually meant.

    Args:
        spec: ``"64MB"``, ``"2G"``, ``"1048576"``, a plain int, or ``None``.

    Returns:
        The byte count, or ``None`` if ``spec`` is ``None``.

    Raises:
        ValueError: If the string is not a number with an optional unit, or the
            unit is not one of B/KB/M/GB/TB (binary: 1 KB = 1024 B).
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"byte budget must be >= 0, got {spec}")
        return spec
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(
            f"cannot read {spec!r} as a byte budget. Use forms like '64MB', '2GB', "
            "'1048576' (units are binary: 1KB = 1024B)."
        )
    number, unit = match.groups()
    key = unit.upper()
    if key not in _UNITS:
        raise ValueError(
            f"unknown byte unit {unit!r} in {spec!r}. Use B, KB, MB, GB or TB."
        )
    value = int(float(number) * _UNITS[key])
    if value <= 0:
        raise ValueError(f"byte budget must be > 0, got {spec!r}")
    return value


def format_bytes(nbytes: float) -> str:
    """Render a byte count the way a budget was written.

    Args:
        nbytes: The count to render.

    Returns:
        A short string such as ``"64.0MB"`` or ``"912B"``.
    """
    for unit, scale in (
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
    ):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f}{unit}"
    return f"{int(nbytes)}B"


def temp_state_bytes(n_layers: int, n_heads: int, d_k: int, d_v: int) -> int:
    """Bytes of DeltaBank working state, int8, one byte per state cell.

    ``n_layers * n_heads * d_k * d_v``: each head holds a ``(d_k, d_v)`` matrix.

    Note what is *not* in here: ``n_banks``. The banks are the decay schedule --
    a convex mix over decay rates -- not memory slots, so widening B costs no
    bytes. The design-phase formula ``L * B * 2d`` assumed each bank was a slot
    and so overstated the knob's reach in one direction while assuming a key and
    value of equal width in the other.

    Args:
        n_layers: Number of DeltaBank blocks.
        n_heads: Independent memories per layer.
        d_k: Key rows per head.
        d_v: Value columns per head.

    Returns:
        Total temporary-state bytes for the model.
    """
    return int(n_layers) * int(n_heads) * int(d_k) * int(d_v)


def perm_entry_bytes(bits: int, d_value: int) -> int:
    """Bytes one VectorVault entry occupies.

    A hypervector key is ``bits // 8`` bytes, a value is a float32
    (``4 * d_value``), plus :data:`ENTRY_META_BYTES` of bookkeeping.

    Args:
        bits: Hypervector width in bits. This dominates: at the default 8192
            bits a key is 1024 B against a 128 B value.
        d_value: Value width in elements.

    Returns:
        Bytes for one stored entry.
    """
    return int(bits) // 8 + 4 * int(d_value) + ENTRY_META_BYTES
