"""The numeric grid the delta rule runs on: normalise, reciprocal, logistic.

Split out of ``bhanox.core.deltabank`` for law C5 (one concern per module).
The head in :mod:`bhanox.core.deltabank` is the mechanism -- the recurrence,
its projections, and its read gate. This is the arithmetic vocabulary that
mechanism is written in, and it is a separate concern for a concrete reason:
:mod:`bhanox.train.mirror` needs the *same* table and the *same* normalisation
as the reference head, or the two disagree about what an int8 code means. One
definition, imported by both, is what keeps them bit-compatible.

Re-exported from ``bhanox.core.deltabank``, so importers take these from there
as before. The dependency runs one way: the head imports from here, and this
module knows nothing about the head.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["l2_normalize", "recip_lut"]


def _sigmoid(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logistic function, branch-free.

    Args:
        z: Real array.

    Returns:
        ``sigmoid(z)`` in ``(0, 1)``.

    Why a lookup instead of ``exp``: inference runs in the int8 regime (I3), so
    the deployed graph uses a 256-entry LUT. The native runtime is what makes
    that exact; the reference needs to agree to 1e-3 (I4), and this is accurate
    to ~1e-7, so the two agree comfortably.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def l2_normalize(
    x: NDArray[np.floating], axis: int = -1, eps: float = 1e-6
) -> NDArray[np.floating]:
    """L2-normalise along an axis, leaving near-zero rows as zeros.

    Args:
        x: Input array.
        axis: Axis to normalise over.
        eps: Floor on the denominator, so an all-zero row returns zero rather
            than NaN. A zero key/query must contribute nothing, not poison the
            state.

    Returns:
        Array of the same shape with unit-norm rows (or zero rows).
    """
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / np.maximum(norm, eps)


def recip_lut(n_entries: int = 256, *, bits: int = 16) -> NDArray[np.float32]:
    """Reciprocal lookup table for the delta-rule step size.

    The delta rule requires ``beta = 1 / ||k||^2`` for the update to be a true
    projection. A reciprocal is a divide, and division is not in the I3
    whitelist; a 256-entry LUT is. With K L2-normalised, ``||k||^2 == 1`` and
    the table is read at index 1, so the LUT is exact in the normal case and
    only approximates the pathological one.

    Args:
        n_entries: Table size. Must be <= 256 to satisfy I3's LUT bound.
        bits: Fixed-point fraction bits when encoding the index.

    Returns:
        ``float32`` table of ``1 / i`` for ``i`` in ``1..n_entries``.

    Raises:
        ValueError: If ``n_entries`` exceeds the 256-entry LUT limit.
    """
    if n_entries > 256:
        raise ValueError(
            f"recip_lut size {n_entries} exceeds the 256-entry LUT bound (I3)"
        )
    idx = np.arange(1, n_entries + 1, dtype=np.float64)
    scale = float(1 << bits)
    return (np.rint(scale / idx) / scale).astype(np.float32)


_RECIP = recip_lut(256)
