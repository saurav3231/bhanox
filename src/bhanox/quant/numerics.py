"""Numerics: int8 quantization by default, ternary as an opt-in regime.

Purpose: keep every weight and activation inside a small integer domain so the
inference graph contains no floating-point arithmetic (invariant I3) and stays
mappable to hardware where the only cheap primitive is low-precision integer
accumulate.

In simple words: store the numbers as small whole numbers, not decimals, so the
CPU (and future analog/photonic hardware) can do the arithmetic cheaply.

Reference: architecture spec D3 "Numerics", C9, I3.

Storage note (honesty, spec A3): the NumPy reference stores these values in
``float32``/``float64`` arrays, but the *values are always exactly integral*
and every operation is an integer-domain add/sub/shift/small-multiply. That is
what makes the reference usable as the source of truth for the math while the
native runtime (M4) executes the identical graph on int8 registers. The
assertions in :func:`assert_integral` make "exactly integral" a checked
property, not a claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "INT8_HI",
    "INT8_LO",
    "INT8_MAX",
    "Q_BITS",
    "Q_ONE",
    "TERNARY_LEVELS",
    "QuantizedTensor",
    "absmax_quantize",
    "assert_integral",
    "dequantize",
    "from_q",
    "pack_int4",
    "pack_ternary",
    "qmul",
    "quantize_activation",
    "saturate_int8",
    "ste_quantize",
    "ste_round",
    "ternary_quantize",
    "to_q",
    "unpack_int4",
    "unpack_ternary",
]

INT8_MAX = 127.0
INT4_MAX = 7.0
#: Ternary alphabet. Two bits, four codes, three used.
TERNARY_LEVELS = (-1.0, 0.0, 1.0)


@dataclass(frozen=True)
class QuantizedTensor:
    """An integer tensor plus the scale needed to undo quantization.

    Attributes:
        q: Quantized values. Integral, in ``[-2**(bits-1), 2**(bits-1)-1]``.
        scale: Per-column absmax scale, broadcastable against ``q``.
        bits: Bit width of ``q`` (4 or 8).
        zero_point: Integer that maps to real 0.0. Absmax symmetric
            quantization uses 0.
    """

    q: NDArray[np.floating]
    scale: NDArray[np.floating]
    bits: int = 8
    zero_point: int = 0

    @property
    def nbytes(self) -> int:
        """Storage cost in bytes, assuming the stated bit width.

        Why: invariant I2 counts bytes touched, so the quantizer must be able
        to report its own footprint honestly.
        """
        return int(self.q.size * self.bits // 8)

    def dequantize(self) -> NDArray[np.floating]:
        """Return the approximate real-valued tensor."""
        return self.q * self.scale + self.zero_point

    def astype_int(self) -> NDArray[np.integer]:
        """Return the integer codes, for a native runtime to consume.

        Why always ``int8``: numpy has no 4-bit dtype, so ``np.int4`` does not
        exist and naming it raised ``AttributeError`` for any sub-byte tensor.
        Sub-byte widths are carried in an int8 and packed by :func:`pack_int4`
        (or :func:`pack_ternary`), which is where the bit layout belongs.
        """
        return self.q.astype(np.int8)


def assert_integral(values: NDArray[np.floating], *, where: str = "tensor") -> None:
    """Assert an array holds exactly integral values.

    Args:
        values: Array to check.
        where: Label used in the failure message.

    Raises:
        AssertionError: If any value has a fractional part.
    """
    if not np.all(values == np.rint(values)):
        bad = int(np.count_nonzero(values != np.rint(values)))
        raise AssertionError(
            f"{where}: {bad} value(s) are not integral -- the int8 regime "
            "leaked a floating-point value into the inference path (I3)"
        )


# --- Fixed point ------------------------------------------------------------
#
# The recurrent state is int8. Multiplying an int8 state by a fractional decay
# rate is the one place the recurrence would silently reintroduce floating
# point, and a native kernel cannot do it. So rates, step sizes and the
# read-gate all live in Q16 fixed point and every recurrence step is
# (multiply, arithmetic shift, saturate) -- three whitelisted I3 ops.
#
# The carrier dtype is int32 rather than the spec's int16 accumulator because
# an int8 x int32 product needs 40 bits before the shift; saturating to int8 on
# the way back out is what bounds it. This is a reference-implementation
# detail, not a change to the arithmetic (I4 still holds).

Q_BITS = 16
Q_ONE = 1 << Q_BITS
INT8_LO = -128
INT8_HI = 127


def to_q(x: NDArray[np.floating] | float) -> NDArray[np.int32]:
    """Convert real values to Q16 fixed point.

    Args:
        x: Real value(s) expected in roughly ``[-1, 1]`` for rates, or any
            magnitude for step sizes.

    Returns:
        ``int32`` array of ``round(x * 2**16)``.
    """
    return np.rint(np.asarray(x, dtype=np.float64) * Q_ONE).astype(np.int32)


def from_q(q: NDArray[np.integer] | int) -> NDArray[np.float32]:
    """Convert Q16 fixed point back to real values.

    Args:
        q: Fixed-point value(s).

    Returns:
        ``float32`` array of ``q / 2**16``.
    """
    return (np.asarray(q, dtype=np.float64) / Q_ONE).astype(np.float32)


def qmul(a_q: NDArray[np.integer], b_q: NDArray[np.integer]) -> NDArray[np.int32]:
    """Multiply two Q16 values, shifting back down.

    Args:
        a_q, b_q: Q16 fixed-point operands.

    Returns:
        ``int32`` Q16 product, ``(a * b) >> 16``.
    """
    return (np.asarray(a_q, dtype=np.int32) * np.asarray(b_q, dtype=np.int32)) >> (
        Q_BITS
    )


def saturate_int8(q: NDArray[np.integer]) -> NDArray[np.int32]:
    """Clamp a fixed-point or integer array into the int8 domain.

    Args:
        q: Values to clamp.

    Returns:
        ``int32`` array with every element in ``[-128, 127]``.

    Why saturation and not wraparound: a wrapped int8 turns a large positive
    accumulator into a large negative one, which is a silent, unbounded
    correctness bug. Clamping is lossy but monotone, so the error stays bounded
    and visible. This is the ``SAT`` op in the I3 whitelist.
    """
    return np.clip(np.asarray(q, dtype=np.int32), INT8_LO, INT8_HI)


def quantize_activation(x: NDArray[np.floating]) -> NDArray[np.int32]:
    """Quantise activations to the int8 grid, unit range.

    Args:
        x: Real activations, expected roughly in ``[-1, 1]`` (all Bhanox
            activations pass through a layer norm first, so this holds).

    Returns:
        ``int32`` array holding int8 codes in ``[-127, 127]``.

    Why ``-127`` and not ``-128``: the quantiser is symmetric, so the two
    ranges must match. Spending a level on ``-128`` with no positive
    counterpart only adds quantisation noise.
    """
    return np.clip(
        np.rint(np.asarray(x, dtype=np.float64) * INT8_MAX), -INT8_MAX, INT8_MAX
    ).astype(np.int32)


def absmax_quantize(
    weight: NDArray[np.floating], *, bits: int = 8, axis: int = 0
) -> QuantizedTensor:
    """Symmetric per-column absmax quantization (the default regime).

    Args:
        weight: Real-valued tensor to quantize.
        bits: 4 or 8.
        axis: Axis treated as the column axis; the scale is reduced over the
            remaining axes so each column gets its own scale.

    Returns:
        A :class:`QuantizedTensor` whose ``dequantize()`` approximates
        ``weight`` to within one quantization step per column.

    Why per-column and not per-tensor: activations after a LayerNorm-ish
        projection have wildly different ranges per output channel, and one
        shared scale would give sparse channels almost no resolution.
    """
    limit = INT8_MAX if bits == 8 else INT4_MAX
    if weight.size == 0:
        raise ValueError("cannot quantize an empty tensor")
    keep = tuple(i for i in range(weight.ndim) if i != axis)
    absmax = np.max(np.abs(weight), axis=keep, keepdims=True)
    scale = np.where(absmax > 0, absmax / limit, 1.0)
    q = np.rint(weight / scale)
    q = np.clip(q, -limit - 1, limit)
    assert_integral(q, where="absmax_quantize")
    return QuantizedTensor(q, scale, bits=bits)


def dequantize(
    weight: NDArray[np.floating], *, bits: int = 8, axis: int = 0
) -> NDArray[np.floating]:
    """Quantize then immediately reconstruct. Convenience for tests.

    Args:
        weight: Real-valued tensor.
        bits: 4 or 8.
        axis: Column axis for the scale.

    Returns:
        The reconstructed tensor.
    """
    return absmax_quantize(weight, bits=bits, axis=axis).dequantize()


def ste_round(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Round with a straight-through estimator.

    In the forward pass this is ``rint(x)``. The backward pass of a quantizer
    should pass the gradient straight through, which is exactly what a
    detach-and-re-add formulation expresses: the value is rounded, but the
    ``x`` term keeps the graph connected so ``d(ste_round)/dx == 1``.

    Args:
        x: Input tensor.

    Returns:
        Rounded values, differentiable with unit gradient.
    """
    return x + (np.rint(x) - x)


def ste_quantize(
    x: NDArray[np.floating], *, bits: int = 8, axis: int = -1
) -> NDArray[np.floating]:
    """Quantize with a straight-through gradient.

    Args:
        x: Input tensor.
        bits: 4 or 8.
        axis: Column axis for the scale.

    Returns:
        Dequantized values, numerically identical to
        :func:`absmax_quantize(...).dequantize()` but with unit gradient.
    """
    qt = absmax_quantize(x, bits=bits, axis=axis)
    return ste_round(x / qt.scale) * qt.scale


def ternary_quantize(
    weight: NDArray[np.floating], *, threshold: float = 0.0
) -> NDArray[np.floating]:
    """Ternary {-1, 0, +1} quantization with a straight-through gradient.

    Args:
        weight: Real-valued tensor.
        threshold: Magnitude below which a weight is sent to zero. 0.0 is the
            plain sign rule; raising it trades density for magnitude fidelity.

    Returns:
        A tensor of values in {-1, 0, +1} with unit gradient.

    Why the STE matters here: the sign function has zero derivative almost
    everywhere, so without it ternary training has no gradient at all. This is
    the mechanism that lets the opt-in regime train on a short schedule.
    """
    magnitude = np.abs(weight)
    sign = np.where(weight >= 0, 1.0, -1.0)
    hard = np.where(magnitude > threshold, sign, 0.0)
    return weight + (hard - weight)


def pack_ternary(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Pack a ternary tensor at 4 values per byte (2 bits each).

    Args:
        values: Tensor of values in {-1, 0, +1}.

    Returns:
        ``uint8`` array with the last axis divided by 4.

    Raises:
        ValueError: If the last axis is not divisible by 4 or a value is not
            in the ternary alphabet.
    """
    codes = ternary_to_codes(values)
    if codes.shape[-1] % 4:
        raise ValueError(
            f"last axis must be divisible by 4 to pack 2-bit values, got "
            f"{codes.shape[-1]}"
        )
    groups = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4)
    # A 2-bit code already occupies its own 2-bit field, so the weights are
    # 1 << (2*j). Shifting by an extra bit pushes the last code out of the byte.
    weights = (1 << (2 * np.arange(4, dtype=np.uint8))).astype(np.uint8)
    return np.sum(groups * weights, axis=-1, dtype=np.uint8)


def unpack_ternary(packed: NDArray[np.uint8]) -> NDArray[np.floating]:
    """Inverse of :func:`pack_ternary`.

    Args:
        packed: ``uint8`` array, 4 ternary values per byte.

    Returns:
        ``float32`` tensor of values in {-1, 0, +1}.
    """
    shifts = np.arange(4, dtype=np.uint8) * 2
    codes = (packed[..., None] >> shifts) & np.uint8(0b11)
    return codes_to_ternary(codes.astype(np.float64)).reshape(*packed.shape[:-1], -1)


def ternary_to_codes(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Map {-1, 0, +1} to 2-bit codes {0, 1, 2}.

    The code is the index into :data:`TERNARY_LEVELS`, so the packing and the
    unpacking cannot drift apart -- an earlier version had two independent
    lookup tables and they disagreed about what code 1 meant.

    Args:
        values: Ternary tensor.

    Returns:
        ``uint8`` codes.

    Raises:
        ValueError: If a value is outside the ternary alphabet.
    """
    v = np.asarray(values)
    if np.any((v != 0) & (v != -1) & (v != 1)):
        raise ValueError("pack_ternary requires values in {-1, 0, +1}")
    return np.where(v < 0, 0, np.where(v > 0, 2, 1)).astype(np.uint8)


def codes_to_ternary(codes: NDArray[np.floating]) -> NDArray[np.floating]:
    """Map 2-bit codes {0, 1, 2} back to {-1, 0, +1}."""
    return np.take(np.array(TERNARY_LEVELS, dtype=np.float64), codes.astype(np.intp))


def pack_int4(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Pack signed int4 weights at 2 per byte.

    Args:
        values: Integral values in ``[-8, 7]``.

    Returns:
        ``uint8`` array with the last axis halved.

    Raises:
        ValueError: If a value is outside ``[-8, 7]``.
    """
    v = np.asarray(values)
    if v.shape[-1] % 2:
        raise ValueError(
            f"last axis must be even to pack two int4 per byte, got {v.shape[-1]}"
        )
    if np.any(v < -8) or np.any(v > 7):
        raise ValueError("pack_int4 requires values in [-8, 7]")
    # Two values share a byte: the low nibble holds the even element, the high
    # nibble the odd one. Each is masked to 4 bits of two's complement, so
    # sign is a property of the nibble -- which is what makes int4 signed.
    nib = v.reshape(*v.shape[:-1], v.shape[-1] // 2, 2).astype(np.int16) & 0x0F
    lo = nib[..., 0].astype(np.uint8)
    hi = nib[..., 1].astype(np.uint8)
    return lo | (hi << np.uint8(4))


def unpack_int4(packed: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Inverse of :func:`pack_int4`.

    Args:
        packed: ``uint8`` array, 2 int4 values per byte.

    Returns:
        ``int8`` array of the original signed values.
    """
    p = np.asarray(packed, dtype=np.uint8)
    lo = (p & np.uint8(0x0F)).astype(np.int16)
    hi = ((p >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int16)
    nibbles = np.stack([lo, hi], axis=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    return np.where(nibbles > 7, nibbles - 16, nibbles).astype(np.int8)
