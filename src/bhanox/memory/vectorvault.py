"""VectorVault: an optional hyperdimensional episodic store.

Purpose: give the model *exact* long-range recall, which a fixed-size decayed
state cannot promise. The DeltaBank is a lossy, decaying summary; the Vault is
an addressable list of things worth keeping verbatim.

In simple words: the DeltaBank is a notebook that smudges. The Vault is a
filing cabinet: you ask "which item did I store under this key?" and get it
back exactly, or not at all.

Architecture (spec D3, frozen)::

    bipolar hypervectors, 8192 bits (1 KB) per key and per value
    write: when ||e_t|| > theta_s  (salience, reusing the DeltaBank surprise)
    store: fixed M slots, eviction = lowest salience * recency
    query: Hamming top-1 via XOR + POPCOUNT
    value decodes to logits via a tiny int8 decoder

Why hypervectors: XOR is addition over GF(2) and POPCOUNT is a single CPU
instruction with an efficient AVX-512 implementation, so similarity search over
a thousand items is a thousand XOR-and-popcount operations on 128 bytes each.
That is a hardware primitive which stays cheap on in-memory-analog and
photonic substrates, which is exactly the portability requirement.

Enabled only at >= Small scale (50M+ params). Off for nano and mini, where the
1 KB per key is a bad trade against a 1 MiB L2 budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["HYPERVECTOR_BITS", "VectorVault", "bipolar_encode"]

HYPERVECTOR_BITS = 8192

#: Cached random projections, keyed by ``(d, bits, seed)``. The projection has
#: to be identical on every call -- that is what makes a hypervector a stable
#: address -- and regenerating 8192x512 Gaussian samples per write would cost
#: far more than the store itself. One entry per shape, for the process
#: lifetime. At the Vault's intended scale (small, ``d_value=512``) this is
#: ~16 MB, which is honestly larger than the 1024-slot key table; it is a
#: reference-model cost, not a deployment one, because the native runtime
#: derives the projection from a counter-based hash instead of storing it.
_PROJECTIONS: dict[tuple[int, int, int], NDArray[np.float32]] = {}


def _projection(d: int, bits: int, seed: int) -> NDArray[np.float32]:
    """Return the cached fixed random projection of shape ``(bits, d)``."""
    key = (d, bits, seed)
    if key not in _PROJECTIONS:
        rng = np.random.default_rng(seed)
        _PROJECTIONS[key] = rng.standard_normal((bits, d)).astype(np.float32)
    return _PROJECTIONS[key]


def bipolar_encode(
    x: NDArray[np.floating], bits: int = HYPERVECTOR_BITS, *, seed: int = 0x2545F491
) -> NDArray[np.uint8]:
    """Project a real vector into a bipolar hypervector.

    Args:
        x: Source vector, any shape ``(..., d)``.
        bits: Hypervector width. Must be a positive multiple of 8.
        seed: Salt for the random projection, so the same vector always encodes
            to the same hypervector within a model.

    Returns:
        ``uint8`` array of shape ``x.shape[:-1] + (bits // 8,)``, one bit per
        sign bit: 1 means +1, 0 means -1.

    Why ``sign(R x)`` and not a learned projection: the whole point of HDC is
    that similarity in the hypervector *is* similarity in the source space,
    with no training. A random projection concentrates the angle between two
    projected vectors, so nearby inputs give nearby hypervectors and distant
    ones give near-orthogonal ones. It also makes ``-x`` the exact bitwise
    complement of ``x``, which is the property that defines a bipolar code: an
    earlier version XOR-reduced a random projection gated on the *sign* of each
    source dimension, which silently threw the negative dimensions away and
    encoded ``-x`` identically to ``x``.

    Raises:
        ValueError: If ``bits`` is not a positive multiple of 8, or ``x`` has no
            feature axis.
    """
    if bits <= 0 or bits % 8:
        raise ValueError(f"bits must be a positive multiple of 8, got {bits}")
    src = np.asarray(x, dtype=np.float32)
    if src.ndim < 1 or src.shape[-1] == 0:
        raise ValueError("x must have at least one feature axis")
    projection = _projection(src.shape[-1], bits, seed)  # (bits, d)
    sign_bits = src @ projection.T > 0.0  # (..., bits)
    return np.packbits(sign_bits, axis=-1).astype(np.uint8)


def _popcount(u8: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Hamming distance: total set bits across the last axis, per row.

    Args:
        u8: ``uint8`` array. The last axis is a packed bit string; every other
            axis is independent.

    Returns:
        ``int64`` array of shape ``u8.shape[:-1]`` holding each row's bit count,
        in ``[0, u8.shape[-1] * 8]``.

    Why ``unpackbits`` rather than the usual ``* 0x0101...01 >> 56`` horizontal
    sum: that last step is a 64-bit multiply, and invariant I3 whitelists
    POPCOUNT but not MUL_I64. ``unpackbits`` is a bit shuffle the native runtime
    gets for free from POPCNT, and it is exact.
    """
    return np.unpackbits(np.asarray(u8, dtype=np.uint8), axis=-1).sum(
        axis=-1, dtype=np.int64
    )


@dataclass
class VectorVault:
    """Fixed-slot HDC episodic store with Hamming top-1 retrieval.

    Attributes:
        n_slots: Fixed capacity M. Never grows -- that is the point.
        d_value: Width of a stored value vector.
        bits: Hypervector width. 8192 (1 KB) per key and per value.
        theta_s: Salience threshold. Writes below it are dropped.
        keys: ``(n_slots, bits // 8)`` packed hypervectors.
        values: ``(n_slots, d_value)`` stored values.
        salience: ``(n_slots,)`` salience score per slot.
        age: ``(n_slots,)`` write timestamp per slot.
        filled: Number of occupied slots.
        clock: Monotonic write counter, used for recency in eviction.
    """

    n_slots: int = 1024
    d_value: int = 32
    bits: int = HYPERVECTOR_BITS
    theta_s: float = 0.5
    keys: NDArray[np.uint8] = field(init=False)
    values: NDArray[np.floating] = field(init=False)
    salience: NDArray[np.float32] = field(init=False)
    age: NDArray[np.int64] = field(init=False)
    filled: int = 0
    clock: int = 0

    def __post_init__(self) -> None:
        """Allocate the fixed slots. The vault never reallocates."""
        if self.n_slots <= 0 or self.d_value <= 0:
            raise ValueError("n_slots and d_value must be > 0")
        if self.bits % 8:
            raise ValueError("bits must be a multiple of 8")
        n_bytes = self.bits // 8
        self.keys = np.zeros((self.n_slots, n_bytes), dtype=np.uint8)
        self.values = np.zeros((self.n_slots, self.d_value), dtype=np.float32)
        self.salience = np.zeros(self.n_slots, dtype=np.float32)
        self.age = np.zeros(self.n_slots, dtype=np.int64)

    # -- capacity ------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """Total bytes resident for the vault."""
        return int(
            self.keys.nbytes
            + self.values.nbytes
            + self.salience.nbytes
            + self.age.nbytes
        )

    def is_full(self) -> bool:
        """Whether every slot is occupied."""
        return self.filled >= self.n_slots

    # -- write ---------------------------------------------------------------

    def write(
        self,
        key_vec: NDArray[np.floating],
        value: NDArray[np.floating],
        *,
        surprise: float = 1.0,
    ) -> bool:
        """Store a value if it is salient enough, evicting if necessary.

        Args:
            key_vec: Cue vector to encode as the retrieval key.
            value: Value vector to store verbatim.
            surprise: The DeltaBank's surprise for this token, ``||e_t||``.
                Reused rather than recomputed: the core already paid for it.

        Returns:
            True if a write happened.

        Raises:
            ValueError: If the value width does not match ``d_value``.
        """
        val = np.asarray(value, dtype=np.float32).reshape(-1)
        if val.size != self.d_value:
            raise ValueError(
                f"VectorVault expected d_value={self.d_value}, got {val.size}"
            )
        # The gate is on *surprise*, per spec D3. Gating on the value's own norm
        # instead would mean a large but entirely predictable value -- exactly
        # the thing the DeltaBank does not need to remember -- occupies a slot,
        # while a small but genuinely novel one is discarded.
        if float(surprise) < self.theta_s:
            return False
        slot = self._select_slot()
        self.keys[slot] = bipolar_encode(key_vec, self.bits)
        self.values[slot] = val
        self.salience[slot] = float(surprise)
        self.clock += 1
        self.age[slot] = self.clock
        self.filled = min(self.filled + 1, self.n_slots)
        return True

    def _select_slot(self) -> int:
        """Pick a slot: a free one, else the weakest salience x recency.

        Returns:
            A slot index.

        Why the product: pure salience eviction throws away old-but-important
        memories; pure recency throws away important old memories. Their
        product keeps items that are either still surprising or still recent,
        which matched the design-phase recall target of 100% at 1024 items with
        10% noise.
        """
        if self.filled < self.n_slots:
            return self.filled
        recency = (self.age.astype(np.float64) + 1.0) / (self.clock + 1.0)
        return int(np.argmin(self.salience.astype(np.float64) * recency))

    # -- query ---------------------------------------------------------------

    def query(
        self, key_vec: NDArray[np.floating], *, top_k: int = 1
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Hamming top-k retrieval via XOR + POPCOUNT.

        Args:
            key_vec: Cue vector.
            top_k: How many slots to return.

        Returns:
            ``(indices, similarities)``. Similarity is in ``[0, 1]``, 1 being
            an exact hypervector match. Empty vault returns empty arrays.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.filled == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        n_bytes = self.bits // 8
        probe = bipolar_encode(key_vec, self.bits)
        distance = _popcount(np.bitwise_xor(self.keys[: self.filled], probe))
        similarity = 1.0 - distance / float(n_bytes * 8)
        k = min(top_k, self.filled)
        order = np.argsort(-similarity, kind="stable")[:k]
        return order.astype(np.int64), similarity[order]

    def retrieve(self, key_vec: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Return the nearest stored value, or None if the vault is empty.

        Args:
            key_vec: Cue vector.

        Returns:
            The stored value vector, or None.
        """
        idx, _ = self.query(key_vec, top_k=1)
        return None if idx.size == 0 else self.values[idx[0]].copy()

    # -- introspection -------------------------------------------------------

    def recall_at(
        self, n_items: int, n_probes: int, *, noise: float = 0.1, seed: int = 0
    ) -> float:
        """Measure cued recall: fraction of probes returning their own value.

        Args:
            n_items: How many items to store (must fit in ``n_slots``).
            n_probes: How many of them to probe.
            noise: Std-dev of Gaussian noise added to each probe key, as a
                fraction of the key's scale.
            seed: RNG seed, so the measurement is reproducible.

        Returns:
            Recall in ``[0, 1]``.

        Why this exists: the Vault's whole claim is exact recall, so it needs a
        number that can be falsified, not a demonstration that looks right.
        """
        if n_items > self.n_slots:
            raise ValueError(f"n_items={n_items} exceeds n_slots={self.n_slots}")
        rng = np.random.default_rng(seed)
        keys = rng.standard_normal((n_items, self.d_value)).astype(np.float32)
        values = rng.standard_normal((n_items, self.d_value)).astype(np.float32)
        for i in range(n_items):
            self.write(keys[i], values[i], surprise=1.0)
        n_probes = min(n_probes, n_items)
        probe_idx = rng.choice(n_items, size=n_probes, replace=False)
        hits = 0
        for i in probe_idx:
            noisy = keys[i] + noise * rng.standard_normal(self.d_value)
            got = self.retrieve(noisy.astype(np.float32))
            if got is not None and np.allclose(got, values[i], atol=0.25):
                hits += 1
        return hits / max(1, n_probes)
