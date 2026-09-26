"""HashBind front-end: open-vocabulary embeddings from a tiny learned table.

Purpose: give every possible input -- any unicode, any byte, any 4-gram, any
token a tokenizer has never seen -- a vector, using a small fixed pool of
vectors plus a handful of hash functions instead of a full embedding matrix.

In simple words: instead of one vector per word (a huge table), keep a small
pool of vectors and let each input pick a few of them by hash. Two unseen words
that collide share a vector, which is the ~96% memory saving and also the
failure mode the design phase measured at 0.008 BPC.

Reference: architecture spec D3 "HashBind", I3 (the mixer must be multiply-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bhanox.quant.numerics import absmax_quantize, dequantize

__all__ = ["BYTE_GRAM_ORDER", "MASK64", "HashBind", "encode_bytes", "mix64"]

MASK64 = (1 << 64) - 1
#: Byte 4-gram read order (architecture D5). Big-endian, so "abcd" and "bcde"
#: differ in the high bits as well as the low ones.
BYTE_GRAM_ORDER = "big"


def mix64(x: int, seed: int) -> int:
    """Multiply-free 64-bit avalanche mixer.

    Args:
        x: Input integer (token id or packed 4-gram).
        seed: Per-function salt, so independent hashes decorrelate.

    Returns:
        A 64-bit integer whose low bits depend on all input bits.

    Why no multiply: splitmix64's finalizer is the good mixer, but a 64-bit
    multiply is not in the I3 whitelist and on a CPU without a wide integer
    multiplier it is several shift-adds plus a carry chain. The xorshift rounds
    below give adequate low-bit diffusion for bucketing into an 8192-row pool
    at a fraction of the cost, and they are the sanctioned ceiling: the
    alternative is not "a faster multiply", it is "no multiply at all".
    """
    z = (x ^ seed) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    z = ((z << 17) & MASK64) ^ z
    z = (z ^ (z >> 13)) & MASK64
    z = ((z << 5) & MASK64) ^ z
    return (z ^ (z >> 27)) & MASK64


def _mix64_vec(x: NDArray[np.integer], seed: int) -> NDArray[np.uint64]:
    """Vectorised :func:`mix64` over an array of ids.

    Args:
        x: Integer ids, any signed/unsigned width.
        seed: Per-function salt.

    Returns:
        ``uint64`` array of mixed values.
    """
    z = x.astype(np.uint64) ^ np.uint64(seed & MASK64)
    z ^= z >> np.uint64(31)
    z = ((z << np.uint64(17)) & np.uint64(MASK64)) ^ z
    z ^= z >> np.uint64(13)
    z = ((z << np.uint64(5)) & np.uint64(MASK64)) ^ z
    return z ^ (z >> np.uint64(27))


def encode_bytes(data: bytes | str, *, n: int = 4) -> NDArray[np.int64]:
    """Turn raw text or bytes into a stream of integer n-gram ids.

    In simple words: slide a window of ``n`` bytes over the input and read each
    window as one big number. No vocabulary, no tokenizer, no UNK token, and
    it works on any unicode input because every string is bytes in UTF-8.

    Args:
        data: Text or bytes.
        n: Gram length. The spec's unit is a byte 4-gram.

    Returns:
        ``int64`` array of ``max(0, len(raw) - n + 1)`` ids, each below
        ``2**(8*n)``.

    Raises:
        ValueError: If ``n`` is not in 2..8 (ids must fit in int64).
    """
    if not 2 <= n <= 8:
        raise ValueError(f"n-gram length must be in 2..8, got {n}")
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if len(raw) < n:
        return np.zeros(0, dtype=np.int64)
    view = np.lib.stride_tricks.as_strided(
        np.frombuffer(raw, dtype=np.uint8), shape=(len(raw) - n + 1, n)
    )
    if BYTE_GRAM_ORDER == "big":
        weights = 256 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    else:
        weights = 256 ** np.arange(0, n, dtype=np.int64)
    return view.astype(np.int64) @ weights


@dataclass
class HashBind:
    """Multi-hash embedding front-end with a small learned direct table.

    The embedding of an id is, per architecture D3::

        e(x) = Table[x] * 1[x < vocab_table] + sum_j g_j * Pool[h_j(x)]

    Attributes:
        d_model: Embedding width.
        pool_size: Rows in the shared hash pool.
        vocab_table: Rows in the learned direct table. Ids below this get a
            dedicated row; the rest are hashed.
        n_hashes: Number of independent hash functions summed over.
        seed: Base salt for the hash functions.
        pool: ``(pool_size, d_model)`` int8-quantized hash pool.
        table: ``(vocab_table, d_model)`` int8-quantized direct table.
        g: Learned per-hash mix weights, shape ``(n_hashes,)``.
    """

    d_model: int
    pool_size: int = 8192
    vocab_table: int = 1024
    n_hashes: int = 4
    seed: int = 0x9E3779B9
    pool: NDArray[np.floating] = field(init=False)
    table: NDArray[np.floating] = field(init=False)
    g: NDArray[np.floating] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate the pool, the table and the mix weights.

        Why tiny random init rather than zeros: two ids hashing to the same
        pool row must not start with identical embeddings, or the collision is
        unrecoverable no matter how long you train.
        """
        if self.vocab_table > self.pool_size:
            raise ValueError("vocab_table must not exceed pool_size")
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.pool = (
            rng.standard_normal((self.pool_size, self.d_model)) * scale
        ).astype(np.float32)
        self.table = (
            rng.standard_normal((self.vocab_table, self.d_model)) * scale
        ).astype(np.float32)
        self.g = np.full(self.n_hashes, 1.0 / self.n_hashes, dtype=np.float32)

    @property
    def nbytes(self) -> int:
        """int8 bytes resident for the whole front-end.

        Why: this is the number the ~-96% embedding-memory claim rests on, and
        invariant I2 needs it to include the front-end in the audit.
        """
        return (self.pool.size + self.table.size) * 1

    def hash_rows(self, ids: NDArray[np.integer]) -> NDArray[np.int64]:
        """Map ids to their pool rows, one column per hash function.

        Args:
            ids: Integer ids, any shape.

        Returns:
            ``(..., n_hashes)`` int64 array of pool row indices.
        """
        seeds = [(self.seed * (j + 1)) & MASK64 for j in range(self.n_hashes)]
        cols = [
            _mix64_vec(np.asarray(ids), seeds[j]) % np.uint64(self.pool_size)
            for j in range(self.n_hashes)
        ]
        return np.stack(cols, axis=-1).astype(np.int64)

    def __call__(self, ids: NDArray[np.integer]) -> NDArray[np.float32]:
        """Embed ids. See :meth:`embed`."""
        return self.embed(ids)

    def embed(self, ids: NDArray[np.integer]) -> NDArray[np.float32]:
        """Embed a batch of ids.

        Args:
            ids: Integer ids of any shape; the last axis is treated as the
                token axis, so ``(B, T)`` ids give ``(B, T, d_model)``.

        Returns:
            ``float32`` embeddings of shape ``ids.shape + (d_model,)``.

        Why no ``float`` in the *inference* op set: the gathers, adds and
        scale-multiplies here are all integer-domain in the deployed int8
        graph; see :func:`bhanox.audit.audit_bytes_per_token` for the op
        listing and the I3 check that backs it.
        """
        arr = np.asarray(ids).astype(np.int64)
        rows = self.hash_rows(arr)
        hashed = np.einsum(
            "...hk,h->...k", self.pool[rows], self.g, dtype=np.float32
        ).reshape(*arr.shape, self.d_model)
        # The frozen D3 equation is a SUM, not a choice: a known id gets its
        # dedicated table row *in addition to* the hashed contribution, and an
        # unknown id gets only the hashed one. Replacing instead of adding would
        # quietly change what the direct table is for.
        #
        # The lower bound is not decoration: without it a negative id passes the
        # test and numpy then indexes the table from the end, so id -1 silently
        # reads the last row.
        known = (arr >= 0) & (arr < self.vocab_table)
        direct = self.table[np.where(known, arr, 0)]
        return (hashed + np.where(known[..., None], direct, np.float32(0.0))).astype(
            np.float32
        )

    def quantize(self) -> None:
        """Quantize the pool and table to int8, in place.

        Why: the int8 regime is the default deployment state, so the reference
        model should be able to *be* in that regime rather than merely
        describe it. Values stay exactly integral.
        """
        self.pool = absmax_quantize(self.pool, axis=0).q.astype(np.float32)
        self.table = absmax_quantize(self.table, axis=0).q.astype(np.float32)

    def dequantized_pool(self) -> NDArray[np.float32]:
        """Return the pool in real units (for tests and the audit)."""
        return dequantize(self.pool, axis=0).astype(np.float32)

    def param_count(self) -> int:
        """Total stored values across pool, table and mix weights."""
        return int(self.pool.size + self.table.size + self.g.size)
