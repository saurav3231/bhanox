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

from bhanox.quant.numerics import absmax_quantize

__all__ = [
    "BYTE_GRAM_N",
    "BYTE_GRAM_ORDER",
    "MASK64",
    "HashBind",
    "encode_bytes",
    "mix64",
]

MASK64 = (1 << 64) - 1
#: Byte 4-gram read order (architecture D5). Big-endian, so "abcd" and "bcde"
#: differ in the high bits as well as the low ones.
BYTE_GRAM_ORDER = "big"
#: Gram length the spec trains on. :func:`encode_bytes` accepts 2..8, but one
#: model input is a byte 4-gram and one output class is a single byte, so the
#: generation loop needs the trained length as a name rather than as a literal
#: it decides for itself.
BYTE_GRAM_N = 4


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


def encode_bytes(data: bytes | str, *, n: int = BYTE_GRAM_N) -> NDArray[np.int64]:
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

    Note:
        A gram at position ``i`` predicts the byte at ``i + n``, so a buffer of
        ``L`` bytes yields ``L - n + 1`` inputs but only ``L - n`` targets. The
        last gram has no target inside its own buffer and is dropped when a
        training window is cut. That off-by-one is a property of next-byte
        prediction, not a bug to paper over.
    """
    if not 2 <= n <= 8:
        raise ValueError(f"n-gram length must be in 2..8, got {n}")
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if len(raw) < n:
        return np.zeros(0, dtype=np.int64)
    # ``sliding_window_view``, not ``as_strided``. as_strided with an explicit
    # shape and no strides gave this a row stride of n bytes instead of 1, so
    # row i started at byte ``i * n`` and the last ``n - 1`` rows ran off the end
    # of the buffer. It does not bounds-check, so those ids were adjacent heap
    # memory: `encode_bytes(b"abcde")` returned ``[0x61626364, 0x65000000]`` --
    # 'a','b','c','d' correctly and then 'e' followed by whatever was next in
    # memory. Only the first id of every buffer was right, and the rest varied
    # with the allocation, so training on this was neither correct nor
    # reproducible under a fixed seed.
    view = np.lib.stride_tricks.sliding_window_view(
        np.frombuffer(raw, dtype=np.uint8), n
    )
    if BYTE_GRAM_ORDER == "big":
        weights = 256 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    else:
        weights = 256 ** np.arange(0, n, dtype=np.int64)
    return np.ascontiguousarray(view).astype(np.int64) @ weights


@dataclass
class HashBind:
    """Multi-hash embedding front-end with a small learned direct table.

    The embedding of an id is, per architecture D3::

        e(x) = Table[x] * 1[x < vocab_table] + sum_j g_j * Pool[h_j(x)]

    Representation, since it decides what every other method has to do:
    ``pool`` and ``table`` hold int8 *codes*, and ``pool_scale``/``table_scale``
    hold the per-column absmax scale that turns them back into real weights.
    :meth:`embed` applies the scale, so the array a caller reads is not the
    array the arithmetic uses. Storing the dequantized weights in ``pool``
    instead would inflate every embedding by 283x-795x (the scale is ~1.3e-3)
    and the residual-stream add that follows has nothing to absorb it; storing
    the bare codes and dropping the scale, as this did, is the same failure with
    the sign flipped. Keeping both halves named is what stops either.

    Attributes:
        d_model: Embedding width.
        pool_size: Rows in the shared hash pool.
        vocab_table: Rows in the learned direct table. Ids below this get a
            dedicated row; the rest are hashed.
        n_hashes: Number of independent hash functions summed over.
        seed: Base salt for the hash functions.
        pool: ``(pool_size, d_model)`` int8 codes for the hash pool.
        table: ``(vocab_table, d_model)`` int8 codes for the direct table.
        pool_scale: ``(d_model,)`` per-column absmax scale for ``pool``.
        table_scale: ``(d_model,)`` per-column absmax scale for ``table``.
        g: Learned per-hash mix weights, shape ``(n_hashes,)``.
    """

    d_model: int
    pool_size: int = 8192
    vocab_table: int = 1024
    n_hashes: int = 4
    seed: int = 0x9E3779B9
    pool: NDArray[np.floating] = field(init=False)
    table: NDArray[np.floating] = field(init=False)
    pool_scale: NDArray[np.floating] = field(init=False)
    table_scale: NDArray[np.floating] = field(init=False)
    g: NDArray[np.floating] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate the pool, the table and the mix weights.

        Why tiny random init rather than zeros: two ids hashing to the same
        pool row must not start with identical embeddings, or the collision is
        unrecoverable no matter how long you train.

        The scales start at 1.0 and the codes start as the real values, so an
        unquantized front-end is exact rather than merely close: ``quantize()``
        is what moves the pair from (real, 1.0) to (code, scale).
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
        self.pool_scale = np.ones(self.d_model, dtype=np.float32)
        self.table_scale = np.ones(self.d_model, dtype=np.float32)
        self.g = np.full(self.n_hashes, 1.0 / self.n_hashes, dtype=np.float32)

    @property
    def nbytes(self) -> int:
        """int8 bytes resident for the whole front-end, plus the scales.

        Why: this is the number the ~-96% embedding-memory claim rests on, and
        invariant I2 needs it to include the front-end in the audit.

        The scales are counted rather than ignored. They are ``2 * d_model``
        float32 next to ``(pool_size + vocab_table) * d_model`` int8 codes, so
        they move the total by a fraction of a percent, but a residency number
        that quietly omits an array it holds is the kind of number that stops
        being true as the model grows.
        """
        return int(self.pool.size + self.table.size) + int(
            self.pool_scale.nbytes + self.table_scale.nbytes
        )

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
        """Embed a batch of ids, in real units.

        Args:
            ids: Integer ids of any shape; the last axis is treated as the
                token axis, so ``(B, T)`` ids give ``(B, T, d_model)``.

        Returns:
            ``float32`` embeddings of shape ``ids.shape + (d_model,)``.

        The scales are applied here, once, on the gathered rows rather than on
        the whole pool: multiplying ``(n_tokens, d_model)`` by a
        ``(d_model,)`` vector is the same arithmetic as scaling the table in
        advance, and it keeps the stored array integral. Before
        :meth:`quantize` the scales are 1.0, so this is the identity.

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
        hashed *= self.pool_scale
        # The frozen D3 equation is a SUM, not a choice: a known id gets its
        # dedicated table row *in addition to* the hashed contribution, and an
        # unknown id gets only the hashed one. Replacing instead of adding would
        # quietly change what the direct table is for.
        #
        # The lower bound is not decoration: without it a negative id passes the
        # test and numpy then indexes the table from the end, so id -1 silently
        # reads the last row.
        known = (arr >= 0) & (arr < self.vocab_table)
        direct = self.table[np.where(known, arr, 0)] * self.table_scale
        return (hashed + np.where(known[..., None], direct, np.float32(0.0))).astype(
            np.float32
        )

    def quantize(self) -> None:
        """Quantize the pool and table to int8 codes, in place.

        Why: the int8 regime is the default deployment state, so the reference
        model should be able to *be* in that regime rather than merely
        describe it.

        The codes go in ``pool``/``table`` and the per-column absmax scale goes
        in ``pool_scale``/``table_scale``, and :meth:`embed` multiplies them back
        together. Keeping both halves is the point:

        - codes alone, consumed as real values, inflate the embedding by
          283x-795x (measured at nano: RMS 0.070 -> 39.5). The next thing to
          touch the array is a residual-stream add, which has no scale to absorb
          that, and the layer norm downstream rescales the damage back to
          looking merely uninformative.
        - dequantized values in the field, as this stored them, are not
          quantization at all -- it is a rescale, and it breaks the "stored
          values are exactly integral" property that invariant I3 documents and
          that a native int8 kernel can actually consume.

        Absmax quantization is idempotent here: the codes carry absmax exactly
        127, so a second call recomputes the same scale and returns the same
        codes.
        """
        pool_q = absmax_quantize(self.pool, axis=1)
        table_q = absmax_quantize(self.table, axis=1)
        self.pool = pool_q.q.astype(np.float32)
        self.table = table_q.q.astype(np.float32)
        self.pool_scale = np.asarray(pool_q.scale, dtype=np.float32).reshape(
            self.d_model
        )
        self.table_scale = np.asarray(table_q.scale, dtype=np.float32).reshape(
            self.d_model
        )

    def dequantized_pool(self) -> NDArray[np.float32]:
        """Return the pool in real units, for tests and the audit.

        Applies :attr:`pool_scale` to the stored codes, so this reconstructs the
        weights the codes represent. It is a genuine dequantize now; when
        :meth:`quantize` stored dequantized values in ``pool`` this was a
        no-op, and before that it re-quantized what was already stored.
        """
        return (self.pool * self.pool_scale).astype(np.float32)

    def param_count(self) -> int:
        """Total stored values across pool, table, scales and mix weights."""
        return int(
            self.pool.size
            + self.table.size
            + self.pool_scale.size
            + self.table_scale.size
            + self.g.size
        )
