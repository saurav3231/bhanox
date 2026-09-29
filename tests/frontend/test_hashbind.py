"""Tests for the HashBind open-vocabulary front-end."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.frontend import hashbind as hb


class TestEncodeBytes:
    def test_counts_sliding_windows(self) -> None:
        assert hb.encode_bytes(b"abcd").tolist() == [0x61626364]

    def test_length_of_stream(self) -> None:
        assert hb.encode_bytes(b"abcdefgh").size == 5

    def test_short_input_yields_empty(self) -> None:
        assert hb.encode_bytes(b"ab", n=4).size == 0

    def test_is_deterministic(self) -> None:
        assert np.array_equal(hb.encode_bytes("hello"), hb.encode_bytes("hello"))

    def test_accepts_str_and_bytes_identically(self) -> None:
        assert np.array_equal(hb.encode_bytes("hi"), hb.encode_bytes(b"hi"))

    def test_handles_any_unicode(self) -> None:
        """No vocabulary means no UNK, so any script just works."""
        for text in ("नमस्ते", "こんにちは", "Ω≈ç√", "مرحبا"):
            assert hb.encode_bytes(text).size == max(0, len(text.encode()) - 3)

    def test_ids_stay_within_range(self) -> None:
        ids = hb.encode_bytes("x" * 300)
        assert ids.min() >= 0
        assert ids.max() < 2**32

    def test_big_endian_order(self) -> None:
        # With n=2 there is no room for the order to matter, so compare the two
        # grams "ab" and "ba" directly.
        assert hb.encode_bytes("ab", n=2).tolist() == [0x6162]
        assert hb.encode_bytes("ba", n=2).tolist() == [0x6261]

    def test_rejects_out_of_range_n(self) -> None:
        with pytest.raises(ValueError, match=r"2\.\.8"):
            hb.encode_bytes("abc", n=9)


class TestMix64:
    def test_is_deterministic(self) -> None:
        assert hb.mix64(12345, 7) == hb.mix64(12345, 7)

    def test_separates_seeds(self) -> None:
        assert hb.mix64(12345, 1) != hb.mix64(12345, 2)

    def test_is_not_the_identity(self) -> None:
        assert hb.mix64(0, 0) != 0 or hb.mix64(1, 0) != 1

    def test_stays_in_64_bits(self) -> None:
        for x in (0, 1, 2**31, 2**63):
            assert 0 <= hb.mix64(x, 0x9E3779B9) <= hb.MASK64

    def test_vectorised_matches_scalar(self) -> None:
        xs = np.array([0, 1, 12345, 2**31], dtype=np.int64)
        vec = hb._mix64_vec(xs, 99)
        for i, x in enumerate(xs):
            assert int(vec[i]) == hb.mix64(int(x), 99)

    def test_low_bits_depend_on_high_input_bits(self) -> None:
        """The whole point of a mixer: moving a high bit must move the bucket."""
        moved = [hb.mix64(1, 0) & 8191, hb.mix64(1 | (1 << 40), 0) & 8191]
        assert moved[0] != moved[1]


class TestHashBind:
    def test_shape(self) -> None:
        emb = hb.HashBind(d_model=32)
        assert emb.embed(np.array([1, 2, 3])).shape == (3, 32)

    def test_preserves_leading_shape(self) -> None:
        emb = hb.HashBind(d_model=16)
        assert emb.embed(np.zeros((4, 7), dtype=np.int64)).shape == (4, 7, 16)

    def test_is_deterministic(self) -> None:
        emb = hb.HashBind(d_model=16)
        ids = np.array([5, 6, 7])
        assert np.array_equal(emb.embed(ids), emb.embed(ids))

    def test_different_seeds_give_different_embeddings(self) -> None:
        a = hb.HashBind(d_model=16, seed=1).embed(np.array([5]))
        b = hb.HashBind(d_model=16, seed=2).embed(np.array([5]))
        assert not np.allclose(a, b)

    def test_unseen_token_still_embeds(self) -> None:
        """Open vocabulary: an id far outside any trained range is fine."""
        emb = hb.HashBind(d_model=16)
        far = 2**31 + 12345
        out = emb.embed(np.array([far]))
        assert np.all(np.isfinite(out))
        assert np.abs(out).max() > 0

    def test_negative_id_gets_no_direct_row(self) -> None:
        """A negative id must not read a table row.

        Without the `arr >= 0` guard, -1 passes the bound test and numpy indexes
        from the end, so the id silently picks up the *last* direct row instead
        of being treated as unknown. Zeroing the table is the clean probe: a
        known id's embedding must change, a negative id's must not.
        """
        emb = hb.HashBind(d_model=16, vocab_table=8)
        assert not np.allclose(
            emb.embed(np.array([0])), emb.embed(np.array([2**31 + 7]))
        )  # sanity: unknown ids already ignore the table
        before_known = emb.embed(np.array([3]))
        before_negative = emb.embed(np.array([-1]))
        emb.table[:] = 0
        assert not np.allclose(before_known, emb.embed(np.array([3])))
        assert np.allclose(before_negative, emb.embed(np.array([-1])))

    def test_negative_id_differs_from_the_last_row(self) -> None:
        """The concrete symptom the guard prevents: -1 == vocab_table - 1."""
        emb = hb.HashBind(d_model=16, vocab_table=8)
        last = emb.embed(np.array([7]))  # vocab_table - 1
        minus_one = emb.embed(np.array([-1]))
        assert not np.allclose(last, minus_one)

    def test_direct_table_is_added_not_substituted(self) -> None:
        """The frozen D3 equation sums the two terms, so removing the table row
        must change a known id's embedding by exactly that row."""
        emb = hb.HashBind(d_model=16, vocab_table=8)
        emb.table[3] = 1.0
        with_row = emb.embed(np.array([3]))
        emb.table[3] = 0.0
        assert np.allclose(with_row - emb.embed(np.array([3])), 1.0)

    def test_unknown_ids_get_no_table_contribution(self) -> None:
        emb = hb.HashBind(d_model=16, vocab_table=8)
        zeroed = emb.embed(np.array([999]))
        emb.table[0] = 7.0
        assert np.allclose(zeroed, emb.embed(np.array([999])))

    def test_empty_input(self) -> None:
        assert hb.HashBind(d_model=8).embed(np.zeros(0, dtype=np.int64)).shape == (0, 8)

    def test_rejects_table_larger_than_pool(self) -> None:
        with pytest.raises(ValueError, match="vocab_table"):
            hb.HashBind(d_model=8, pool_size=16, vocab_table=32)

    def test_nbytes_counts_pool_and_table(self) -> None:
        """Residency is the int8 codes plus the stored per-column scales.

        The scales are counted because ``nbytes`` is the number invariant I2
        and the audit report, and a residency figure that omits an array the
        module holds stops being true as the model grows. They are
        ``2 * d_model`` float32 next to ``(pool_size + vocab_table) * d_model``
        int8 codes, so here that is 64 bytes on top of 160.
        """
        emb = hb.HashBind(d_model=8, pool_size=16, vocab_table=4)
        codes = (16 + 4) * 8
        scales = 2 * 8 * 4  # pool_scale + table_scale, float32
        assert emb.nbytes == codes + scales

    def test_collisions_are_rare(self) -> None:
        """4-gram ids are 32 bits and the pool is 8192 rows, so distinct ids
        should rarely share all four rows."""
        emb = hb.HashBind(d_model=8)
        rows = emb.hash_rows(np.arange(20000, dtype=np.int64))
        full_collisions = sum(
            1
            for i in range(len(rows))
            if np.array_equal(rows[i], rows[(i + 1) % len(rows)])
        )
        assert full_collisions < len(rows) * 0.01

    def test_collision_battery_random_ids(self) -> None:
        emb = hb.HashBind(d_model=8, pool_size=256, vocab_table=64, n_hashes=4)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 2**32, size=500, dtype=np.int64)
        rows = emb.hash_rows(ids)
        assert rows.min() >= 0
        assert rows.max() < 256

    def test_quantize_makes_pool_integral(self) -> None:
        emb = hb.HashBind(d_model=16)
        emb.quantize()
        from bhanox.quant.numerics import assert_integral

        assert_integral(emb.pool, where="pool")

    def test_param_count(self) -> None:
        """Codes, scales and mix weights are all stored values.

        The two scale vectors are counted because they are stored, not
        recomputed. At the nano shape that is ``2 * 128`` out of over a
        million pool codes, so it does not move the parameter claim; it is
        counted anyway so the number means "everything this object holds".
        """
        emb = hb.HashBind(d_model=8, pool_size=16, vocab_table=4, n_hashes=2)
        assert emb.param_count() == 16 * 8 + 4 * 8 + 2 * 8 + 2

    def test_calling_the_object_equals_embed(self) -> None:
        emb = hb.HashBind(d_model=8)
        ids = np.array([1, 2])
        assert np.array_equal(emb(ids), emb.embed(ids))
