"""Tests for the optional VectorVault HDC store."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.memory.vectorvault import (
    HYPERVECTOR_BITS,
    VectorVault,
    _popcount,
    bipolar_encode,
)

D = 32
BITS = 1024


def vault(n_slots: int = 64, **kwargs) -> VectorVault:
    """A small vault with a small hypervector, so the tests stay fast."""
    return VectorVault(n_slots=n_slots, d_value=D, bits=BITS, **kwargs)


def unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(D).astype(np.float32)
    return v / np.linalg.norm(v)


class TestBipolarEncode:
    def test_shape_and_dtype(self) -> None:
        assert bipolar_encode(np.ones(D, np.float32), BITS).shape == (BITS // 8,)
        assert bipolar_encode(np.ones(D, np.float32), BITS).dtype == np.uint8

    def test_leading_shape_is_preserved(self) -> None:
        out = bipolar_encode(np.ones((3, 4, D), np.float32), BITS)
        assert out.shape == (3, 4, BITS // 8)

    def test_deterministic(self) -> None:
        x = unit(np.random.default_rng(0))
        assert np.array_equal(bipolar_encode(x, BITS), bipolar_encode(x, BITS))

    def test_negation_is_the_bitwise_complement(self) -> None:
        """The defining property of a bipolar code. An encoder that dropped the
        negative dimensions would pass a similarity test and fail this one."""
        x = unit(np.random.default_rng(1))
        assert np.array_equal(bipolar_encode(-x, BITS), 255 - bipolar_encode(x, BITS))

    def test_near_vectors_are_closer_than_distant_ones(self) -> None:
        rng = np.random.default_rng(2)
        x = unit(rng)
        near = bipolar_encode(
            (x + 0.05 * rng.standard_normal(D)).astype(np.float32), BITS
        )
        far = bipolar_encode(unit(rng), BITS)
        d_near = int(_popcount(np.bitwise_xor(bipolar_encode(x, BITS), near)))
        d_far = int(_popcount(np.bitwise_xor(bipolar_encode(x, BITS), far)))
        assert d_near < d_far / 2

    def test_default_width_is_1kb(self) -> None:
        assert bipolar_encode(np.ones(D, np.float32)).shape == (HYPERVECTOR_BITS // 8,)

    def test_rejects_bad_width(self) -> None:
        with pytest.raises(ValueError, match="multiple of 8"):
            bipolar_encode(np.ones(D, np.float32), 7)

    def test_rejects_empty_feature_axis(self) -> None:
        with pytest.raises(ValueError, match="feature axis"):
            bipolar_encode(np.zeros((0,), np.float32), BITS)


class TestPopcount:
    def test_counts_total_bits_per_row(self) -> None:
        assert _popcount(np.array([[0x00, 0xFF, 0x0F]], np.uint8)).tolist() == [12]

    def test_drops_the_byte_axis(self) -> None:
        assert _popcount(np.zeros((5, 7), np.uint8)).shape == (5,)

    def test_agrees_with_a_naive_count(self) -> None:
        rng = np.random.default_rng(3)
        u = rng.integers(0, 256, size=(4, 9), dtype=np.uint8)
        want = np.array([sum(bin(int(v)).count("1") for v in row) for row in u])
        assert np.array_equal(_popcount(u), want)


class TestCapacity:
    def test_starts_empty(self) -> None:
        v = vault()
        assert v.filled == 0 and not v.is_full()

    def test_rejects_bad_sizes(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            VectorVault(n_slots=0)

    def test_rejects_misaligned_hypervector(self) -> None:
        with pytest.raises(ValueError, match="multiple of 8"):
            VectorVault(n_slots=4, d_value=D, bits=7)

    def test_nbytes_counts_every_buffer(self) -> None:
        v = vault(n_slots=8)
        want = 8 * (BITS // 8) + 8 * D * 4 + 8 * 4 + 8 * 8
        assert v.nbytes == want

    def test_never_grows_past_capacity(self) -> None:
        v = vault(n_slots=4)
        rng = np.random.default_rng(4)
        for _ in range(20):
            v.write(unit(rng), unit(rng), surprise=2.0)
        assert v.filled == 4
        assert v.is_full()
        assert v.keys.shape == (4, BITS // 8)


class TestWrite:
    def test_surprise_gates_the_write(self) -> None:
        v = vault()
        assert (
            v.write(
                unit(np.random.default_rng(5)), np.ones(D, np.float32), surprise=0.01
            )
            is False
        )
        assert v.filled == 0

    def test_accepts_a_surprising_write(self) -> None:
        v = vault()
        assert (
            v.write(
                unit(np.random.default_rng(6)), np.ones(D, np.float32), surprise=2.0
            )
            is True
        )
        assert v.filled == 1

    def test_large_but_unsurprising_value_is_dropped(self) -> None:
        """The gate is on surprise, not on the value's own magnitude."""
        v = vault()
        big = np.full(D, 1e3, dtype=np.float32)
        assert v.write(unit(np.random.default_rng(7)), big, surprise=0.01) is False

    def test_stores_value_verbatim(self) -> None:
        v = vault()
        val = np.arange(D, dtype=np.float32)
        v.write(unit(np.random.default_rng(8)), val, surprise=2.0)
        assert np.array_equal(v.values[0], val)

    def test_records_salience_and_age(self) -> None:
        v = vault()
        for i, s in enumerate((3.0, 5.0)):
            v.write(
                unit(np.random.default_rng(9 + i)), np.ones(D, np.float32), surprise=s
            )
        assert v.salience[:2].tolist() == [3.0, 5.0]
        assert v.age[:2].tolist() == [1, 2]

    def test_rejects_wrong_value_width(self) -> None:
        with pytest.raises(ValueError, match="d_value"):
            vault().write(np.ones(D, np.float32), np.ones(5, np.float32), surprise=2.0)


class TestQuery:
    def test_empty_vault_returns_nothing(self) -> None:
        idx, sim = vault().query(unit(np.random.default_rng(10)))
        assert idx.size == 0 and sim.size == 0

    def test_exact_key_returns_its_own_slot(self) -> None:
        v = vault()
        rng = np.random.default_rng(11)
        key = unit(rng)
        v.write(key, np.arange(D, dtype=np.float32), surprise=2.0)
        idx, sim = v.query(key)
        assert idx.tolist() == [0]
        assert sim[0] == pytest.approx(1.0)

    def test_top_k_is_respected(self) -> None:
        v = vault()
        rng = np.random.default_rng(12)
        for _ in range(5):
            v.write(unit(rng), np.ones(D, np.float32), surprise=2.0)
        idx, sim = v.query(unit(rng), top_k=3)
        assert idx.size == 3
        assert sim.tolist() == sorted(sim.tolist(), reverse=True)

    def test_top_k_larger_than_filled(self) -> None:
        v = vault()
        v.write(unit(np.random.default_rng(13)), np.ones(D, np.float32), surprise=2.0)
        assert v.query(unit(np.random.default_rng(14)), top_k=99)[0].size == 1

    def test_retrieve_returns_none_when_empty(self) -> None:
        assert vault().retrieve(unit(np.random.default_rng(15))) is None

    def test_retrieve_round_trips(self) -> None:
        v = vault()
        rng = np.random.default_rng(16)
        key, val = unit(rng), unit(rng)
        v.write(key, val, surprise=2.0)
        got = v.retrieve(key)
        assert got is not None and np.allclose(got, val)


class TestEviction:
    def test_evicts_the_weakest_salience_times_recency(self) -> None:
        """With two live slots, the third write must land on whichever slot has
        the lower salience x recency product -- checked on the stored values, not
        via a query, because the evicted key is gone and a query would just
        return the wrong nearest neighbour."""
        v = vault(n_slots=2)
        keys = [unit(np.random.default_rng(20 + i)) for i in range(3)]
        vals = [np.full(D, float(i + 1), np.float32) for i in range(3)]
        v.write(keys[0], vals[0], surprise=5.0)
        v.write(keys[1], vals[1], surprise=1.0)
        v.write(keys[2], vals[2], surprise=2.0)
        # Slot 1 was the least salient and the older of the two, so it loses.
        assert np.array_equal(v.values[0], vals[0])
        assert np.array_equal(v.values[1], vals[2])

    def test_a_recent_important_item_survives(self) -> None:
        """The product, not the minimum: an old but very salient item beats a
        recent but insignificant one."""
        v = vault(n_slots=2)
        rng = np.random.default_rng(30)
        keys = [unit(rng) for _ in range(3)]
        vals = [np.full(D, float(i + 1), np.float32) for i in range(3)]
        v.write(keys[0], vals[0], surprise=9.0)
        v.write(keys[1], vals[1], surprise=0.6)
        v.write(keys[2], vals[2], surprise=9.0)
        # Slot 0 is old but very salient; slot 1 is recent but insignificant.
        # The product picks the insignificant one, so slot 0 survives.
        assert np.array_equal(v.values[0], vals[0])
        assert np.array_equal(v.values[1], vals[2])


class TestRecall:
    def test_exact_probes_all_hit(self) -> None:
        v = vault(n_slots=32)
        assert v.recall_at(16, 16, noise=0.0) == 1.0

    def test_noise_still_recovers_most_items(self) -> None:
        """The Vault's claim is exact recall; 10% key noise should still be
        mostly recoverable, because that is what the hypervector buys."""
        v = vault(n_slots=64)
        assert v.recall_at(32, 32, noise=0.1) > 0.7

    def test_is_deterministic(self) -> None:
        v = vault(n_slots=32)
        first = v.recall_at(16, 8)
        assert first == vault(n_slots=32).recall_at(16, 8)

    def test_stored_keys_use_the_whole_hypervector(self) -> None:
        """Regression: an encoder output indexed with ``[0]`` stored only the
        first byte, which collapses 1024 bits of address into 256 possible keys
        and makes unrelated items collide."""
        v = vault(n_slots=16)
        rng = np.random.default_rng(40)
        for _ in range(16):
            v.write(unit(rng), unit(rng), surprise=2.0)
        assert len({row.tobytes() for row in v.keys[:16]}) == 16

    def test_rejects_overflowing_item_count(self) -> None:
        with pytest.raises(ValueError, match="n_items"):
            vault(n_slots=8).recall_at(9, 4)
