"""Tests for autoregressive generation.

The unit contract these pin: one model input is a packed byte 4-gram, one model
output class is a single byte, and the loop rolls the window between them. The
rolling itself is pinned in ``test_gram_contract.py``, which watches what the
model is fed; this module covers the public surface around it -- types, lengths,
determinism, reset behaviour, and refusals.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.frontend.hashbind import BYTE_GRAM_N, encode_bytes
from bhanox.generate import generate, generate_ids
from bhanox.model import Bhanox

CFG = load_config("nano")
PROMPT = b"hello world"
PROMPT_IDS = encode_bytes(PROMPT)


@pytest.fixture(scope="module")
def model() -> Bhanox:
    return Bhanox(CFG)


class TestByteApi:
    def test_returns_bytes(self, model: Bhanox) -> None:
        assert isinstance(generate(model, PROMPT, max_new=5), bytes)

    def test_length_is_prompt_plus_new(self, model: Bhanox) -> None:
        assert len(generate(model, PROMPT, max_new=5)) == len(PROMPT) + 5

    def test_prompt_is_a_prefix(self, model: Bhanox) -> None:
        out = generate(model, PROMPT, max_new=5)
        assert out[: len(PROMPT)] == PROMPT

    def test_generated_bytes_are_byte_values(self, model: Bhanox) -> None:
        """Not a promise of text: the model emits byte values, and nothing in
        the return type claims they decode."""
        tail = np.frombuffer(generate(model, PROMPT, max_new=16), dtype=np.uint8)
        assert tail.min() >= 0 and tail.max() < 256

    def test_zero_new_tokens_is_the_prompt(self, model: Bhanox) -> None:
        assert generate(model, PROMPT, max_new=0) == PROMPT

    def test_zero_new_tokens_does_not_touch_the_model(self, model: Bhanox) -> None:
        """A no-op query, so it does no model work -- including no reset."""
        model.reset()
        before = model.n_forward
        assert generate(model, PROMPT, max_new=0) == PROMPT
        assert model.n_forward == before

    def test_the_shortest_prompt_is_one_context(self, model: Bhanox) -> None:
        """Four bytes is the floor: fewer form no context at all."""
        out = generate(model, b"abcd", max_new=3, seed=1)
        assert len(out) == 7
        assert out[:4] == b"abcd"

    def test_a_bytearray_prompt_is_accepted(self, model: Bhanox) -> None:
        assert generate(model, bytearray(PROMPT), max_new=4, seed=1) == generate(
            model, PROMPT, max_new=4, seed=1
        )

    def test_a_memoryview_prompt_is_accepted(self, model: Bhanox) -> None:
        assert generate(model, memoryview(PROMPT), max_new=4, seed=1) == generate(
            model, PROMPT, max_new=4, seed=1
        )


class TestRefusals:
    def test_rejects_a_short_prompt(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="at least 4 bytes"):
            generate(model, b"abc", max_new=4)

    def test_rejects_an_empty_prompt(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="at least 4 bytes"):
            generate(model, b"", max_new=4)

    def test_rejects_a_negative_count(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="max_new"):
            generate(model, PROMPT, max_new=-1)

    def test_rejects_a_str_prompt(self, model: Bhanox) -> None:
        """No text wrapper. Encoding here quietly would invite reading the
        output as text, and the output is not promised to be text."""
        with pytest.raises(TypeError, match="bytes prompt"):
            generate(model, "hello world", max_new=4)  # type: ignore[arg-type]

    def test_rejects_a_head_that_is_not_one_class_per_byte(self) -> None:
        """A valid config is not necessarily a byte-predicting one.

        ``BhanoxConfig`` only asks for ``output_vocab > 0``, so this is
        constructible, and a 1000-way head would silently yield 1000-way
        nonsense bytes if nothing checked.
        """
        cfg = dataclasses.replace(CFG, output_vocab=1000)
        # dataclasses.replace runs __post_init__, so reaching here means the
        # config really is constructible -- valid, but not byte-predicting.
        assert cfg.output_vocab == 1000
        with pytest.raises(ValueError, match="output_vocab=256"):
            generate(Bhanox(cfg), PROMPT, max_new=4)

    def test_the_low_level_api_checks_the_head_too(self) -> None:
        cfg = dataclasses.replace(CFG, output_vocab=1000)
        with pytest.raises(ValueError, match="output_vocab=256"):
            generate_ids(Bhanox(cfg), PROMPT_IDS, max_new=4)

    def test_the_id_api_rejects_an_empty_prompt(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            generate_ids(model, np.array([], np.int64), max_new=4)

    def test_the_id_api_rejects_ids_that_are_not_grams(self, model: Bhanox) -> None:
        """A 4-gram is 4 bytes, so an id wider than that is not a context."""
        with pytest.raises(ValueError, match="4-gram ids"):
            generate_ids(model, np.array([1 << 33], np.int64), max_new=4)

    def test_the_id_api_rejects_a_negative_id(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="4-gram ids"):
            generate_ids(model, np.array([-1], np.int64), max_new=4)

    def test_the_id_api_rejects_a_negative_count(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="max_new"):
            generate_ids(model, PROMPT_IDS, max_new=-1)


class TestDeterminism:
    def test_a_seed_makes_sampling_reproducible(self, model: Bhanox) -> None:
        a = generate(model, PROMPT, max_new=12, temperature=0.9, seed=7)
        b = generate(model, PROMPT, max_new=12, temperature=0.9, seed=7)
        assert a == b

    def test_different_seeds_can_differ(self, model: Bhanox) -> None:
        a = generate(model, PROMPT, max_new=24, temperature=1.5, seed=1)
        b = generate(model, PROMPT, max_new=24, temperature=1.5, seed=2)
        assert a != b

    def test_greedy_is_deterministic(self, model: Bhanox) -> None:
        a = generate(model, PROMPT, max_new=8, temperature=0.0)
        b = generate(model, PROMPT, max_new=8, temperature=0.0)
        assert a == b

    def test_the_id_api_is_also_reproducible(self, model: Bhanox) -> None:
        a = generate_ids(model, PROMPT_IDS, max_new=8, temperature=0.9, seed=4)
        b = generate_ids(model, PROMPT_IDS, max_new=8, temperature=0.9, seed=4)
        assert a.tolist() == b.tolist()


class TestStateIsFreshByDefault:
    """A prompt must not inherit state from whatever ran before it."""

    def test_repeat_calls_agree_without_an_explicit_reset(self, model: Bhanox) -> None:
        a = generate(model, PROMPT, max_new=10, temperature=0.9, seed=6)
        b = generate(model, PROMPT, max_new=10, temperature=0.9, seed=6)
        assert a == b

    def test_a_dirty_model_does_not_change_the_result(self, model: Bhanox) -> None:
        clean = generate(model, PROMPT, max_new=10, temperature=0.9, seed=6)
        generate(
            model, b"a different prompt entirely", max_new=30, temperature=1.4, seed=9
        )
        assert generate(model, PROMPT, max_new=10, temperature=0.9, seed=6) == clean

    def test_reset_false_continues_the_current_sequence(self, model: Bhanox) -> None:
        """The escape hatch: with ``reset=False`` the state carries over, so the
        second call conditions on the first call's output rather than starting
        over."""
        model.reset()
        first = generate(model, PROMPT, max_new=6, temperature=0.0, reset=True)
        carried = generate(model, PROMPT, max_new=6, temperature=0.0, reset=False)
        assert carried != first, "a carried state must change the continuation"
        model.reset()


class TestTemperature:
    def test_greedy_repeats_the_argmax(self, model: Bhanox) -> None:
        """At temperature 0 each sampled byte must be the argmax of the logits
        the model produced after consuming the preceding context."""
        model.reset()
        out = generate(model, PROMPT, max_new=6, temperature=0.0)
        new = out[len(PROMPT) :]
        # Replay the same contexts the loop fed and check each prediction. The
        # loop also steps once after the final sampled byte and discards that
        # result, so the stream here is one longer than the generated run.
        contexts = encode_bytes(out, n=BYTE_GRAM_N).tolist()
        model.reset()
        argmaxes = [int(model.step(g).argmax()) for g in contexts]
        assert argmaxes[len(PROMPT_IDS) - 1 : -1] == list(new)

    def test_a_flat_distribution_still_yields_valid_bytes(self, model: Bhanox) -> None:
        tail = generate(model, PROMPT, max_new=8, temperature=1e6)[len(PROMPT) :]
        assert all(0 <= b < 256 for b in tail)

    def test_higher_temperature_is_not_more_than_the_head(self, model: Bhanox) -> None:
        tail = generate(model, PROMPT, max_new=32, temperature=50.0, seed=3)
        assert len(set(tail)) <= CFG.output_vocab


class TestModelIntegration:
    def test_the_model_method_matches_the_function(self, model: Bhanox) -> None:
        assert model.generate(PROMPT, max_new=6, seed=5) == generate(
            model, PROMPT, max_new=6, seed=5
        )

    def test_the_model_id_method_matches_the_function(self, model: Bhanox) -> None:
        assert model.generate_ids(PROMPT_IDS, max_new=6, seed=5).tolist() == (
            generate_ids(model, PROMPT_IDS, max_new=6, seed=5).tolist()
        )

    def test_generation_is_not_degenerate(self, model: Bhanox) -> None:
        """An untrained model should still produce varied bytes. If this ever
        collapses to a single value, the output projection is dead again."""
        tail = generate(model, PROMPT, max_new=32, temperature=1.0, seed=11)
        assert len(set(tail[len(PROMPT) :])) > 5

    def test_state_stays_bounded_while_generating(self, model: Bhanox) -> None:
        generate(model, PROMPT, max_new=64, temperature=0.9, seed=2)
        assert model.state_nbytes() == 8192

    def test_a_prompt_longer_than_max_context_still_generates(
        self, model: Bhanox
    ) -> None:
        """``max_context`` is a training-window limit, not a runtime one.

        The loop feeds one context at a time, so a prompt longer than the
        training window is a normal request rather than an error. Shrinking
        ``max_context`` keeps the test cheap while still putting the prompt
        well past the limit.
        """
        small = dataclasses.replace(CFG, max_context=8)
        tight = Bhanox(small)
        long = b"a prompt comfortably past the training window"
        assert len(long) > small.max_context
        with pytest.raises(ValueError, match="max_context"):
            tight.forward(encode_bytes(long)[None, :])
        out = generate(tight, long, max_new=4, seed=1)
        assert out == long + out[len(long) :]
