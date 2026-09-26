"""Tests for autoregressive generation."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.generate import generate
from bhanox.model import Bhanox

CFG = load_config("nano")
PROMPT = np.array([3, 9, 14, 22], dtype=np.int64)


@pytest.fixture(scope="module")
def model() -> Bhanox:
    return Bhanox(CFG)


class TestContract:
    def test_length_is_prompt_plus_new(self, model: Bhanox) -> None:
        assert len(generate(model, PROMPT, max_new=5)) == len(PROMPT) + 5

    def test_prompt_is_a_prefix(self, model: Bhanox) -> None:
        out = generate(model, PROMPT, max_new=5)
        assert out[: len(PROMPT)].tolist() == PROMPT.tolist()

    def test_dtype_is_int64(self, model: Bhanox) -> None:
        assert generate(model, PROMPT, max_new=3).dtype == np.int64

    def test_ids_are_in_vocab(self, model: Bhanox) -> None:
        out = generate(model, PROMPT, max_new=16)
        assert out.min() >= 0 and out.max() < CFG.output_vocab

    def test_zero_new_tokens_is_the_prompt(self, model: Bhanox) -> None:
        assert generate(model, PROMPT, max_new=0).tolist() == PROMPT.tolist()

    def test_rejects_an_empty_prompt(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="empty"):
            generate(model, np.array([], np.int64), max_new=4)

    def test_rejects_a_negative_count(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="max_new"):
            generate(model, PROMPT, max_new=-1)

    def test_an_id_past_the_table_is_hashed_not_rejected(self, model: Bhanox) -> None:
        """Open vocabulary: an id beyond the direct table still embeds, it just
        skips the dedicated row. Rejecting it would break unseen tokens."""
        # vocab_table, not output_vocab: the point is an id *past* the direct
        # table, so that the HashBind hash path actually runs. output_vocab
        # (256) sits inside the 1024-row table and would test nothing.
        assert CFG.vocab_table > CFG.output_vocab
        assert model.embed(np.array([CFG.vocab_table], np.int64)).shape == (
            1,
            CFG.d_model,
        )


class TestDeterminism:
    def test_a_seed_makes_sampling_reproducible(self, model: Bhanox) -> None:
        model.reset()
        a = generate(model, PROMPT, max_new=12, temperature=0.9, seed=7)
        model.reset()
        b = generate(model, PROMPT, max_new=12, temperature=0.9, seed=7)
        assert a.tolist() == b.tolist()

    def test_different_seeds_can_differ(self, model: Bhanox) -> None:
        model.reset()
        a = generate(model, PROMPT, max_new=24, temperature=1.5, seed=1)
        model.reset()
        b = generate(model, PROMPT, max_new=24, temperature=1.5, seed=2)
        assert a.tolist() != b.tolist()

    def test_greedy_is_deterministic(self, model: Bhanox) -> None:
        model.reset()
        a = generate(model, PROMPT, max_new=8, temperature=0.0)
        model.reset()
        b = generate(model, PROMPT, max_new=8, temperature=0.0)
        assert a.tolist() == b.tolist()


class TestTemperature:
    def test_greedy_repeats_the_argmax(self, model: Bhanox) -> None:
        """At temperature 0 each sampled token must be the argmax of the logits
        the model produced after consuming the previous token."""
        model.reset()
        out = generate(model, PROMPT, max_new=6, temperature=0.0)
        new = out[len(PROMPT) :].tolist()
        # Replay. generate() also steps once after the final sampled token and
        # discards that result, so the stream here is one longer than the
        # generated run, and the argmax that predicts new[0] is the one after
        # the *last* prompt token.
        stream = PROMPT.tolist() + new
        model.reset()
        argmaxes = [int(model.step(np.array([t], np.int64)).argmax()) for t in stream]
        assert argmaxes[len(PROMPT) - 1 : -1] == new

    def test_a_flat_distribution_still_yields_valid_ids(self, model: Bhanox) -> None:
        out = generate(model, PROMPT, max_new=8, temperature=1e6)
        assert out.min() >= 0 and out.max() < CFG.output_vocab

    def test_higher_temperature_is_not_more_than_the_vocab(self, model: Bhanox) -> None:
        out = generate(model, PROMPT, max_new=32, temperature=50.0, seed=3)
        assert len(np.unique(out)) <= CFG.output_vocab


class TestModelIntegration:
    def test_the_model_method_matches_the_function(self, model: Bhanox) -> None:
        model.reset()
        a = generate(model, PROMPT, max_new=6, seed=5)
        model.reset()
        b = model.generate(PROMPT, max_new=6, seed=5)
        assert a.tolist() == b.tolist()

    def test_generation_is_not_degenerate(self, model: Bhanox) -> None:
        """An untrained model should still produce varied tokens. If this ever
        collapses to a single id, the output projection is dead again."""
        model.reset()
        out = generate(model, PROMPT, max_new=32, temperature=1.0, seed=11)
        assert len(np.unique(out)) > 5

    def test_state_stays_bounded_while_generating(self, model: Bhanox) -> None:
        model.reset()
        generate(model, PROMPT, max_new=64, temperature=0.9, seed=2)
        assert model.state_nbytes() == 8192
