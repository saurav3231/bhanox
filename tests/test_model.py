"""End-to-end tests for the assembled Bhanox model."""

from __future__ import annotations

import numpy as np
import pytest

from bhanox.config import load_config
from bhanox.model import Bhanox

CFG = load_config("nano")
VOCAB = CFG.output_vocab


@pytest.fixture(scope="module")
def model() -> Bhanox:
    return Bhanox(CFG)


def ids(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, VOCAB, n).astype(np.int64)


class TestShapes:
    def test_single_token(self, model: Bhanox) -> None:
        assert model.step(np.array([5], np.int64)).shape == (VOCAB,)

    def test_a_1d_input_is_treated_as_one_sequence(self, model: Bhanox) -> None:
        assert model.forward(ids(6)).shape == (1, 6, VOCAB)

    def test_batch(self, model: Bhanox) -> None:
        b, t = 2, 5
        assert model.forward(np.tile(ids(t), (b, 1))).shape == (b, t, VOCAB)

    def test_output_is_finite(self, model: Bhanox) -> None:
        assert np.all(np.isfinite(model.forward(ids(8))))

    def test_embed_shape(self, model: Bhanox) -> None:
        assert model.embed(ids(4)).shape == (4, CFG.d_model)

    def test_rejects_a_3d_input(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            model.forward(np.zeros((2, 2, 2), np.int64))

    def test_rejects_an_over_long_context(self, model: Bhanox) -> None:
        with pytest.raises(ValueError, match="max_context"):
            model.forward(np.zeros((1, CFG.max_context + 1), np.int64))


class TestStepping:
    def test_step_advances_state(self, model: Bhanox) -> None:
        model.reset()
        a = model.step(np.array([5], np.int64))
        b = model.step(np.array([6], np.int64))
        assert not np.allclose(a, b)

    def test_state_starts_empty_and_fills_on_the_first_write(
        self, model: Bhanox
    ) -> None:
        """A bank holds deltas, so it starts at zero and becomes non-zero only
        once something has been written."""
        model.reset()
        head = model.deltabanks[0].heads[0]
        assert not head.state.any()
        model.step(np.array([5], np.int64))
        assert head.state.any()

    def test_state_is_constant_over_a_long_context(self, model: Bhanox) -> None:
        """The headline claim: recurrent state does not grow with context."""
        model.reset()
        for i in range(64):
            model.step(np.array([i % VOCAB], np.int64))
        assert model.state_nbytes() == 8192

    def test_reset_restores_the_first_step_exactly(self, model: Bhanox) -> None:
        model.reset()
        first = model.step(np.array([7], np.int64))
        model.reset()
        assert np.allclose(first, model.step(np.array([7], np.int64)))

    def test_reset_also_flushes_the_gates(self, model: Bhanox) -> None:
        model.reset()
        for i in range(8):
            model.step(np.array([i], np.int64))
        model.reset()
        assert all(g.awake.all() for g in model.gates)
        assert not any(g.cached.any() for g in model.gates)


class TestCausality:
    def test_a_token_cannot_see_the_future(self, model: Bhanox) -> None:
        model.reset()
        a = ids(8).copy()
        b = a.copy()
        b[-1] = (b[-1] + 1) % VOCAB
        out_a, out_b = model.forward(a), model.forward(b)
        assert np.allclose(out_a[:-1], out_b[:-1], atol=1e-5)

    def test_history_changes_the_output(self, model: Bhanox) -> None:
        model.reset()
        a, b = ids(8).copy(), ids(8).copy()
        b[3] = (b[3] + 1) % VOCAB
        assert not np.allclose(model.forward(a)[-1], model.forward(b)[-1])

    def test_forward_carries_state_between_calls(self, model: Bhanox) -> None:
        """forward() is teacher-forced over one window but the banks persist, so
        calling it twice is not the same as calling it once with the same ids."""
        model.reset()
        one = model.forward(ids(8))
        model.reset()
        model.forward(ids(8))
        assert not np.allclose(one, model.forward(ids(8)))


class TestBatchSemantics:
    """The state is a single instance, so ``B`` is a throughput knob for the
    mixer and not a set of independent streams."""

    def test_a_batch_is_exactly_one_concatenated_stream(self, model: Bhanox) -> None:
        a = np.array([3, 9, 14, 22], np.int64)
        b = np.array([31, 40, 7, 19], np.int64)
        model.reset()
        batched = model.forward(np.stack([a, b]))
        model.reset()
        stream = model.forward(np.concatenate([a, b]))
        assert np.allclose(batched[0], stream[0, :4])
        assert np.allclose(batched[1], stream[0, 4:])

    def test_a_row_sees_the_rows_before_it(self, model: Bhanox) -> None:
        """Pins the shared-state semantics, so nobody later 'fixes' it into a
        silent per-sample state and changes every number measured against it."""
        ids = np.array([3, 9, 14, 22, 31, 40], np.int64)
        model.reset()
        alone = model.forward(ids)[0]
        model.reset()
        preceded = model.forward(np.concatenate([np.array([99, 99]), ids]))[0]
        assert not np.allclose(alone, preceded[2:])

    def test_batch_size_one_is_the_reference(self, model: Bhanox) -> None:
        ids = np.array([5, 6, 7, 8], np.int64)
        model.reset()
        plain = model.forward(ids)
        model.reset()
        assert np.allclose(model.forward(ids[None, :]), plain)


class TestCost:
    def test_parameter_count_is_pinned(self) -> None:
        """A frozen architecture should not drift by accident. This is a
        regression guard, not a target: if a change moves it, say why."""
        assert Bhanox(load_config("nano")).param_count() == 2_107_460

    def test_packed_size_is_one_byte_per_parameter(self, model: Bhanox) -> None:
        assert model.packed_nbytes() == model.param_count()

    def test_state_excludes_the_parameters(self, model: Bhanox) -> None:
        """State is a buffer, not a checkpoint: it must not inflate the model."""
        assert model.state_nbytes() < model.param_count()

    def test_a_bigger_preset_costs_more_of_both(self) -> None:
        nano, mini = Bhanox(load_config("nano")), Bhanox(load_config("mini"))
        assert mini.param_count() > nano.param_count()
        assert mini.state_nbytes() > nano.state_nbytes()

    def test_n_forward_counts_calls(self, model: Bhanox) -> None:
        before = model.n_forward
        model.forward(ids(3))
        model.step(np.array([1], np.int64))
        assert model.n_forward == before + 2


class TestInvariants:
    def test_no_nan_after_many_steps(self, model: Bhanox) -> None:
        model.reset()
        for i in range(64):
            out = model.step(np.array([i % VOCAB], np.int64))
        assert np.all(np.isfinite(out))

    def test_output_scale_is_bounded(self, model: Bhanox) -> None:
        """An untrained model that explodes would make every downstream
        measurement meaningless."""
        assert np.abs(model.forward(ids(16))).max() < 100.0

    def test_training_mode_does_not_corrupt_inference(self, model: Bhanox) -> None:
        """train=True applies the router's load-balancing bias nudge, so the
        logits shift slightly. It must not change them materially -- the shift
        should be orders of magnitude below the signal, not comparable to it.

        Measured on a fresh model, not the shared module-scoped fixture. The
        nudge scales with accumulated expert load, so a fixture carrying state
        from earlier tests measures something different each run, and the
        observed ratio is only ~1.7x under the tolerance. That margin is a CI
        coin flip; the invariant does not need one.
        """
        fresh = Bhanox(CFG)
        a = fresh.forward(ids(8), train=True)
        fresh.reset()
        b = fresh.forward(ids(8), train=False)
        assert np.abs(a - b).max() < 1e-3 * np.abs(b).max()
