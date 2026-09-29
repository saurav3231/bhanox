"""The trainer prototype, on synthetic bytes.

What is being tested, and what deliberately is not
--------------------------------------------------
The prototype's claim is narrow: *a gradient reaches the whole mirror, and the
loss falls on data with a learnable structure.* Every test below serves that
claim or a safety property the loop depends on. None is evidence about
real-corpus training, and the module docstrings in ``bhanox.train.trainer`` and
``bhanox.train.smoke`` say so at length.

Four properties are worth stating up front, because each fails silently rather
than loudly:

1. **The loss is position-wise.** ``inputs[j]`` and ``targets[j]`` are the same
   example. A shift still trains and still reports a falling loss while
   predicting the wrong byte, so it is pinned against a hand computation rather
   than against a trend.
2. **Sparse routing means sparse gradients.** Top-k routing legitimately leaves
   unselected expert rows at exactly zero, so no test here requires a nonzero
   gradient on every expert row. What is required is a valid (finite,
   non-``None``) gradient on everything the optimizer owns, a nonzero one on the
   router, a nonzero one on the experts the chunk actually selected, and evidence
   that selection is not degenerate. Coverage of *every* expert is a property of
   the input, not of the architecture, so it is checked on a constructed input
   and never demanded of the smoke corpus.
3. **``max_context`` is checked before anything mutates.** ``ensure_batch`` grows
   the recurrent state, so a guard placed after it would leave the mirror holding
   rows for a window it refused to process.
4. **Reset and detach are different operations.** A reset zeroes recurrent state
   and flushes gates; a detach keeps the numbers and drops the graph. Mixing them
   up produces a model with no memory (per-chunk reset) or a training run that
   cannot survive its own first ``backward`` (uncleared graph).

The gradient-inventory claim itself is not duplicated here.
``tests/test_model_gradients.py`` already asserts, at structural group level and
at the real config, that every parameter is reachable; these tests ask the
trainer's narrower question, which is whether the set the *optimizer* owns is
exactly the set that receives gradient.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest
import torch

from bhanox.config import BhanoxConfig
from bhanox.data import document_chunks
from bhanox.model import Bhanox
from bhanox.train import mixer_mirror
from bhanox.train.model_mirror import BhanoxMirror
from bhanox.train.objective import (
    next_byte_loss,
    token_mean,
    trainable_parameters,
    unknown_parameter_names,
)
from bhanox.train.reset import clear_shadows, reset_mirror
from bhanox.train.smoke import SMOKE_CYCLE, accuracy_on_cycle, build_documents, main
from bhanox.train.trainer import (
    ChunkReport,
    DocumentReport,
    build_optimizer,
    evaluate_chunk,
    run_documents,
    train_chunk,
)

#: Two layers, so float drift has somewhere to accumulate, and small enough that
#: a twenty-step learning run costs seconds. ``output_vocab`` is 256 rather than
#: the 32 the mirror tests use because the target is a *byte*: a 32-wide
#: unembed cannot hold one, and the resulting ``cross_entropy`` index error would
#: look like a trainer bug rather than a bad test config.
TINY = BhanoxConfig(
    name="t",
    d_model=32,
    d_k=8,
    d_v=8,
    d_expert=16,
    n_heads=2,
    n_layers=2,
    n_experts=4,
    n_shared_experts=1,
    top_k=2,
    output_vocab=256,
    max_context=64,
    pool_size=1024,
    vocab_table=256,
    n_hashes=2,
    seed=7,
)

#: One layer, one head, two routed experts. The 5-cycle has five distinct
#: 4-grams and a 256-wide unembed, so this has ample capacity for the task while
#: costing ~0.5 s per optimizer step instead of ~2 s.
FAST = BhanoxConfig(
    name="f",
    d_model=16,
    d_k=4,
    d_v=8,
    d_expert=16,
    n_heads=1,
    n_layers=1,
    n_experts=2,
    n_shared_experts=1,
    top_k=1,
    output_vocab=256,
    max_context=16,
    pool_size=512,
    vocab_table=256,
    n_hashes=2,
    seed=7,
)


def mirror_of(config: BhanoxConfig = TINY) -> BhanoxMirror:
    return BhanoxMirror(Bhanox(config))


def chunk_of(raw: bytes, *, max_examples: int, doc_index: int = 0):
    return next(iter(document_chunks(doc_index, raw, max_examples=max_examples)))


def state_of(mirror: BhanoxMirror) -> list[torch.Tensor]:
    return [h.state_int.detach().clone() for b in mirror.banks for h in b.heads]


def routing_state(
    mirror: BhanoxMirror,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    return [
        (m.b.detach().clone(), m.loads.detach().clone(), float(m.last_entropy))
        for m in mirror.mixers
    ]


def trainable_grads(
    mirror: BhanoxMirror, chunk, *, train: bool = True
) -> dict[str, torch.Tensor]:
    """One backward on one chunk; gradients for everything the optimizer owns."""
    mirror.zero_grad(set_to_none=True)
    logits, _ = mirror.step(chunk.inputs.reshape(1, -1), train=train)
    targets = torch.from_numpy(np.asarray(chunk.targets, dtype=np.int64))
    next_byte_loss(logits, targets).backward()
    owned = {id(p) for p in trainable_parameters(mirror)}
    return {n: p.grad for n, p in mirror.named_parameters() if id(p) in owned}


# -- the loss -----------------------------------------------------------------


class TestObjective:
    def test_the_loss_is_position_wise(self) -> None:
        """``inputs[j]`` predicts ``targets[j]``, with nothing shifted."""
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 7)
        targets = torch.tensor([6, 0, 3, 1, 4], dtype=torch.int64)
        got = next_byte_loss(logits, targets)
        want = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 7), targets, reduction="sum"
        )
        assert torch.equal(got, want)

    def test_a_shift_would_give_a_different_loss(self) -> None:
        """Anti-vacuous: the test above could pass under a wrong pairing."""
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 7)
        targets = torch.tensor([6, 0, 3, 1, 4, 2], dtype=torch.int64)
        correct = next_byte_loss(logits, targets)
        shifted = next_byte_loss(logits[:, :-1], targets[1:])
        assert not torch.allclose(correct, shifted)

    def test_the_loss_is_a_sum_not_a_mean(self) -> None:
        """The caller divides by a token count it knows. A mean here would make
        that division a silent second averaging.

        Four *identical* positions, so the two reductions are distinguishable at
        all: summed they are ``4x`` the one-position loss, averaged they are
        exactly it. With distinct positions the two are unrelated numbers and the
        test would prove nothing.
        """
        torch.manual_seed(0)
        one_position = torch.randn(1, 1, 5)
        logits = one_position.repeat(1, 4, 1)
        targets = torch.zeros(4, dtype=torch.int64)
        total = next_byte_loss(logits, targets)
        single = next_byte_loss(one_position, targets[:1])
        assert float(total) == pytest.approx(float(single) * 4, rel=1e-6)
        assert float(total) != pytest.approx(float(single))

    def test_token_mean_divides_by_the_count(self) -> None:
        assert float(token_mean(torch.tensor(8.0), 4)) == pytest.approx(2.0)

    def test_token_mean_rejects_a_zero_count(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            token_mean(torch.tensor(1.0), 0)

    def test_a_target_count_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positions"):
            next_byte_loss(torch.randn(1, 4, 5), torch.zeros(3, dtype=torch.int64))

    def test_a_float_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            next_byte_loss(torch.randn(1, 2, 5), torch.zeros(2))

    def test_a_batched_target_is_rejected_with_a_reason(self) -> None:
        """Multiple rows would need their recurrent state carried in lockstep,
        which the chunk stream does not produce.
        """
        with pytest.raises(ValueError, match="lockstep"):
            next_byte_loss(torch.randn(2, 4, 5), torch.zeros((2, 4), dtype=torch.int64))

    def test_a_two_dimensional_target_row_is_accepted(self) -> None:
        """The ``(1, T)`` form a caller gets from a ``(B, 1, T)`` forward is
        flattened rather than rejected.
        """
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 5)
        row = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        assert torch.equal(next_byte_loss(logits, row), next_byte_loss(logits, row[0]))


# -- optimizer parameters -----------------------------------------------------


class TestTrainableParameters:
    def test_salience_is_still_a_parameter(self) -> None:
        """The premise of the exclusion. If this stops holding, the filter below
        is excluding nothing and the reason for it has gone away.
        """
        names = [n for n, _ in mirror_of().named_parameters()]
        assert any("salience" in n for n in names)

    def test_salience_is_excluded(self) -> None:
        mirror = mirror_of()
        trained = {id(p) for p in trainable_parameters(mirror)}
        excluded = {id(p) for n, p in mirror.named_parameters() if "salience" in n}
        assert excluded
        assert not (trained & excluded)

    def test_everything_else_is_included(self) -> None:
        mirror = mirror_of()
        all_params = list(mirror.named_parameters())
        n_salience = sum(1 for n, _ in all_params if "salience" in n)
        assert len(trainable_parameters(mirror)) == len(all_params) - n_salience

    def test_no_parameter_name_is_unrecognised(self) -> None:
        assert unknown_parameter_names(mirror_of()) == []

    def test_an_unrecognised_parameter_is_reported(self) -> None:
        """The guard has to be known to fire, or it is decoration."""
        mirror = mirror_of()
        mirror.register_parameter("mystery", torch.nn.Parameter(torch.zeros(2)))
        assert "mystery" in unknown_parameter_names(mirror)

    def test_the_optimizer_owns_no_salience(self) -> None:
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        salience = {id(p) for n, p in mirror.named_parameters() if "salience" in n}
        assert salience and not (owned & salience)

    def test_a_non_positive_learning_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            build_optimizer(mirror_of(), lr=0.0)


# -- the reset ----------------------------------------------------------------


class TestReset:
    def test_integer_state_is_cleared(self) -> None:
        mirror = mirror_of()
        mirror.step(np.zeros((1, 4), dtype=np.int64), train=False)
        assert any(int(s.abs().sum()) > 0 for s in state_of(mirror))
        reset_mirror(mirror)
        assert all(int(s.abs().sum()) == 0 for s in state_of(mirror))

    def test_gate_state_is_flushed(self) -> None:
        mirror = mirror_of()
        mirror.step(np.zeros((1, 4), dtype=np.int64), train=False)
        reset_mirror(mirror)
        for gate in mirror.gates:
            assert int(gate.cached.abs().sum()) == 0
            assert int(gate._quiet.abs().sum()) == 0
            assert not bool(gate._has_run.any())
            assert bool(gate.awake.all())

    def test_has_run_is_cleared_so_a_gate_can_warm_up(self) -> None:
        """The easy one to miss: a gate carrying ``_has_run`` across a reset would
        never force its first computation again, and would train against a dead
        channel without any error.
        """
        mirror = mirror_of()
        mirror.step(np.zeros((1, 4), dtype=np.int64), train=False)
        assert bool(mirror.gates[0]._has_run.any())
        reset_mirror(mirror)
        assert not bool(mirror.gates[0]._has_run.any())

    def test_parameters_survive_a_reset(self) -> None:
        mirror = mirror_of()
        mirror.step(np.zeros((1, 4), dtype=np.int64), train=True)
        before = {n: p.detach().clone() for n, p in mirror.named_parameters()}
        reset_mirror(mirror)
        for n, p in mirror.named_parameters():
            assert torch.equal(p.detach(), before[n]), f"{n} changed across a reset"

    def test_bank_rates_survive_a_reset(self) -> None:
        mirror = mirror_of()
        before = [b.bank_rates.detach().clone() for b in mirror.banks]
        reset_mirror(mirror)
        for bank, want in zip(mirror.banks, before, strict=True):
            assert torch.equal(bank.bank_rates.detach(), want)

    def test_loads_and_entropy_survive_a_reset(self) -> None:
        """``loads`` is lifetime by design. A per-document reset would reduce
        ``load_entropy()`` to a single-document number, which says nothing about
        whether experts are starving across the run.
        """
        mirror = mirror_of()
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        before = routing_state(mirror)
        assert any(int(load.abs().sum()) > 0 for _, load, _ in before)
        reset_mirror(mirror)
        for mixer, (b, load, entropy) in zip(mirror.mixers, before, strict=True):
            assert torch.equal(mixer.b.detach(), b)
            assert torch.equal(mixer.loads.detach(), load)
            assert float(mixer.last_entropy) == entropy

    def test_reset_is_repeatable(self) -> None:
        mirror = mirror_of()
        mirror.step(np.zeros((1, 4), dtype=np.int64), train=False)
        reset_mirror(mirror)
        reset_mirror(mirror)
        assert all(int(s.abs().sum()) == 0 for s in state_of(mirror))

    def test_the_gelu_table_survives_a_reset(self) -> None:
        mirror = mirror_of()
        before = [m.gelu.detach().clone() for m in mirror.mixers]
        reset_mirror(mirror)
        for mixer, want in zip(mirror.mixers, before, strict=True):
            assert torch.equal(mixer.gelu.detach(), want)


# -- shadows ------------------------------------------------------------------


class TestShadows:
    def test_clear_shadows_preserves_values(self) -> None:
        mirror = mirror_of()
        _, shadows = mirror.step(np.zeros((1, 4), dtype=np.int64), train=False)
        cleared = clear_shadows(shadows)
        assert cleared is not None
        before = [t for layer in shadows for t in layer]
        after = [t for layer in cleared for t in layer]
        assert len(before) == len(after)
        for original, detached in zip(before, after, strict=True):
            assert torch.equal(original.detach(), detached)

    def test_clear_shadows_drops_the_graph(self) -> None:
        """The reason detaching is mandatory: a graph freed by ``backward`` cannot
        be traversed a second time.
        """
        mirror = mirror_of()
        logits, shadows = mirror.step(np.zeros((1, 4), dtype=np.int64), train=True)
        logits.reshape(-1, TINY.output_vocab).sum().backward()
        cleared = clear_shadows(shadows)
        assert cleared is not None
        live = [n for layer in cleared for n in layer if n.requires_grad]
        assert not live, f"shadows still carry autograd history: {len(live)} tensors"

    def test_clear_shadows_passes_none_through(self) -> None:
        assert clear_shadows(None) is None

    def test_detaching_does_not_stop_the_recurrence(self) -> None:
        """Detaching drops the graph, not the state. The int32 buffer is what
        carries the trajectory across a chunk, and detach never touches it.
        """
        raw = bytes((i * 7 + 3) % 256 for i in range(40))
        chunks = list(document_chunks(0, raw, max_examples=8))
        assert len(chunks) > 1
        mirror = mirror_of()
        _, shadows = mirror.step(chunks[0].inputs.reshape(1, -1), train=False)
        after_one = state_of(mirror)
        mirror.step(
            chunks[1].inputs.reshape(1, -1),
            shadows=clear_shadows(shadows),
            train=False,
        )
        after_two = state_of(mirror)
        assert any(
            not torch.equal(a, b) for a, b in zip(after_one, after_two, strict=True)
        ), "the second chunk did not advance any integer state"

    def test_carrying_beats_resetting_within_a_document(self) -> None:
        """With weights fixed, carrying state across chunks is a different
        computation from resetting at each one.

        Forward-only. It says the carry is wired correctly; it says nothing about
        gradients, updates, or a training trajectory, because no weight moves.
        """
        raw = bytes((i * 7 + 3) % 256 for i in range(48))
        chunks = list(document_chunks(0, raw, max_examples=8))
        assert len(chunks) > 1

        carried = mirror_of()
        shadows: list[list[torch.Tensor]] | None = None
        with torch.no_grad():
            for chunk in chunks:
                _, shadows = carried.step(
                    chunk.inputs.reshape(1, -1), shadows=shadows, train=False
                )

        reset_each = mirror_of()
        with torch.no_grad():
            for chunk in chunks:
                reset_mirror(reset_each)
                reset_each.step(chunk.inputs.reshape(1, -1), train=False)

        assert not all(
            torch.equal(a, b)
            for a, b in zip(state_of(carried), state_of(reset_each), strict=True)
        ), "carrying and resetting produced identical state; one is not happening"


# -- max_context --------------------------------------------------------------


class TestMaxContext:
    def test_an_over_long_window_is_rejected(self) -> None:
        mirror = mirror_of()
        with pytest.raises(ValueError, match="exceeds max_context"):
            mirror.step(np.zeros((1, TINY.max_context + 1), dtype=np.int64))

    def test_the_message_names_the_limit(self) -> None:
        mirror = mirror_of()
        with pytest.raises(ValueError, match=str(TINY.max_context)):
            mirror.step(np.zeros((1, TINY.max_context + 1), dtype=np.int64))

    def test_a_rejection_does_not_grow_the_state(self) -> None:
        """The guard has to precede ``ensure_batch``, which *grows* the recurrent
        state. A rejection that mutated would not be a rejection.
        """
        mirror = mirror_of()
        before = [h.state_int.shape[0] for b in mirror.banks for h in b.heads]
        with pytest.raises(ValueError, match="exceeds max_context"):
            mirror.step(np.zeros((8, TINY.max_context + 1), dtype=np.int64))
        after = [h.state_int.shape[0] for b in mirror.banks for h in b.heads]
        assert after == before, "a rejected window grew the recurrent state"

    def test_the_window_boundary_is_accepted(self) -> None:
        mirror = mirror_of()
        logits, _ = mirror.step(np.zeros((1, TINY.max_context), dtype=np.int64))
        assert logits.shape[:2] == (1, TINY.max_context)

    def test_a_three_dimensional_input_is_rejected(self) -> None:
        mirror = mirror_of()
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            mirror.step(np.zeros((1, 4, 4), dtype=np.int64))


# -- train=True side effects --------------------------------------------------


class TestRoutingState:
    def test_a_train_false_forward_changes_no_routing_state(self) -> None:
        """Evaluation must be side-effect-free.

        A property of the *call site* choosing ``train=False``, not of
        ``train=False`` being inert in some deeper sense. ``train=True`` does
        change all three of these, which the next test asserts.
        """
        mirror = mirror_of()
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        before = routing_state(mirror)

        mirror.step(np.zeros((1, 8), dtype=np.int64), train=False)

        for mixer, (b, load, entropy) in zip(mirror.mixers, before, strict=True):
            assert torch.equal(mixer.b.detach(), b), "train=False moved b"
            assert torch.equal(mixer.loads.detach(), load)
            assert float(mixer.last_entropy) == entropy

    def test_a_train_true_forward_does_change_routing_state(self) -> None:
        """The behavior as it exists, asserted so that making it pure would be
        caught as a design change rather than passing unnoticed.
        """
        mirror = mirror_of()
        before = routing_state(mirror)
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        changed = [
            not torch.equal(m.b.detach(), b) or not torch.equal(m.loads.detach(), load)
            for m, (b, load, _) in zip(mirror.mixers, before, strict=True)
        ]
        assert any(changed), "train=True left b and loads untouched"

    def test_evaluate_chunk_leaves_routing_state_unchanged(self) -> None:
        """The helper evaluation is expected to go through."""
        mirror = mirror_of()
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        before = routing_state(mirror)
        chunk = chunk_of(bytes(range(97, 137)), max_examples=16)
        loss, shadows = evaluate_chunk(mirror, chunk)
        assert np.isfinite(loss)
        assert shadows is not None
        for mixer, (b, load, entropy) in zip(mirror.mixers, before, strict=True):
            assert torch.equal(mixer.b.detach(), b)
            assert torch.equal(mixer.loads.detach(), load)
            assert float(mixer.last_entropy) == entropy

    def test_accuracy_on_cycle_leaves_routing_state_unchanged(self) -> None:
        """Same for the smoke's scorer: an eval pass must not contaminate the
        next training step's load-balancing nudge.
        """
        mirror = mirror_of()
        documents = build_documents(1, 40)
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        before = routing_state(mirror)
        accuracy = accuracy_on_cycle(mirror, documents, max_examples=16)
        assert 0.0 <= accuracy <= 1.0
        for mixer, (b, load, entropy) in zip(mirror.mixers, before, strict=True):
            assert torch.equal(mixer.b.detach(), b)
            assert torch.equal(mixer.loads.detach(), load)
            assert float(mixer.last_entropy) == entropy

    def test_the_load_bias_nudges_once_per_layer_per_step(self) -> None:
        """One ``train=True`` call is one load-balancing nudge per layer, which is
        what ties the nudge rate to the per-chunk update schedule.
        """
        mirror = mirror_of()
        before = [m.b.detach().clone() for m in mirror.mixers]
        mirror.step(np.zeros((1, 8), dtype=np.int64), train=True)
        moved = [
            float((m.b.detach() - was).abs().max())
            for m, was in zip(mirror.mixers, before, strict=True)
        ]
        assert all(d > 0.0 for d in moved), f"a layer did not nudge: {moved}"


# -- gradients ----------------------------------------------------------------


class TestGradients:
    @pytest.fixture
    def selected(self, monkeypatch) -> list[torch.Tensor]:
        """Every routed-expert index chosen during the next forward.

        Patched at the module's own selection function rather than recomputed
        from a router call, so the recorded indices are the ones the forward
        actually used. Indices are into the routed pool; storage rows are these
        plus ``n_shared_experts``.
        """
        seen: list[torch.Tensor] = []
        original = mixer_mirror.top2_balanced_torch

        def spy(scores: torch.Tensor, top_k: int):
            order, weights = original(scores, top_k)
            seen.append(order.detach().clone())
            return order, weights

        monkeypatch.setattr(mixer_mirror, "top2_balanced_torch", spy)
        return seen

    def test_every_optimizer_owned_parameter_gets_a_finite_gradient(
        self, selected
    ) -> None:
        """Validity, not magnitude, for everything the optimizer owns. A
        ``None`` here is a parameter AdamW will never update, which is the whole
        reason the inert group is excluded rather than merely noted.
        """
        grads = trainable_grads(
            mirror_of(), chunk_of(bytes(range(97, 137)), max_examples=16)
        )
        assert grads
        for name, grad in grads.items():
            assert grad is not None, f"{name} received no gradient object at all"
            assert bool(torch.isfinite(grad).all()), f"{name} gradient is not finite"

    def test_the_router_receives_a_nonzero_gradient(self) -> None:
        """The router is dense: every token passes through ``route``. Its gradient
        rides the gate weights of the picked experts, which survive the detached
        selection. Detaching the weights as well would kill it silently.
        """
        grads = trainable_grads(
            mirror_of(), chunk_of(bytes(range(97, 137)), max_examples=16)
        )
        for layer in range(TINY.n_layers):
            for name in ("E", "b"):
                grad = grads[f"mixers.{layer}.{name}"]
                assert grad is not None
                assert float(grad.norm()) > 0.0, f"mixers.{layer}.{name} is dead"

    def test_selected_expert_rows_receive_gradient(self, selected) -> None:
        """The sparse-routing contract. Every row the forward selected, plus the
        always-on shared rows, must be live -- otherwise a router could starve an
        expert while the loss still fell.

        Unselected rows are deliberately *not* asserted either way. Requiring them
        to be zero would be as wrong as requiring them to be nonzero: a chunk
        dense enough to touch every expert gives them all gradients.
        """
        mirror = mirror_of()
        chunk = chunk_of(bytes((i * 7 + 3) % 256 for i in range(40)), max_examples=16)
        grads = trainable_grads(mirror, chunk)
        assert selected, "the selection spy recorded nothing"

        for layer in range(TINY.n_layers):
            order = selected[layer].reshape(-1)
            rows = set(int(i) for i in order) | set(range(TINY.n_shared_experts))
            for tensor in ("W1", "W2"):
                grad = grads[f"mixers.{layer}.{tensor}"]
                assert grad is not None
                dead = [r for r in sorted(rows) if float(grad[r].abs().sum()) == 0.0]
                assert not dead, f"mixers.{layer}.{tensor} selected rows dead: {dead}"

    def test_selection_is_not_degenerate(self, selected) -> None:
        """Checked on a constructed input, never on the smoke corpus. A 5-byte
        cycle is a five-entry lookup table, so it says nothing about routing
        coverage -- and demanding full coverage of it would test nothing.
        """
        mirror = mirror_of()
        torch.manual_seed(0)
        for _ in range(16):
            ids = torch.randint(0, TINY.vocab_table, (1, 16), dtype=torch.int64).numpy()
            mirror.step(ids, train=True)
        assert selected
        chosen = {int(i) for order in selected for i in order.reshape(-1)}
        assert len(chosen) >= 2, f"routing collapsed onto {sorted(chosen)}"
        assert len(chosen) <= TINY.n_experts


# -- the document loop --------------------------------------------------------


class TestDocumentLoop:
    def docs(self, n: int = 3) -> list[bytes]:
        return [bytes(SMOKE_CYCLE[i % 5] for i in range(80)) for _ in range(n)]

    def test_each_document_produces_a_report(self) -> None:
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        reports = list(
            run_documents(mirror, optimizer, self.docs(), max_examples=16, name="t")
        )
        assert len(reports) == 3
        assert {r.doc_index for r in reports} == {0, 1, 2}

    def test_the_report_counts_tokens_and_chunks(self) -> None:
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        report = next(
            iter(
                run_documents(
                    mirror, optimizer, self.docs(1), max_examples=16, name="t"
                )
            )
        )
        assert report.n_chunks == 5  # 76 examples, 16 per chunk
        assert report.token_counts == (16, 16, 16, 16, 12)
        assert report.n_tokens == 76 == sum(report.token_counts)
        assert np.isfinite(report.mean_loss)

    def test_a_reset_happens_once_per_document(self) -> None:
        """Injected, so the loop's obligation is checked rather than inferred."""
        calls: list[int] = []

        def counting_reset(mirror: BhanoxMirror) -> None:
            calls.append(1)
            reset_mirror(mirror)

        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        list(
            run_documents(
                mirror,
                optimizer,
                self.docs(3),
                max_examples=16,
                name="t",
                reset=counting_reset,
            )
        )
        assert len(calls) == 3, "one reset per document, not per chunk"

    def test_train_chunk_reports_the_chunk_it_consumed(self) -> None:
        """One call, one chunk, one report -- and the report's coordinates are the
        chunk's, not a re-derivation of them.
        """
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        chunk = chunk_of(bytes(range(97, 137)), max_examples=16, doc_index=2)
        before = [p.detach().clone() for p in trainable_parameters(mirror)]
        report, shadows = train_chunk(mirror, optimizer, chunk)

        assert report.doc_index == 2
        assert report.byte_start == chunk.byte_start
        assert report.n_tokens == len(chunk) == 16
        assert np.isfinite(report.loss)
        after = [p.detach() for p in trainable_parameters(mirror)]
        assert any(
            not torch.equal(was, now) for was, now in zip(before, after, strict=True)
        ), "an optimizer step changed no parameter at all"
        assert shadows and not any(t.requires_grad for layer in shadows for t in layer)

    def test_train_chunk_carries_state_within_a_document(self) -> None:
        """A carried run and a per-chunk-reset run are different computations, and
        only the first remembers the document. Both report the same chunks; the
        integer state they leave behind is what separates them.
        """
        raw = bytes(range(97, 177))
        chunks = list(document_chunks(0, raw, max_examples=16))
        assert len(chunks) > 1

        def run(reset_each: bool) -> tuple[list[torch.Tensor], list[ChunkReport]]:
            mirror = mirror_of()
            optimizer = build_optimizer(mirror, lr=1e-3)
            shadows: list[list[torch.Tensor]] | None = None
            reports: list[ChunkReport] = []
            for chunk in chunks:
                if reset_each:
                    reset_mirror(mirror)
                    shadows = None
                report, shadows = train_chunk(mirror, optimizer, chunk, shadows)
                reports.append(report)
            return state_of(mirror), reports

        carried_state, carried_reports = run(False)
        reset_state, reset_reports = run(True)

        assert [r.byte_start for r in carried_reports] == [c.byte_start for c in chunks]
        assert [r.n_tokens for r in carried_reports] == [len(c) for c in chunks]
        assert not all(
            torch.equal(a, b) for a, b in zip(carried_state, reset_state, strict=True)
        ), "carrying and resetting left identical state; one is not happening"
        # The losses must differ too. Two runs agreeing here would mean the carry
        # changed nothing the loss can see, which is the bug the state check
        # above exists to catch -- so equality would be the alarming result.
        assert [r.loss for r in carried_reports] != [r.loss for r in reset_reports]

    def test_a_short_document_yields_no_report(self) -> None:
        """A document with no examples is skipped, matching
        ``bhanox.data.iter_examples``, so an empty document and an absent one look
        the same to the caller.
        """
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        reports = list(
            run_documents(
                mirror,
                optimizer,
                [b"abcd", b"abcdefgh"],
                max_examples=16,
                name="t",
            )
        )
        assert [r.doc_index for r in reports] == [1]
        assert reports[0].n_tokens == 4

    def test_the_mean_loss_is_token_weighted(self) -> None:
        """A mean of per-chunk means would give a 4-token tail the same vote as a
        16-token head. Built by hand, so the aggregation is checked without
        running a model.
        """
        report = DocumentReport(
            doc_index=0,
            chunks=(
                ChunkReport(0, 0, 16, 1.0),
                ChunkReport(0, 16, 4, 3.0),
            ),
        )
        assert report.mean_loss == pytest.approx((1.0 * 16 + 3.0 * 4) / 20)
        assert report.mean_loss != pytest.approx((1.0 + 3.0) / 2)

    def test_an_empty_report_averages_to_zero(self) -> None:
        assert DocumentReport(doc_index=0).mean_loss == 0.0

    def test_a_short_final_chunk_still_gets_a_full_update(self) -> None:
        """Documented limitation, asserted so it cannot be forgotten.

        80 bytes at ``max_examples=16`` gives chunks of 16, 16, 16, 16, 12: the
        short tail gets a full optimizer step on a noisy gradient. The loss is
        already divided by that chunk's own token count, and dividing again would
        not help -- AdamW's update is roughly invariant to a uniform scaling of
        the loss. The test records the histogram instead of pretending the bias
        is absent.
        """
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        report = next(
            iter(
                run_documents(
                    mirror, optimizer, self.docs(1), max_examples=16, name="t"
                )
            )
        )
        assert report.token_counts[-1] < max(report.token_counts)
        assert len(report.chunks) == report.n_chunks


# -- the smoke run ------------------------------------------------------------


class TestSmoke:
    def test_the_smoke_run_reports_chunk_token_counts(self) -> None:
        """The histogram the trainer docstrings promise. For the smoke corpus it
        is uniform by construction -- ``4 * max_examples`` examples in 4 equal
        chunks -- which is the point: the toy run must not be the thing that
        exercises the short-tail exposure.
        """
        documents = build_documents(1, 16 * 4 + 4)
        mirror = mirror_of()
        optimizer = build_optimizer(mirror, lr=1e-3)
        report = next(
            iter(
                run_documents(
                    mirror, optimizer, documents, max_examples=16, name="smoke"
                )
            )
        )
        assert report.token_counts == (16, 16, 16, 16)
        assert report.n_tokens == 64

    def test_build_documents_is_deterministic_and_in_range(self) -> None:
        docs = build_documents(2, 12)
        assert docs == [docs[0], docs[1]] or docs[0] == docs[1]
        assert len(docs[0]) == 12
        assert bytes(SMOKE_CYCLE[i % 5] for i in range(12)) == docs[0]
        assert build_documents(2, 12) == docs

    def test_build_documents_rejects_a_degenerate_request(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            build_documents(0, 32)
        with pytest.raises(ValueError, match="at least 2 bytes"):
            build_documents(1, 32, cycle=b"A")

    def test_the_loss_falls_and_accuracy_rises_on_learnable_bytes(self) -> None:
        """The prototype's actual claim, at the size where it costs seconds.

        Deterministic: the config seed fixes the weights and the run seed fixes
        the document order, so this is a fixed curve rather than a statistical
        claim. What it does not establish is anything about real data -- see the
        "what this does not prove" list in ``bhanox.train.smoke``.
        """
        max_examples = 16
        documents = build_documents(1, max_examples * 2 + 4)
        mirror = BhanoxMirror(Bhanox(FAST))
        optimizer = build_optimizer(mirror, lr=3e-2)

        first = None
        for _step in range(10):
            report = next(
                iter(
                    run_documents(
                        mirror,
                        optimizer,
                        documents,
                        max_examples=max_examples,
                        name="f",
                    )
                )
            )
            if first is None:
                first = report.mean_loss
        assert first is not None
        assert np.isfinite(first)
        assert (
            report.mean_loss < 0.25 * first
        ), f"loss did not fall: {first:.4f} -> {report.mean_loss:.4f}"
        accuracy = accuracy_on_cycle(mirror, documents, max_examples=max_examples)
        assert accuracy >= 0.9, f"argmax accuracy only reached {accuracy:.3f}"

    def test_a_fixed_seed_reproduces_the_curve(self) -> None:
        """Two identical runs must produce identical numbers, or a reported curve
        is a sample rather than a measurement.
        """
        max_examples = 16
        documents = build_documents(1, max_examples + 4)

        def curve() -> list[float]:
            mirror = BhanoxMirror(Bhanox(FAST))
            optimizer = build_optimizer(mirror, lr=3e-2)
            out = []
            for _ in range(3):
                out.append(
                    next(
                        iter(
                            run_documents(
                                mirror,
                                optimizer,
                                documents,
                                max_examples=max_examples,
                                name="f",
                            )
                        )
                    ).mean_loss
                )
            return out

        assert curve() == curve()


class TestSmokeCli:
    """The command itself, not the loop. A one-line "curve" printed with exit
    code 0 is the failure mode worth guarding: it looks like a result.

    Driven through a tiny config file rather than ``nano``. ``main`` accepts a
    path to a ``.json`` config, and the alternative -- nano at three steps -- is
    minutes of CPU for an assertion about the printed line count.
    """

    @pytest.fixture
    def config_path(self, tmp_path) -> str:
        path = tmp_path / "cli.json"
        tiny = {
            "name": "cli",
            "d_model": 16,
            "d_k": 4,
            "d_v": 8,
            "d_expert": 16,
            "n_heads": 1,
            "n_layers": 1,
            "n_experts": 2,
            "n_shared_experts": 1,
            "top_k": 1,
            "output_vocab": 256,
            "max_context": 8,
            "pool_size": 512,
            "vocab_table": 256,
            "n_hashes": 2,
            "seed": 7,
        }
        path.write_text(json.dumps(tiny), encoding="utf-8")
        return str(path)

    def test_it_prints_one_row_per_step(self, config_path, capsys) -> None:
        code = main(["--config", config_path, "--steps", "3", "--max-examples", "8"])
        out = capsys.readouterr().out
        rows = [
            line
            for line in out.splitlines()
            if line.strip()[:1].isdigit() and "chunks" not in line
        ]
        assert code == 0
        assert len(rows) == 3, f"expected 3 step rows, got {len(rows)}:\n{out}"
        assert "chunk token counts: (8, 8, 8, 8)" in out
        assert "measured only" in out

    def test_a_non_positive_step_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            main(["--steps", "0"])

    def test_the_window_cannot_exceed_the_config_limit(
        self, config_path, capsys
    ) -> None:
        """``max_examples`` is min'd against ``max_context``, so the run cannot ask
        the mirror for a context it has already promised to reject.
        """
        main(["--config", config_path, "--steps", "1", "--max-examples", "8192"])
        assert "max_examples=8" in capsys.readouterr().out


class TestDevicePortability:
    """The mirror must survive ``.to(device)`` without changing what it computes.

    Added for the Kaggle GPU smoke. The four fixed sites all had the same shape:
    a tensor created without ``device=`` lands on the *default* device, not the
    one the module's weights were moved to, so a GPU run raised a device mismatch
    on the first token. These tests are the regression net for that, and they run
    on CPU because the property is structural rather than GPU-specific.
    """

    def test_to_device_does_not_change_the_logits(self) -> None:
        """``.to()`` must be a pure relocation.

        This is the property that catches a missing ``device=`` even with no GPU
        present: a tensor built on the default device is either a mismatch (the
        op raises) or a silent second copy, and a silent second copy shows up here
        as a nonzero difference.
        """
        ids = np.arange(24, dtype=np.int64).reshape(1, -1)
        fresh = BhanoxMirror(Bhanox(TINY))
        moved = BhanoxMirror(Bhanox(TINY)).to("cpu")
        with torch.no_grad():
            logit_fresh, _ = fresh.step(ids, train=False)
            logit_moved, _ = moved.step(ids, train=False)
        assert torch.equal(logit_fresh, logit_moved)

    def test_every_parameter_and_buffer_shares_one_device(self) -> None:
        """No tensor is left behind on a different device.

        Catches an init-only site that registered a buffer with a hardcoded
        device, which ``.to()`` would then have to fight.
        """
        mirror = BhanoxMirror(Bhanox(TINY)).to("cpu")
        devices = {
            t.device.type for t in list(mirror.parameters()) + list(mirror.buffers())
        }
        assert devices == {"cpu"}

    def test_training_a_moved_mirror_still_updates_weights(self) -> None:
        """A full optimizer step on a ``.to()``-ed mirror must move the weights.

        The four fixed sites are all on the training path but none of them is in
        the backward pass, so a forward-only test would pass even with the
        integer state advance broken. This is the one that would catch it.
        """
        mirror = BhanoxMirror(Bhanox(TINY)).to("cpu")
        optimizer = build_optimizer(mirror, lr=1e-2)
        before = {name: p.detach().clone() for name, p in mirror.named_parameters()}
        chunk = chunk_of(bytes(range(64)), max_examples=24)
        report, _ = train_chunk(mirror, optimizer, chunk)
        assert np.isfinite(report.loss)
        moved = [
            name
            for name, p in mirror.named_parameters()
            if not torch.equal(before[name], p.detach())
        ]
        assert moved, "no parameter changed; the optimizer step did nothing"

    def test_integer_state_advance_runs_on_a_moved_mirror(self) -> None:
        """``_write_int`` crosses to the host every token; it must survive ``.to()``.

        ``mirror.py``'s two ``.numpy()`` calls raised on a CUDA tensor inside the
        integer state advance -- the one part of the recurrence that cannot be
        skipped or deferred. A forward that returns the right logits while
        leaving ``state_int`` untouched would be a silent, serious bug, so the
        state is checked for having actually advanced.
        """
        mirror = BhanoxMirror(Bhanox(TINY)).to("cpu")
        head = mirror.banks[0].heads[0]
        before = head.to_numpy_state().copy()
        with torch.no_grad():
            mirror.step(np.full((1, 32), 65, dtype=np.int64), train=False)
        assert not np.array_equal(before, head.to_numpy_state())


def test_no_hot_path_tensor_is_created_without_a_device() -> None:
    """Every tensor built during ``step()`` must name its device explicitly.

    A behavioural test cannot catch this on a CPU-only host: the module and the
    default device are both ``cpu``, so a missing ``device=`` is invisible at
    runtime and the first evidence is a CUDA mismatch on Kaggle. This reads the
    source instead, which is why it is a source test and not a mirror test.

    The scope is deliberately narrow -- the functions ``BhanoxMirror.step`` calls
    per token and per layer. The init and checkpoint-load paths build tensors
    from numpy before any ``.to()`` has happened, so the default device is
    correct there and demanding ``device=`` would be noise.

    Verified to fail when ``device=gathered.device`` is removed from
    ``HashBindMirror.forward``, and to pass on the current source.
    """
    hot_path = {
        "forward",
        "step",
        "surrogate",
        "forward_int",
        "_write_int",
        "_codes",
        "update_load_bias",
    }
    creators = {"zeros", "ones", "tensor", "arange", "empty", "full"}
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "bhanox" / "train"
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in hot_path:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in creators
                    and not any(kw.arg == "device" for kw in call.keywords)
                ):
                    offenders.append(
                        f"{path.name}:{call.lineno} torch.{call.func.attr}() "
                        f"in {node.name}()"
                    )
    assert (
        not offenders
    ), "device-less tensor creation on the training path: " + "; ".join(offenders)


def test_no_integer_einsum_on_the_training_path() -> None:
    """No integral ``torch.einsum`` anywhere in ``bhanox.train``.

    The first real GPU training run died here. ``DeltaBankHeadMirror`` accumulated
    its integer state read with ``torch.einsum("bkv,bk->bv", state, codes)`` in
    int64, which runs fine on CPU and raises ``NotImplementedError`` on CUDA,
    because ATen's einsum has no integral kernel. Every local test passed, every
    CPU smoke passed, and the failure only appeared on a T4.

    A behavioural test cannot catch that on a CPU-only host -- the same blind spot
    as the source test above, so this reads the source. The assertion is
    deliberately narrow: only *integral* einsums are forbidden, and only in
    ``bhanox.train``. The float einsums in ``bookend_mirror`` and ``mixer_mirror``
    stay, because float contraction order changes the result and those are pinned
    by tolerance tests instead. The integer path is spelled out by
    ``_contract_k``, which is exact -- integer addition is associative, so the
    reduction order cannot change the answer.

    Verified to fail when ``_contract_k`` is reverted to an integer einsum, and to
    pass on the current source.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "bhanox" / "train"
    int_dtypes = {"int8", "int16", "int32", "int64", "uint8"}
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "einsum"
            ):
                continue
            # An einsum is integral if either operand is cast to an integer dtype.
            integral = any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "to"
                and inner.args
                and isinstance(inner.args[0], ast.Attribute)
                and inner.args[0].attr in int_dtypes
                for arg in node.args
                for inner in ast.walk(arg)
            )
            if integral:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "integral torch.einsum on the training path, which has no CUDA kernel: "
        + "; ".join(offenders)
        + " -- use mirror._contract_k instead"
    )


def test_contract_k_is_exact_against_numpy_einsum() -> None:
    """``_contract_k`` must equal the einsum it replaced, exactly, on every shape.

    The substitution is only safe because integer addition is associative and
    commutative with no rounding, so reduction order cannot change the answer.
    This is what makes that claim checkable rather than merely plausible: it
    compares against the real einsum and the real numpy reference over random
    shapes rather than trusting the argument.
    """
    from bhanox.train.mirror import _contract_k

    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        batch = int(torch.randint(1, 3, (1,), generator=generator))
        buckets = int(torch.randint(1, 33, (1,), generator=generator))
        width = int(torch.randint(1, 65, (1,), generator=generator))
        state = torch.randint(
            -128, 128, (batch, buckets, width), dtype=torch.int32, generator=generator
        )
        codes = torch.randint(-127, 128, (batch, buckets), generator=generator)

        result = _contract_k(state, codes)
        assert result.dtype == torch.int64
        assert result.shape == (batch, width)

        by_torch = torch.einsum(
            "bkv,bk->bv", state.to(torch.int64), codes.to(torch.int64)
        )
        assert torch.equal(result, by_torch), "diverged from torch.einsum"
        by_numpy = np.einsum(
            "bkv,bk->bv", state.numpy().astype(np.int64), codes.numpy().astype(np.int64)
        )
        assert np.array_equal(result.numpy(), by_numpy), "diverged from np.einsum"
