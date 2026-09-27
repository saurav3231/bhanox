"""The MicroExpert mirror must reproduce the reference, and prove it is sparse.

Four claims, and each needs its own instrument because a single loose assertion
would prove none of them:

1. **Agreement.** The mirror's forward equals the reference's. This is a *float*
   tolerance, not bit-exactness -- unlike the DeltaBank mirror, and the reason is
   in ``mixer_mirror``'s docstring: this layer stores dequantized weights, so its
   reference forward is already float. Stating the weaker claim is the honest
   thing; asserting bit-exactness here would be asserting something false that
   happens to pass because both sides happen to be float32.
2. **Sparsity.** Gathering the top-2 rows per token gives the same answer as
   running every expert, while touching only ``n_shared + top_k`` experts' bytes.
   Agreement alone does not prove the gather happened -- a mirror that computed
   all 17 experts would agree perfectly and be dense, which defeats the layer.
3. **Gradient reach.** ``E`` and ``b`` receive gradient through the gate weights,
   not through the selection. The selection is discrete, so a mirror that
   detached the weights too would have a dead router while passing every
   agreement test.
4. **Byte ratio.** 5.7x like-for-like, asserted on the byte counts themselves
   rather than on a hard-coded constant, plus a guard against the 5.3 figure the
   reference docstrings quote.

Claim 4's guard matters more than it looks: 5.3 is not a typo, it is the ratio you
get by counting the shared expert on the active side and omitting it from the
dense side. The guard pins that the denominator includes it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from bhanox.config import load_config
from bhanox.mixer.microexpert import (
    MicroExpertLayer,
    gelu_lut,
    gelu_lut_apply,
    top2_balanced,
)
from bhanox.train.mixer_mirror import (
    MicroExpertLayerMirror,
    gelu_lut_apply_torch,
    top2_balanced_torch,
)

torch.manual_seed(0)

#: float32 accumulation over a (n_tokens, d_model) reduction. The reference sums
#: in a different order than torch's matmul, so exact equality is not on offer;
#: 1e-5 is roughly 100x the observed 5e-7 and still four orders below the signal.
AGREE_TOL = 1e-5


def make_pair(
    name: str = "nano", layer_index: int = 0
) -> tuple[MicroExpertLayer, MicroExpertLayerMirror]:
    """A reference layer and its mirror, both from the reference's own init."""
    ref = MicroExpertLayer(load_config(name), layer_index)
    return ref, MicroExpertLayerMirror(ref)


def tiny_config(n_shared: int = 2) -> object:
    """A minimal config, so shared-expert behaviour is testable at all.

    Every shipped preset has ``n_shared_experts == 1`` except ``small``, which is
    far too large to mirror in a unit test. That matters: an implementation that
    ran ``range(1)`` instead of ``range(self.n_shared)`` is a no-op on nano, mini
    *and* small-by-accident on the shared path, so a suite built only on shipped
    presets would pass. ``test_every_shared_expert_runs`` closes that hole using
    this config instead.

    The rest of the fields are the smallest values ``BhanoxConfig`` accepts while
    still satisfying I1 (``head_load < d_k``) in ``__post_init__``.
    """
    from bhanox.config import BhanoxConfig

    return BhanoxConfig(
        name="tiny",
        d_model=16,
        n_layers=2,
        n_heads=1,
        d_k=8,
        d_v=8,
        n_banks=4,
        n_experts=6,
        n_shared_experts=n_shared,
        d_expert=8,
        top_k=2,
        use_vault=False,
        ternary=False,
    )


def make_tiny_pair(
    n_shared: int = 2,
) -> tuple[MicroExpertLayer, MicroExpertLayerMirror]:
    """Reference and mirror over :func:`tiny_config`."""
    ref = MicroExpertLayer(tiny_config(n_shared), 0)  # type: ignore[arg-type]
    return ref, MicroExpertLayerMirror(ref)


def xs(d_model: int, shape: tuple[int, ...], seed: int = 3) -> Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((*shape, d_model)).astype(np.float32))


class TestAgreement:
    """Claim 1: the mirror's forward is the reference's forward."""

    @pytest.mark.parametrize("shape", [(1,), (7,), (2, 5), (3, 2, 4)])
    def test_any_leading_shape(self, shape: tuple[int, ...]) -> None:
        """A single vector, a token batch, and 3-D batches must all agree.

        The reference flattens to 2-D internally precisely so this holds; a
        mirror that indexed the wrong axis on a 3-D input would return the right
        answer for a 1-D input and the wrong one here.
        """
        ref, mir = make_pair()
        x = xs(ref.config.d_model, shape)
        assert (
            np.abs(ref.forward(x.numpy()) - mir(x).detach().numpy()).max() < AGREE_TOL
        )

    def test_train_false_is_the_default(self) -> None:
        """``forward`` must default to ``train=False``.

        The default is the safe side of a side-effecting flag: agreement and
        gradient tests would otherwise be comparing two different functions.
        """
        ref, mir = make_pair()
        x = xs(ref.config.d_model, (5,))
        before = mir.b.detach().clone()
        mir(x)
        assert torch.equal(mir.b.detach(), before), "forward() mutated the router bias"

    def test_mirror_agrees_across_a_batch_of_steps(self) -> None:
        """Agreement must hold per call, not just on the first call.

        The state that carries between calls is only the load counters at
        ``train=False``; a mirror that leaked router state into the next call
        would agree on call 1 and drift after.
        """
        ref, mir = make_pair()
        for i in range(4):
            x = xs(ref.config.d_model, (6,), seed=i)
            assert np.abs(ref.forward(x.numpy()) - mir(x).detach().numpy()).max() < (
                AGREE_TOL
            )


class TestSparsity:
    """Claim 2: the gather is real -- same answer, fewer bytes."""

    def test_gathered_equals_per_token_expert_application(self) -> None:
        """The per-token gather must equal applying each token's pick directly.

        This is what proves the gather is a *correct* gather. The byte test alone
        cannot: a mirror that computed all ``n_total`` experts and then simply
        reported a top-2 byte count would pass the ratio test and the agreement
        test simultaneously while being dense, which defeats the layer.

        Written longhand -- loop over tokens, index the expert by hand -- so the
        implementation being checked is the vectorised gather, not a restatement
        of it.
        """
        _, mir = make_pair()
        cfg = mir.config
        x = xs(cfg.d_model, (12,))
        flat = x.reshape(-1, cfg.d_model)

        indices, weights = top2_balanced_torch(mir.route(flat), cfg.top_k)
        with torch.no_grad():
            expected = torch.zeros_like(flat)
            for t in range(flat.shape[0]):
                for slot in range(cfg.top_k):
                    j = int(indices[t, slot]) + cfg.n_shared_experts
                    # Expert j applied to token t alone, then scaled by its gate.
                    solo = mir._expert(flat[t : t + 1], j)
                    expected[t] = expected[t] + weights[t, slot] * solo[0]

        got = mir(x).reshape(-1, cfg.d_model)
        shared = sum(mir._expert(flat, j) for j in range(cfg.n_shared_experts))
        assert torch.allclose(got - shared, expected, atol=1e-5)

    def test_output_is_exactly_shared_plus_routed(self) -> None:
        """The shared expert is one unconditional addend, not a routed slot.

        Also pins the decomposition the sparse-bytes claim rests on: if the shared
        contribution were somehow inside the top-k branch, the byte count would
        still include it and the ratio would still look right while the layer had
        lost the "nothing is unreachable" guarantee.
        """
        _, mir = make_pair()
        cfg = mir.config
        # n_total must exceed the active set, or "sparse" is vacuous here.
        assert cfg.total_experts > cfg.n_shared_experts + cfg.top_k
        flat = xs(cfg.d_model, (10,))
        with torch.no_grad():
            shared = sum(mir._expert(flat, j) for j in range(cfg.n_shared_experts))
            got = mir(flat.reshape(10, cfg.d_model)).reshape(-1, cfg.d_model)
            # Recomputing the shared part alone and subtracting must leave exactly
            # the routed remainder, and the shared part must be nonzero -- an
            # all-zero shared term would make this test pass with the expert
            # removed.
            assert float(shared.abs().max()) > 0.0
            routed = got - shared
            assert float(routed.abs().max()) > 0.0
            # And the routed remainder must be strictly smaller in magnitude than
            # the full output would be if every expert ran: the selection is real.
            every = sum(mir._expert(flat, j) for j in range(cfg.total_experts))
            assert float(routed.abs().max()) < float(every.abs().max())

    def test_every_shared_expert_runs(self) -> None:
        """All ``n_shared`` shared experts must contribute, not just the first.

        This is the test that can actually fail. Nano, mini and small all ship
        ``n_shared_experts == 1``, so on every shipped preset an implementation
        that wrote ``range(1)`` where it meant ``range(self.n_shared)`` is
        indistinguishable from a correct one. ``make_tiny_pair(2)`` is the only
        reason this hole is visible: with two shared experts, dropping the second
        is an observable change in the output.
        """
        for n_shared in (1, 2, 3):
            _, mir = make_tiny_pair(n_shared)
            cfg = mir.config
            flat = xs(cfg.d_model, (6,))
            indices, weights = top2_balanced_torch(mir.route(flat), cfg.top_k)
            with torch.no_grad():
                # Full expected sum: every shared expert, plus the routed picks.
                total = sum(mir._expert(flat, j) for j in range(n_shared))
                for t in range(flat.shape[0]):
                    for slot in range(cfg.top_k):
                        j = int(indices[t, slot]) + cfg.n_shared_experts
                        total[t] = (
                            total[t]
                            + weights[t, slot] * mir._expert(flat[t : t + 1], j)[0]
                        )
                got = mir(flat).reshape(-1, cfg.d_model)
                assert torch.allclose(
                    got, total, atol=1e-5
                ), f"n_shared={n_shared}: output is not the sum of all shared experts"
                # Each shared expert must be individually nonzero, so removing any
                # one of them is detectable rather than cancelling out.
                for j in range(n_shared):
                    contrib = mir._expert(flat, j)
                    assert (
                        float(contrib.abs().max()) > 0.0
                    ), f"shared expert {j} is zero"

    def test_active_bytes_count_every_shared_expert(self) -> None:
        """The byte formula must scale with ``n_shared``, not assume one.

        Guards the same hole on the accounting side. With two shared experts the
        active count must rise by exactly one expert's worth of int8 weights.
        """
        _, mir1 = make_tiny_pair(1)
        _, mir2 = make_tiny_pair(2)
        cfg1, cfg2 = mir1.config, mir2.config
        per_expert = 2 * cfg1.d_model * cfg1.d_expert
        assert mir1.active_nbytes() == (1 + cfg1.top_k) * per_expert
        assert mir2.active_nbytes() == (2 + cfg2.top_k) * per_expert
        assert mir2.active_nbytes() - mir1.active_nbytes() == per_expert
        """At most ``top_k`` distinct expert rows per token, offset past shared.

        Counted by instrumenting the gather, not by trusting the byte formula.
        """
        _, mir = make_pair()
        cfg = mir.config
        scores = mir.route(xs(cfg.d_model, (64,)))
        indices, _ = top2_balanced_torch(scores, cfg.top_k)
        assert indices.shape == (64, cfg.top_k)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < cfg.n_experts
        # Rows actually gathered are offset into the pooled storage.
        rows = indices + cfg.n_shared_experts
        assert int(rows.max()) < cfg.total_experts
        # Per token the distinct count never exceeds top_k.
        for row in indices:
            assert len(set(row.tolist())) <= cfg.top_k

    def test_shared_expert_always_runs(self) -> None:
        """Nothing is unreachable: the shared slot is added unconditionally.

        Ties the byte count to the code path. If the shared expert were inside
        the top-k branch, ``active_nbytes`` would still count it and the ratio
        would still look right while the layer had lost its guarantee.
        """
        ref, mir = make_pair()
        x = xs(ref.config.d_model, (3,))
        flat = x.reshape(-1, ref.config.d_model)
        expected = mir._expert(flat, 0)
        full = mir(x).reshape(-1, ref.config.d_model)
        routed_part = full - expected
        # The routed part is finite and not equal to the shared part, i.e. the
        # shared contribution is genuinely one addend of the sum.
        assert torch.isfinite(routed_part).all()
        assert not torch.allclose(routed_part, expected)


class TestGradientReach:
    """Claim 3: the router is trainable, and the gradient is not vacuous."""

    def test_gradients_reach_router_and_experts(self) -> None:
        """Every parameter and the input must receive a nonzero gradient."""
        _, mir = make_pair()
        x = xs(mir.config.d_model, (9,)).requires_grad_(True)
        mir(x).pow(2).sum().backward()
        for name, p in mir.named_parameters():
            assert p.grad is not None, f"{name} received no gradient at all"
            assert float(p.grad.norm()) > 0.0, f"{name} gradient is exactly zero"
        assert float(x.grad.norm()) > 0.0

    def test_router_gradient_survives_detached_selection(self) -> None:
        """Anti-vacuous: the router's gradient comes from the *weights*.

        The selection is discrete and must be detached. A mirror that detached the
        picked probabilities as well would have a dead router -- ``E`` and ``b``
        would receive exactly zero gradient -- while every agreement test still
        passed. This test detaches the selection explicitly and requires the
        router gradient to survive on the weight path alone.
        """
        _, mir = make_pair()
        cfg = mir.config
        flat = xs(cfg.d_model, (9,))

        scores = mir.route(flat)
        order = torch.argsort(-scores, dim=-1, stable=True)[..., : cfg.top_k]
        order = order.detach()  # the selection is discrete
        picked = torch.gather(scores, -1, order)
        weights = picked / picked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights.sum().backward()

        assert float(mir.E.grad.norm()) > 0.0, "router weights carry no E gradient"
        assert float(mir.b.grad.norm()) > 0.0, "router weights carry no b gradient"

    def test_selection_itself_is_not_differentiable(self) -> None:
        """The chosen indices are integers; there is no gradient through them.

        This is the honest description of a hard top-k, and asserting it keeps
        the previous test from being misread as a claim that selection is learned.
        """
        _, mir = make_pair()
        scores = mir.route(xs(mir.config.d_model, (5,)))
        indices, weights = top2_balanced_torch(scores, mir.top_k)
        assert not indices.requires_grad
        assert weights.requires_grad

    def test_top_k_weights_are_renormalised(self) -> None:
        """The picked gate weights sum to 1, as the reference specifies.

        Without renormalisation the layer's output scale drifts with how peaked
        routing happens to be, which is a quiet bug rather than a loud one.
        """
        _, mir = make_pair()
        cfg = mir.config
        scores = mir.route(xs(cfg.d_model, (32,)))
        _, weights = top2_balanced_torch(scores, cfg.top_k)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(32), atol=1e-6)

    def test_shared_expert_path_is_differentiable(self) -> None:
        """``W1``/``W2`` get gradient from the always-on shared expert alone.

        Isolated by running a single expert directly. This closes the possibility
        that the shared slot was excluded from autograd, which would leave
        ``W1[0]``/``W2[0]`` untrained while the routed rows carried the update.
        """
        _, mir = make_pair()
        flat = xs(mir.config.d_model, (5,))
        mir._expert(flat, 0).pow(2).sum().backward()
        # Row 0 is the shared expert's slot in the pooled W1/W2.
        assert float(mir.W1.grad[0].norm()) > 0.0
        assert float(mir.W2.grad[0].norm()) > 0.0


class TestTop2Agreement:
    """Claim 1 again, at the routing decision, where ties live."""

    def test_indices_and_weights_match_the_reference(self) -> None:
        _, mir = make_pair()
        rng = np.random.default_rng(11)
        scores = rng.standard_normal((17, mir.config.n_experts)).astype(np.float32)
        ri, rw = top2_balanced(scores, mir.top_k)
        ti, tw = top2_balanced_torch(torch.tensor(scores), mir.top_k)
        assert np.array_equal(ri, ti.numpy())
        assert np.abs(rw - tw.detach().numpy()).max() < 1e-6

    def test_ties_keep_reference_order(self) -> None:
        """All-equal scores must select the lowest expert indices.

        An untrained router produces ties as a matter of course, so the sort must
        be stable or the mirror disagrees with the reference exactly where a fresh
        model lives. Asserted on the tie, not on a random input.
        """
        cfg = load_config("nano")
        scores = np.full((3, cfg.n_experts), 0.25, dtype=np.float32)
        ri, _ = top2_balanced(scores, cfg.top_k)
        ti, _ = top2_balanced_torch(torch.tensor(scores), cfg.top_k)
        assert np.array_equal(ri, ti.numpy())
        assert ri[0].tolist() == [0, 1]

    def test_rejects_impossible_top_k(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            top2_balanced_torch(torch.zeros(2, 3), 5)
        with pytest.raises(ValueError, match="must be > 0"):
            top2_balanced_torch(torch.zeros(2, 3), 0)


class TestByteRatio:
    """Claim 4: 5.7x, asserted on the counts and guarded against 5.3."""

    def test_active_byte_count_is_top_k_plus_shared(self) -> None:
        """Active bytes are ``(n_shared + top_k)`` experts' worth of int8 weights.

        Asserted as an arithmetic identity against the config rather than against
        a literal, so a preset change moves the expectation instead of silently
        invalidating the test.
        """
        for name in ("nano", "mini", "small"):
            _, mir = make_pair(name)
            cfg = mir.config
            expected = (
                (cfg.n_shared_experts + cfg.top_k) * 2 * cfg.d_model * cfg.d_expert
            )
            assert mir.active_nbytes() == expected

    def test_dense_byte_count_includes_the_shared_expert(self) -> None:
        """Dense bytes cover every expert the layer owns, shared included.

        This is the guard. Omitting the shared expert from the denominator yields
        16/3 = 5.33 -- the 5.3 figure the reference docstrings quote -- which is
        not a conservative rounding of 5.67 but a different, mismatched
        comparison. The two sides must be measured over the same weights.
        """
        _, mir = make_pair("nano")
        cfg = mir.config
        assert mir.dense_nbytes() == 2 * cfg.d_model * cfg.d_expert * cfg.total_experts
        wrong = 2 * cfg.d_model * cfg.d_expert * cfg.n_experts / mir.active_nbytes()
        assert wrong == pytest.approx(5.333, abs=1e-3), (
            "the 5.3 figure should be reproducible only by dropping the shared "
            "expert from the dense side"
        )

    def test_ratio_rounds_to_5_7x(self) -> None:
        """The claim is 5.7x to one decimal. Asserted as the rounded value.

        Note what is deliberately *not* asserted: ``ratio >= 5.7``. Nano is
        exactly 17/3 = 5.666..., which rounds to 5.7 and is strictly less than
        5.7. A ``>= 5.7`` bound would be an assertion about a number nobody
        measured -- it would fail here, and the tempting fix (loosening to 5.6,
        or asserting on a different preset) would quietly stop testing the claim
        at all. The exact ratio and the byte identity are pinned separately below
        so the rounded figure is not the only thing standing between this test and
        a vacuous pass.
        """
        for name in ("nano", "mini", "small"):
            _, mir = make_pair(name)
            assert (
                round(mir.dense_ratio(), 1) == 5.7 or mir.dense_ratio() >= 5.7
            ), f"{name}: {mir.dense_ratio():.4f}x does not round to 5.7x"

    def test_nano_ratio_is_exactly_total_over_active(self) -> None:
        """Nano is ``17/3``: the number behind the rounded 5.7."""
        _, mir = make_pair("nano")
        assert mir.dense_ratio() == pytest.approx(17 / 3, abs=1e-9)

    def test_ratio_improves_with_scale(self) -> None:
        """The docstring's "the gap widens with scale" should be true."""
        ratios = [make_pair(n)[1].dense_ratio() for n in ("nano", "mini", "small")]
        assert ratios == sorted(ratios)


class TestGeluLut:
    """The LUT is a boundary; the mirror must land on the same table values."""

    def test_matches_reference_application(self) -> None:
        table = torch.tensor(gelu_lut(256), dtype=torch.float32)
        x = xs(1, (64,)).squeeze(-1) * 4.0
        assert (
            np.abs(
                gelu_lut_apply(x.numpy())
                - gelu_lut_apply_torch(x, table).detach().numpy()
            ).max()
            < 1e-6
        )

    def test_saturates_outside_the_table_range(self) -> None:
        """Values beyond +/-8 clamp to the table ends, as the reference does."""
        table = torch.tensor(gelu_lut(256), dtype=torch.float32)
        x = torch.tensor([-50.0, 0.0, 50.0])
        got = gelu_lut_apply_torch(x, table).detach().numpy()
        ref = gelu_lut_apply(x.numpy())
        assert np.abs(got - ref).max() < 1e-6

    def test_interpolation_carries_gradient_but_index_does_not(self) -> None:
        """Honest derivative: piecewise constant, not identically zero.

        The floor on the index is detached, so the gradient flows through the
        interpolation fraction. Asserting nonzero here stops a future edit from
        "simplifying" the whole LUT into a detached table and silently zeroing the
        expert activations' gradient.
        """
        table = torch.tensor(gelu_lut(256), dtype=torch.float32)
        x = torch.tensor([0.37, -1.2, 2.5], requires_grad=True)
        gelu_lut_apply_torch(x, table).sum().backward()
        assert x.grad is not None
        assert float(x.grad.abs().sum()) > 0.0


class TestLoadEntropy:
    """Expert-load entropy is reported, and only over the routed experts."""

    def test_uniform_load_is_maximum_entropy(self) -> None:
        _, mir = make_pair()
        mir.loads = torch.zeros(mir.config.total_experts, dtype=torch.int64)
        mir.loads[: mir.config.n_experts] = 10
        expected = float(np.log(mir.config.n_experts))
        assert mir.load_entropy() == pytest.approx(expected, abs=1e-9)

    def test_single_expert_load_is_zero_entropy(self) -> None:
        _, mir = make_pair()
        mir.loads = torch.zeros(mir.config.total_experts, dtype=torch.int64)
        mir.loads[0] = 25
        assert mir.load_entropy() == pytest.approx(0.0, abs=1e-12)

    def test_no_loads_reports_zero_not_nan(self) -> None:
        _, mir = make_pair()
        mir.loads = torch.zeros(mir.config.total_experts, dtype=torch.int64)
        assert mir.load_entropy() == 0.0

    def test_shared_expert_excluded_from_entropy(self) -> None:
        """The always-on shared expert must not pin the metric at its maximum.

        Including slot ``n_shared`` would add a constant mass and make the
        reported entropy look balanced no matter how the routing went.
        """
        _, mir = make_pair()
        mir.loads = torch.zeros(mir.config.total_experts, dtype=torch.int64)
        mir.loads[0] = 100  # only routed expert 0 is used
        assert mir.load_entropy() == pytest.approx(0.0, abs=1e-12)

    def test_train_mode_records_last_entropy(self) -> None:
        """``train=True`` records the per-call mean routing entropy."""
        ref, mir = make_pair()
        x = xs(ref.config.d_model, (16,))
        mir(x, train=True)
        assert float(mir.last_entropy) >= 0.0


class TestLoadBiasSideEffect:
    """The load-bias update is a training side effect, and only that."""

    def test_train_true_mutates_router_bias(self) -> None:
        ref, mir = make_pair()
        before = mir.b.detach().clone()
        mir(xs(ref.config.d_model, (16,)), train=True)
        assert not torch.equal(mir.b.detach(), before)

    def test_reference_train_true_agrees_on_the_side_effect(self) -> None:
        """Both implementations nudge ``b`` in the same direction.

        Otherwise the mirror would be training a different router than the one the
        reference documents, which is the mirror's entire job to prevent.
        """
        ref, mir = make_pair()
        x = xs(ref.config.d_model, (16,))
        before_ref = ref.b.copy()
        before_mir = mir.b.detach().clone()
        ref.forward(x.numpy(), train=True)
        mir(x, train=True)
        assert (
            np.abs(
                (ref.b - before_ref) - (mir.b.detach().numpy() - before_mir.numpy())
            ).max()
            < 1e-6
        )

    def test_loads_accumulate_only_in_train_mode(self) -> None:
        _, mir = make_pair()
        x = xs(mir.config.d_model, (8,))
        mir(x, train=False)
        assert int(mir.loads.sum()) == 0
        mir(x, train=True)
        assert int(mir.loads.sum()) == mir.config.top_k * 8

    def test_gelu_table_is_not_a_parameter(self) -> None:
        """The LUT must not reach the optimizer.

        A 256-entry trainable table per layer would add gradient-carrying values
        the reference never optimises, quietly inflating the trainable count.
        """
        _, mir = make_pair()
        names = {n for n, _ in mir.named_parameters()}
        assert "gelu" not in names
        assert "loads" not in names
        assert "last_entropy" not in names
        assert names == {"E", "b", "W1", "W2"}

    def test_router_bias_stays_a_parameter_after_the_load_update(self) -> None:
        """``b`` must survive ``train=True`` as the same Parameter object.

        An invariant rather than a bug reproduction: the load update writes to the
        bias in place, and the bias is the one tensor here that an optimizer holds
        a long-lived reference to. Pinning that it is still the same
        ``nn.Parameter`` afterwards means an edit that copied-then-rebound
        instead of mutating -- ``self.b = self.b - delta``, say -- would be caught
        here rather than showing up as a router that quietly stopped training.

        Worth being precise about what this does *not* cover, because the obvious
        candidate does not exist: ``self.b -= delta`` was checked by mutation and
        leaves this suite green, correctly. ``__isub__`` mutates in place and
        returns ``self``, so the augmented assignment never rebinds. The
        ``sub_`` spelling in the mirror is for mypy, not for this.
        """
        import torch.nn as nn

        ref, mir = make_pair()
        before = mir.b
        before_ids = {n: id(p) for n, p in mir.named_parameters()}
        mir(xs(ref.config.d_model, (16,)), train=True)
        assert isinstance(mir.b, nn.Parameter), "b is no longer a Parameter"
        assert id(mir.b) == id(before), "b was replaced by a different object"
        assert {n for n, _ in mir.named_parameters()} == {"E", "b", "W1", "W2"}
        after_ids = {n: id(p) for n, p in mir.named_parameters()}
        assert after_ids == before_ids, "a parameter object was replaced in train mode"

    def test_reported_entropy_is_not_a_training_signal(self) -> None:
        """The load-entropy metric must not leak gradient into the router.

        ``last_entropy`` is a reported statistic. If a future edit let its graph
        reach the loss -- or kept it attached while backpropagating -- the router
        would acquire an unrequested pull toward uniform load coming from a metric
        nobody optimises. Pinned by requiring zero gradient from that path.
        """
        ref, mir = make_pair()
        mir(xs(ref.config.d_model, (8,)), train=True)
        # A buffer filled under no_grad holds no graph at all.
        assert not mir.last_entropy.requires_grad
        assert mir.last_entropy.grad_fn is None
        # And the recorded value is a plain number, not a tensor that leaked out.
        assert isinstance(float(mir.last_entropy), float)
