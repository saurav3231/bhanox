"""The mirror must reproduce the reference, and the two halves must be tested
separately.

There are two distinct claims here and they need different instruments:

1. **The mirror's forward equals the reference's forward.** Bit-exact. No
   gradients are involved, so this is an equality test.
2. **The mirror's gradients are the gradients of the float surrogate.** The
   integer forward is a staircase -- perturb a parameter and the loss either
   does not move or jumps a whole step -- so it *cannot* be finite-differenced.
   Any test that tried would either pass vacuously or be flaky depending on
   which side of a step it landed.

Keeping them apart is the point. A single test trying to do both would have to
pick a comparison strict enough for one and loose enough for the other, and
would prove neither.

The third test is the empirical check on the premise the whole design rests on:
that the integers are a fixed-point encoding of the float recurrence. If the
surrogate did not track the integer path, the reframe would be wrong and the
gradient would be describing a function nobody ships.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from bhanox.core.deltabank import DeltaBankHead
from bhanox.quant.numerics import INT8_MAX
from bhanox.train.gradcheck import check_tensor, report
from bhanox.train.mirror import DeltaBankHeadMirror
from bhanox.train.ste import (
    identity,
    quantize_activation,
    ste_clip,
    ste_requantise,
    ste_round,
)

torch.manual_seed(0)


def make_head(
    d_in: int = 12, d_k: int = 4, d_v: int = 6, seed: int = 0
) -> DeltaBankHead:
    """A small head with the reference's own construction, not a stub."""
    rng = np.random.default_rng(seed)
    head = DeltaBankHead(d_k=d_k, d_v=d_v, d_in=d_in, n_banks=8, seed=seed)
    for name, shape in (
        ("W_k", (d_in, d_k)),
        ("W_q", (d_in, d_k)),
        ("W_v", (d_in, d_v)),
        ("W_r", (d_in, d_v)),
    ):
        setattr(head, name, rng.standard_normal(shape).astype(np.float32))
    head.bank_logits = (
        rng.standard_normal((d_k, head.bank_logits.shape[1])) * 0.5
    ).astype(np.float32)
    head.reset()
    return head


def bank_rates_for(head: DeltaBankHead) -> np.ndarray:
    return np.array(
        [1.0 - 2.0**-b for b in range(1, head.bank_logits.shape[1] + 1)],
        dtype=np.float32,
    )


def xs_for(
    head: DeltaBankHead, n: int = 4, batch: int = 2, seed: int = 8
) -> list[Tensor]:
    """A short token sequence as torch tensors, shared by both claim classes."""
    rng = np.random.default_rng(seed)
    return [
        torch.tensor(rng.standard_normal((batch, head.d_in)).astype(np.float32))
        for _ in range(n)
    ]


class TestStraightThroughPrimitives:
    def test_round_has_unit_gradient(self) -> None:
        """``d(ste_round)/dx == 1``.

        The ``.detach()`` is load-bearing. Copied into torch from the numpy form
        -- which has no graph to preserve and so does not need one -- the
        expression ``x + (round(x) - x)`` has gradient **zero**, not two:
        ``round`` has no derivative, so the two terms cancel. Nothing raises.
        The STE quietly stops being an estimator and becomes a no-op, and
        training appears to run while updating nothing at all.
        """
        x = torch.tensor([0.3, 1.7, -2.2], requires_grad=True)
        ste_round(x).sum().backward()
        assert torch.equal(x.grad, torch.ones(3))

    def test_round_value_is_the_rounded_value(self) -> None:
        x = torch.tensor([0.3, 1.7, -2.2], requires_grad=True)
        assert torch.equal(ste_round(x).detach(), torch.tensor([0.0, 2.0, -2.0]))

    def test_the_undetached_form_really_would_be_zero(self) -> None:
        """Pins the trap itself, so the fix is known to be load-bearing: same
        forward value, no gradient.
        """
        x = torch.tensor([0.3], requires_grad=True)
        out = x + (torch.round(x) - x)
        out.sum().backward()
        assert out.item() == 0.0
        assert x.grad.item() == 0.0

    def test_clip_gradient_is_zero_outside_and_one_inside(self) -> None:
        x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
        ste_clip(x, -1.0, 1.0).sum().backward()
        assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 0.0]))

    def test_requantise_keeps_the_value_and_passes_identity(self) -> None:
        x = torch.tensor([254.0, -254.0], requires_grad=True)
        out = ste_requantise(x, 127)
        assert torch.allclose(out, torch.tensor([2.0, -2.0]))
        out.sum().backward()
        assert torch.equal(x.grad, torch.ones(2))


class TestIntegerAgreement:
    """Claim 1: the integer recurrence matches the reference. No gradients here.

    Two different claims live in this class and they need different instruments,
    which is the same distinction the checkpoint tests draw:

    - The **state** is bit-exact. It is integers, it is what compounds across
      steps, and one wrong step is a permanently diverged trajectory. Asserted
      with ``array_equal``, no tolerance.
    - The **read output** is equal only to float32 tolerance. It comes out of a
      float32 matmul, and numpy's BLAS and torch's disagree on the summation
      order, so it lands one ULP apart. Asserted with ``allclose``.

    Conflating those would be the third time this project has had to unpick it.
    """

    def test_state_is_bit_identical_after_one_step(self) -> None:
        """A single step from a zero state, on the state rather than the output.

        Worth being explicit about why the output is not the assertion here: at
        step 0 the state is all zeros, so the read is exactly zero and *any*
        implementation agrees. An output-based one-step test passes without
        having checked anything.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x = np.random.default_rng(1).standard_normal((3, head.d_in)).astype(np.float32)
        mirror.forward_int(torch.tensor(x), torch.tensor(rates))
        head.forward(x, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())

    def test_state_stays_bit_identical_over_many_steps(self) -> None:
        """The claim that actually matters. One agreeing step proves little; the
        recurrence is where a mirror drifts, because a small difference in the
        write compounds through the state and eventually saturates.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        rng = np.random.default_rng(2)
        xs = rng.standard_normal((24, 4, head.d_in)).astype(np.float32)
        worst = 0.0
        for t in range(xs.shape[1]):
            x = xs[:, t]
            want = head.forward(x, rates)
            got = mirror.forward_int(torch.tensor(x), torch.tensor(rates))
            assert np.array_equal(
                head.state, mirror.to_numpy_state()
            ), f"state diverged at step {t}"
            worst = max(worst, float(np.abs(want - got.numpy()).max()))
        assert worst < 1e-6, f"read output drifted {worst} from the reference"

    def test_read_output_agrees_to_float32(self) -> None:
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x = np.random.default_rng(1).standard_normal((3, head.d_in)).astype(np.float32)
        want = head.forward(x, rates)
        got = mirror.forward_int(torch.tensor(x), torch.tensor(rates)).numpy()
        assert np.allclose(
            want, got, rtol=1e-6, atol=1e-7
        ), f"max|diff| = {np.abs(want - got).max()}"

    def test_batch_one_and_batch_four_agree_with_the_reference(self) -> None:
        """The batch axis is where a mirror can cheat, since the state carries a
        sample axis. A mirror that broadcast one query against every state row
        would still pass a batch-1 test.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x = np.random.default_rng(3).standard_normal((4, head.d_in)).astype(np.float32)
        mirror.forward_int(torch.tensor(x), torch.tensor(rates))
        head.forward(x, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())

    def test_a_grown_state_leaves_the_existing_rows_alone(self) -> None:
        """The reference's ``ensure_batch`` copies nothing and resets nothing.
        If the mirror re-zeroed or re-shuffled rows on growth, a batch-1 run
        followed by a batch-4 run would silently lose its state.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x1 = (
            np.random.default_rng(11).standard_normal((1, head.d_in)).astype(np.float32)
        )
        mirror.forward_int(torch.tensor(x1), torch.tensor(rates))
        head.forward(x1, rates)
        before = mirror.to_numpy_state().copy()
        x4 = (
            np.random.default_rng(12).standard_normal((4, head.d_in)).astype(np.float32)
        )
        mirror.forward_int(torch.tensor(x4), torch.tensor(rates))
        head.forward(x4, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())
        assert not np.array_equal(
            before, mirror.to_numpy_state()[:1]
        ), "nothing happened"

    def test_a_single_vector_is_treated_as_one_sample(self) -> None:
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x = np.random.default_rng(4).standard_normal(head.d_in).astype(np.float32)
        got = mirror.forward_int(torch.tensor(x), torch.tensor(rates)).numpy()
        assert got.shape == (head.d_v,), f"expected a 1-D read, got shape {got.shape}"
        head.forward(x, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())

    def test_additive_write_mode_also_agrees(self) -> None:
        """The ablation baseline is a real configuration, not a comment, so the
        mirror has to reproduce it too.
        """
        head = make_head()
        head.write_mode = "additive"
        mirror = DeltaBankHeadMirror(head)
        rates = bank_rates_for(head)
        x = np.random.default_rng(5).standard_normal((2, head.d_in)).astype(np.float32)
        mirror.forward_int(torch.tensor(x), torch.tensor(rates))
        head.forward(x, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())


class TestSurrogateTracksTheIntegerPath:
    """The premise: the integers encode a float recurrence.

    If this fails, the reframe is wrong -- the surrogate would be describing a
    function the deployed model does not compute, and the gradient would be
    confidently answering the wrong question.
    """

    def test_surrogate_is_close_to_the_integer_output(self) -> None:
        """The premise: the integers encode a float recurrence.

        If this fails, the reframe is wrong -- the surrogate would be describing a
        function the deployed model does not compute, and the gradient would be
        confidently answering the wrong question.

        Run over a sequence, because the very first token cannot test this at
        all: the state starts at zero, so the read is exactly zero for any
        implementation and the comparison is ``0 == 0``. Measured as a fraction
        of the reference's own scale, so it stays meaningful on later steps where
        the read is small.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates_np = bank_rates_for(head)
        rates = torch.tensor(rates_np)
        xs = xs_for(head, n=6, batch=3)
        for x in xs:
            want = head.forward(x.numpy(), rates_np)
            out_f, _ = mirror.step(x, rates)
            scale = max(float(np.abs(want).max()), 1e-6)
            rel = (
                float((out_f.detach() - torch.tensor(np.asarray(want))).abs().max())
                / scale
            )
            assert rel < 0.05, f"surrogate drifted {rel:.1%} from the integer path"

    def test_surrogate_state_matches_the_integer_state(self) -> None:
        """In real-valued units, the surrogate's next state should be what the
        integer state encodes. This is the sharpest form of the claim.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = torch.tensor(bank_rates_for(head))
        x = torch.tensor(
            np.random.default_rng(7).standard_normal((2, head.d_in)).astype(np.float32)
        )
        _, s_next = mirror.step(x, rates)
        want = mirror.state_int[:2].to(torch.float32) / INT8_MAX
        assert float((want - s_next.detach()).abs().max()) < 0.05


class TestSurrogateGradients:
    """Claim 2: autograd on the surrogate, against finite differences of the
    surrogate. The integer path is not involved and could not be.
    """

    def _sequence_loss(
        self,
        mirror: DeltaBankHeadMirror,
        rates: Tensor,
        xs: list[Tensor],
        quantize: Callable[[Tensor], Tensor] = quantize_activation,
        reanchor: bool = True,
        saturate: bool = True,
    ) -> Tensor:
        """Loss over a short sequence, driving ``step`` the way training will.

        A sequence rather than a single step, and that is not a detail. On one
        step from a zero state the output cannot depend on ``W_k``, ``W_v`` or
        ``bank_logits`` at all -- they write the state, which is read on the *next*
        step -- so a one-step gradient test would correctly report them as zero
        and would be asserting a bug. It also means the re-anchoring, the part of
        the design that is easy to get wrong, is never exercised.
        """
        shadow: Tensor | None = None
        # Reset the integer state, or the loss would not be a function of the
        # weights alone. It advances inside ``step``, so without this the analytic
        # pass and every finite-difference probe would start from different
        # states and be differencing two different functions -- which is a
        # confident, well-conditioned, entirely spurious gradient mismatch.
        mirror.state_int.zero_()
        total = mirror.W_k.new_zeros(())
        for x in xs:
            out, shadow = mirror.step(x, rates, shadow, quantize, reanchor, saturate)
            total = total + (out**2).sum()
        return total

    def test_the_activation_quantiser_has_unit_gradient(self) -> None:
        """``d/dx clip(round(x * 127)) / 127 == 1``, in the interior.

        Pinned directly, because the natural way to write this --
        ``ste_requantise(ste_round(x * 127))`` -- yields **127**. ``ste_round``
        passes the multiply's 127 through to the gradient and the requantise
        contributes its own 1 on top.

        A factor of 127 on every activation gradient is the kind of error that
        never shows up: it rescales ``W_k``, ``W_q`` and ``W_v`` identically, so
        gradient directions stay plausible, the loss still falls, and the only
        symptom is that the effective learning rate is not the one in the config.
        So it gets its own test rather than being left to whatever the aggregate
        gradient check happens to notice.
        """
        x = torch.tensor([0.13, -0.4, 0.72, 0.05], requires_grad=True)
        quantize_activation(x).sum().backward()
        assert torch.equal(
            x.grad, torch.ones(4)
        ), f"grad was {x.grad.tolist()}, expected all ones"

    def test_the_activation_quantiser_forward_is_the_reference_code(self) -> None:
        """The forward value must still be exactly ``clip(rint(x*127))/127``.

        Worth asserting separately from the gradient, because the two come from
        different parts of the STE and a fix to either can silently break the
        other.
        """
        x = torch.tensor([0.0, 0.5 / 127, -2.0, 0.9])
        want = np.clip(np.rint(x.numpy() * 127.0), -127.0, 127.0) / 127.0
        assert np.allclose(quantize_activation(x).detach().numpy(), want, atol=0.0)

    def test_the_surrogate_gradient_equals_the_derivative_of_the_recurrence(
        self,
    ) -> None:
        """Central differences must reproduce the analytic gradient of the graph.

        Checked on the *pure* float recurrence -- no rounding, no saturation, no
        re-anchoring -- because that is the one configuration where a derivative
        exists to be compared. It has real teeth: it is what catches a
        mis-scaled straight-through gradient, a read-after-write ordering
        mistake, or a weight accidentally detached.

        The quantisation boundaries are deliberately *excluded* rather than
        smoothed over, and they are not a small exclusion. The straight-through
        gradient through a clamp is identity while the function's own derivative
        is zero outside the range; the two are supposed to disagree. A
        "smoothed" surrogate that keeps the clamp is still saturated for most
        real inputs, and differencing it against an identity-through-clamp
        analytic gradient produces nonsense -- large, well-conditioned,
        confidently wrong numbers.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = torch.tensor(bank_rates_for(head))
        xs = xs_for(head)

        def loss() -> float:
            with torch.no_grad():
                return float(
                    self._sequence_loss(mirror, rates, xs, identity, False, False)
                )

        mirror.zero_grad()
        self._sequence_loss(mirror, rates, xs, identity, False, False).backward()

        results = []
        for name in ("W_k", "W_v", "W_q", "W_r", "bank_logits"):
            t = getattr(mirror, name)
            results += check_tensor(
                name, t.detach().numpy(), t.grad.detach().numpy(), loss, count=4
            )
        assert not [r for r in results if r.verdict == "mismatch"], report(results)

    def test_the_reanchored_loss_is_mostly_flat_under_a_weight_nudge(self) -> None:
        """The reason the test above has to drop the boundaries, asserted.

        Re-anchored, the forward value is the integer trajectory, which is
        piecewise constant in the weights: a small nudge to ``W_k`` usually
        changes no ``k8`` at all, so the loss is flat and the numeric gradient
        is zero. It jumps occasionally, when the nudge happens to push a code
        across a rounding boundary -- hence "mostly" rather than "always".

        The nudge has to be small relative to one quantisation step. ``W_k`` is
        standard normal and the code is ``round(x @ W_k * 127)``, so a 1e-3
        nudge is worth roughly 0.4 of a code unit: it flips a boundary for a
        large fraction of entries, and the "staircase" disappears. 1e-5 is
        worth 0.004 of a code unit, which is far enough inside a step.

        If this ever stopped holding, the re-anchored forward would have become
        differentiable and the finite-difference test could be extended to cover
        the real training path. That would be a genuine improvement, and this
        test is what would notice it had happened.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = torch.tensor(bank_rates_for(head))
        xs = xs_for(head, n=3, batch=1)
        w = mirror.W_k.detach()

        with torch.no_grad():
            base = float(self._sequence_loss(mirror, rates, xs))
            moved = 0
            for j in range(w.shape[1]):
                for i in range(w.shape[0]):
                    keep = w[i, j].item()
                    w[i, j] = keep + 1e-5
                    if float(self._sequence_loss(mirror, rates, xs)) != base:
                        moved += 1
                    w[i, j] = keep
        fraction = moved / w.numel()
        assert fraction < 0.5, (
            f"{moved}/{w.numel()} nudges moved a re-anchored loss; "
            "the re-anchored forward no longer looks like a staircase"
        )

    def test_gradients_reach_every_learned_tensor(self) -> None:
        """A silently-zero gradient is the classic mirror bug: the model trains,
        reports a falling loss on something else, and nothing complains.

        ``W_k``, ``W_v`` and ``bank_logits`` are the ones worth guarding, because
        they only reach the loss through the *state*, and the state is the part
        that gets re-anchored to a constant. Get that wrong and they go silently
        to zero while every agreement test still passes.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates = torch.tensor(bank_rates_for(head))
        self._sequence_loss(mirror, rates, xs_for(head)).backward()
        for name in ("W_k", "W_q", "W_v", "W_r", "bank_logits"):
            g = getattr(mirror, name).grad
            assert g is not None, f"{name} received no gradient at all"
            assert float(g.abs().max()) > 0, f"{name} gradient is identically zero"

    def test_the_shadow_does_not_change_the_forward_values(self) -> None:
        """Re-anchoring means the shadow influences gradients and nothing else.

        If a shadow could move the output, the forward would be describing a
        trajectory the integer path never took, which is the failure the whole
        design exists to prevent. Compare against the same run with no shadow.
        """
        head = make_head()
        rates = torch.tensor(bank_rates_for(head))
        xs = xs_for(head, n=3, batch=1)

        bare = DeltaBankHeadMirror(head)
        with torch.no_grad():
            shadow = None
            want = [float((bare.step(x, rates, shadow)[0] ** 2).sum()) for x in xs]
            assert bare.to_numpy_state().tolist() is not None

        with_shadow = DeltaBankHeadMirror(head)
        with torch.no_grad():
            # A deliberately wrong shadow. If forward values moved, the two runs
            # would disagree -- which is the property under test.
            shadow = torch.full((1, head.d_k, head.d_v), 0.5)
            got = []
            for x in xs:
                out, shadow = with_shadow.step(x, rates, shadow)
                got.append(float((out**2).sum()))

        for t, (a, b) in enumerate(zip(want, got, strict=True)):
            assert (
                abs(a - b) < 1e-9
            ), f"step {t}: shadow moved the forward value ({a} vs {b})"
        # And the integer state must match too, since the surrogate is not
        # supposed to be able to influence it at all.
        assert np.array_equal(bare.to_numpy_state(), with_shadow.to_numpy_state())

    def test_surrogate_never_mutates_the_integer_state_itself(self) -> None:
        """The integer advance belongs to :meth:`step` alone, and must be exactly
        the reference's -- not the surrogate's float approximation of it. Stepped
        over several tokens, because a single step from a zero state writes a
        state that a float recurrence would also produce closely enough to hide
        the difference.
        """
        head = make_head()
        mirror = DeltaBankHeadMirror(head)
        rates_np = bank_rates_for(head)
        rates = torch.tensor(rates_np)
        xs = xs_for(head, n=3, batch=2)
        for x in xs:
            head.forward(x.numpy(), rates_np)
            mirror.step(x, rates)
        assert np.array_equal(head.state, mirror.to_numpy_state())
