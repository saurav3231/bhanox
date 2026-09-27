"""Direct tests for the straight-through estimators in ``bhanox.train.ste``.

The DeltaBank estimators are covered through the mirror that uses them, but the
two *decision-boundary* estimators -- :func:`ste_gt` and :func:`ste_ge` -- were
added for PulseGate and have properties the rounding ones do not, so they are
tested here rather than only through their one consumer.

The property that matters most is the last one: a threshold estimator has to
send gradient to the *threshold*. An estimator written on the input margin alone
has the right forward value, no gradient on the threshold at all, and passes
every agreement test in the suite, because agreement tests only look at forward
values. That failure is invisible until the thresholds are discovered to still
hold their initial values after a training run.
"""

from __future__ import annotations

import pytest
import torch

from bhanox.train.ste import ste_ge, ste_gt

torch.manual_seed(0)


# -- forward values --------------------------------------------------------


@pytest.mark.parametrize("margin", [-3.0, -1e-6, 0.0, 1e-6, 2.5])
def test_ste_gt_is_strictly_greater_than_zero(margin: float):
    """``ste_gt`` is ``margin > 0``: zero stays zero.

    The strictness is the whole reason it is a separate function from
    :func:`ste_ge`. PulseGate wakes on ``delta > tau_hi`` but sleeps on
    ``_quiet >= sleep_after``, so a mirror that collapsed the two would make the
    gate sleep a step late.
    """
    x = torch.tensor([margin], requires_grad=True)
    out = ste_gt(x)
    assert out.item() == (1.0 if margin > 0.0 else 0.0)


@pytest.mark.parametrize("margin", [-3.0, -1e-6, 0.0, 1e-6, 2.5])
def test_ste_ge_is_greater_than_or_equal_to_zero(margin: float):
    """``ste_ge`` is ``margin >= 0``: zero is one."""
    x = torch.tensor([margin], requires_grad=True)
    out = ste_ge(x)
    assert out.item() == (1.0 if margin >= 0.0 else 0.0)


def test_gt_and_ge_differ_only_at_exactly_zero():
    """The two agree everywhere except the boundary, which is the point.

    Sampling around zero and comparing the two estimators elementwise is the
    direct version of the claim: one element disagrees, and it is the zero.
    """
    margins = torch.linspace(-0.5, 0.5, 201, dtype=torch.float32)
    strict = ste_gt(margins)
    loose = ste_ge(margins)
    disagree = (strict != loose).nonzero().flatten()
    assert margins[disagree].tolist() == [0.0]


def test_values_are_exactly_zero_or_one():
    """No fractional leakage.

    The PulseGate mirror combines these with ``*``, ``max`` and ``1 - x`` to
    rebuild boolean logic, so a value like 0.9997 would not round-trip to a mask.
    """
    margins = torch.linspace(-3.0, 3.0, 1001, dtype=torch.float32)
    for est in (ste_gt, ste_ge):
        assert set(est(margins).unique().tolist()) <= {0.0, 1.0}


# -- gradients -------------------------------------------------------------


@pytest.mark.parametrize("margin", [-2.0, -0.25, 0.0, 0.25, 2.0])
@pytest.mark.parametrize("est", [ste_gt, ste_ge])
def test_gradient_is_one_everywhere_including_at_the_boundary(margin: float, est):
    """Unit gradient regardless of which side of the boundary the value falls on.

    Including at exactly zero. A subgradient is still a choice, and passing the
    gradient through at the discontinuity -- rather than zeroing it, as one
    comfortably might -- keeps the estimator well defined on a lattice that
    really does land on the threshold, which the quiet counter does constantly.
    """
    x = torch.tensor([margin], requires_grad=True)
    est(x).sum().backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0)


def test_gradient_flows_to_the_threshold_and_not_only_the_input():
    """The reason these are written against the signed margin.

    A threshold comparison has to put the learnable quantity inside the margin.
    Here ``tau`` is a leaf, and the gradient has to arrive there. This is the
    failure the governor mirror's threshold tests would not catch, since those
    only compare forward masks.
    """
    tau = torch.tensor([1.0], requires_grad=True)
    delta = torch.tensor([0.5], requires_grad=True)
    out = ste_gt(delta - tau)
    out.sum().backward()
    assert tau.grad is not None, "the threshold never received a gradient"
    assert tau.grad.item() == pytest.approx(-1.0), "d margin / d tau must be -1"
    assert delta.grad is not None and delta.grad.item() == pytest.approx(1.0)


def test_sleep_side_margin_puts_the_gradient_on_tau_lo_with_a_positive_sign():
    """``tau_lo - delta`` gives ``d margin / d tau_lo == +1``.

    Pinned as a pair with the wake side above, because the sign is the part that
    a "clean up the sign" refactor would break. Raising ``tau_lo`` should make
    channels *less* likely to be judged quiet, and the sign follows from that.
    """
    tau_lo = torch.tensor([0.25], requires_grad=True)
    delta = torch.tensor([0.1], requires_grad=True)
    ste_gt(tau_lo - delta).sum().backward()
    assert tau_lo.grad.item() == pytest.approx(1.0)


def test_gradcheck_is_inapplicable_and_that_is_expected():
    """These cannot be gradchecked, and the reason is worth stating.

    ``torch.autograd.gradcheck`` estimates the derivative by perturbing the
    input. The forward here is a step function, so a perturbation of ``eps``
    across a boundary changes the output by a full 1.0 and the numerical
    jacobian comes back as either 0 or ``1/eps``. Pinned so that a future
    "let me just make it gradcheckable" refactor has to argue with the reason
    rather than rediscover it: softening the comparison to make gradcheck pass
    would put a real fractional value in a mask that has to be exactly 0 or 1.

    The analytic gradient is asserted directly above, at unit magnitude on both
    sides of the boundary and at the boundary itself.
    """
    x = torch.tensor([0.3, 1.7, -2.0], dtype=torch.float64, requires_grad=True)
    for est in (ste_gt, ste_ge):
        with pytest.raises(RuntimeError, match=r"[Jj]acobian"):
            torch.autograd.gradcheck(est, (x,), eps=1e-6)


# -- re-anchoring, the pattern the mirror relies on -----------------------


def test_reanchored_form_keeps_the_value_and_the_gradient():
    """``hard + (ste - ste.detach())`` is the mirror's counter idiom.

    The governor mirror rebuilds an integer quiet counter this way so the
    gradient survives a comparison against integer state. This pins both halves
    at once: the value is the hard one, and the gradient is still live. Losing
    either half is a real failure -- the first makes the forward wrong, the
    second silently freezes a threshold.
    """
    quiet_hard = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    delta = torch.tensor([0.1, 5.0, 0.1], requires_grad=True)
    tau_lo = torch.tensor([0.25], requires_grad=True)
    comparison = ste_gt(tau_lo - delta)

    value = torch.where(comparison.detach() > 0.5, quiet_hard + 1.0, quiet_hard * 0.0)
    out = value + (comparison - comparison.detach())
    # Element 0: quiet, counter 1 -> 2. Element 1: loud, counter 0 -> reset to 0.
    # Element 2: quiet, counter 0 -> 1.
    assert out.tolist() == [2.0, 0.0, 1.0], "the value must be the hard counter"
    out.sum().backward()
    assert tau_lo.grad is not None and tau_lo.grad.abs().sum() > 0.0


def test_composes_with_boolean_arithmetic():
    """The estimators have to survive being used as float booleans.

    The mirror's whole approach is running the state machine in float, where
    ``*``, ``max`` and ``1 - x`` stand in for ``&``, ``|`` and ``~``. If these did
    not produce clean 0/1, none of that would reproduce the reference exactly.
    """
    delta = torch.tensor([-1.0, 0.0, 1.0])
    protected = torch.tensor([0.0, 1.0, 0.0])
    woke = torch.maximum(ste_gt(delta), protected)
    assert woke.tolist() == [0.0, 1.0, 1.0]
    assert (woke * (1.0 - protected)).tolist() == [0.0, 0.0, 1.0]
    assert woke.max().item() == 1.0
    assert woke.min().item() == 0.0
