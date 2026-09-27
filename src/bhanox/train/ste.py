"""Straight-through estimators for the quantisation boundaries (law C9, train only).

Runtime stays numpy-only; this package is the optional training extra.

Each function here is the *entire* gradient story for one non-differentiable
boundary in the DeltaBank recurrence: the forward value is the hard one the
deployed model computes, and the backward pass is a deliberate choice about what
the optimiser pretends happened.

They are written as explicit ``x + (f(x) - x).detach()`` expressions rather than
via an ``autograd.Function`` because the forward value and the backward
behaviour are then both readable on one line. That matters more than usual here:
the forward values have to be bit-exact against the numpy reference, and a helper
that hides the arithmetic behind a custom op is a helper nobody can check.

The single most important thing in this file is that composing two
correct-looking STEs can be wrong by a factor of 127. See
:func:`quantize_activation` for the full account; the short version is that
``ste_requantise(ste_round(x * 127))`` is not the same function as quantising.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bhanox.quant.numerics import INT8_MAX

#: ``INT8_MAX`` as an int. The quant module exports it as a float, which is right
#: there and wrong here: an integer path that divides or shifts by a float
#: promotes to float32 and stops being an integer path.
MAX8 = int(INT8_MAX)

__all__ = [
    "MAX8",
    "identity",
    "l2_normalize",
    "quantize_activation",
    "quantize_activation_smooth",
    "sigmoid",
    "ste_clip",
    "ste_requantise",
    "ste_round",
]


def _ste(x: Tensor, forward_value: Tensor) -> Tensor:
    """Value ``forward_value``, gradient 1 with respect to ``x``.

    ``x + (forward_value - x).detach()``. The value is the forward one, and the
    gradient is 1 because the only term carrying graph is the leading ``x``.

    The leading ``x`` is the part that is easy to lose. An earlier version of
    this helper took the already-rounded tensor and re-attached a gradient to
    it, which does not work: ``round`` has severed the graph by then, so the
    result is silently a constant. Straight-through estimators have to be
    written in terms of the *input*, not the output.
    """
    return x + (forward_value - x).detach()


def ste_round(x: Tensor) -> Tensor:
    """Round in forward, pass the gradient straight through.

    The ``.detach()`` is load-bearing, and the failure it guards against is
    worse than a wrong learning rate. The numpy form ``x + (rint(x) - x)`` is
    correct there precisely because numpy has no graph to preserve. Copied into
    torch unchanged it has gradient **zero**: ``round`` has no derivative, so
    the two terms cancel exactly. Nothing raises, the STE quietly stops being
    an estimator, and training appears to run while updating nothing.
    ``test_the_undetached_form_really_would_be_zero`` pins that.
    """
    return _ste(x, torch.round(x))


def ste_clip(x: Tensor, lo: float, hi: float) -> Tensor:
    """Clamp in forward; gradient 1 inside the range and 0 outside.

    Saturation is a projection, so this is the *exact* derivative rather than an
    approximation -- one of the few non-differentiable boundaries in the model
    where that is true. The mask matters: written as ``x + (clamp(x) - x)`` the
    gradient is 1 everywhere, including the saturated channels where the true
    answer is 0, which would hand the optimizer gradient for a value the
    hardware is not actually producing.
    """
    inside = ((x >= lo) & (x <= hi)).to(x.dtype)
    return x * inside + (x.clamp(lo, hi) - x * inside).detach()


def ste_requantise(x: Tensor, divisor: int = MAX8) -> Tensor:
    """Divide by a fixed-point scale in forward; treat the divide as identity.

    The requantisation ``// INT8_MAX`` is a scale factor baked into fixed point.
    Its true derivative is ``1/divisor``, but the standard convention is
    identity, because the value it is attached to has already been rounded and
    the two errors do not compose into anything meaningful. Using the true
    derivative would be *more* accurate about a function nobody is optimising.
    """
    return _ste(x, x / divisor)


def quantize_activation(x: Tensor) -> Tensor:
    """``quantize_activation`` with a straight-through gradient.

    Forward value: exactly the reference's int8 code as a float,
    ``clip(rint(x * 127), -127, 127) / 127``.

    **The whole quantiser is one STE, and that is load-bearing.** The obvious
    construction -- ``ste_requantise(ste_round(x * 127))`` -- is wrong by a
    factor of 127. ``ste_round`` passes the multiply's 127 straight through to
    the gradient and the requantise then contributes its own 1, giving
    ``d k_f / dk == 127`` for every activation.

    That is wrong because of what ``k_f`` *is* here. In the float recurrence the
    quantiser has already been rounded away, so ``k_f`` stands in for ``k``
    itself and its derivative is 1. A 127x error on every activation gradient is
    not a subtle one: it rescales ``W_k``, ``W_q`` and ``W_v`` identically, so
    relative gradient directions stay plausible and the loss still falls. Nothing
    downstream would notice.

    Writing it as a single outer STE pins the scale to identity explicitly
    instead of leaving it as the product of two conventions that happen to
    disagree.
    """
    return _ste(
        x,
        torch.clamp(torch.round(x * INT8_MAX), -float(INT8_MAX), float(INT8_MAX))
        / INT8_MAX,
    )


def quantize_activation_smooth(x: Tensor) -> Tensor:
    """The quantiser with the rounding removed: ``clip(x, -1, 1)``.

    Exists to be inspected rather than optimised. Note it is still a clamp, and
    still saturated for most real inputs, so it is *not* what
    :func:`test_the_surrogate_gradient_equals_the_derivative_of_the_recurrence`
    differences. That test uses identity in both places, because the straight
    through gradient through a clamp is identity while the function's own
    derivative is zero outside the range -- the two are meant to disagree, and
    averaging them into one test proves nothing about either.
    """
    return ste_clip(x, -1.0, 1.0)


def identity(x: Tensor) -> Tensor:
    """No quantisation at all. The pure float recurrence, for differentiation."""
    return x


def l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Torch ``l2_normalize``: near-zero rows stay zero rather than becoming NaN.

    Mirrors the reference's guard, which exists because a zero row would
    otherwise divide by zero and the delta rule's ``beta = 1/||k||^2`` would
    divide by zero again.
    """
    norm = x.norm(dim=-1, keepdim=True)
    return x * torch.where(
        norm > eps, 1.0 / norm.clamp(min=eps), torch.zeros_like(norm)
    )


def sigmoid(z: Tensor) -> Tensor:
    """The reference's read gate.

    Real, not clipped, and the only boundary in the head with no quantisation
    at all.
    """
    return torch.sigmoid(z)
