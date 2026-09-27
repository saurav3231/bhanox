"""Torch mirror of the MicroExpert mixer, for training only (law C9).

Runtime stays numpy-only. As with the DeltaBank mirror, this is a faithful copy of
the reference that also happens to have a backward pass -- not a redesign.

The integer path
----------------
There is nothing to reproduce bit-exactly here, and that is worth being explicit
about rather than discovering later. ``MicroExpertLayer`` stores its weights
*dequantized*: ``absmax_quantize(w).dequantize()``, not the raw int8 codes. Its
module docstring records why -- a float matmul against +/-127 codes gives router
logits a spread of ~500, which saturates the softmax, pins the router to the same
two experts forever, and makes the order-1 load-balancing bias unable to move it.

So the reference forward is already float, and this mirror is too. The int8
regime is a property of the deployed op sequence (``bhanox.audit.INFERENCE_OPS``),
not of the reference arithmetic. Agreement is therefore a float tolerance rather
than a bit-exactness claim, and the tolerance is stated in the tests.

What is discrete, and what that costs
-------------------------------------
Two things in this layer are not differentiable: **which** experts get selected,
and the GELU lookup table's index. Both are boundaries, and both are handled the
same way -- decide with a detached value, carry the gradient through the
surrounding arithmetic.

The selection is genuinely discrete and there is no surrogate to soften it: a
hard top-2 is the architecture. ``top2_balanced`` returns integer indices, so
they are non-differentiable by construction. What *is* differentiable is the gate
weight attached to each pick, and that is the only path the router has:

    g = softmax(E x + b) -> order (detached) -> weights = g[order] renormalised

so ``E`` and ``b`` receive gradient through ``weights``, and ``x`` receives it
through both the router and the shared expert. Detaching the *order* but keeping
the *weights* is the whole trick. The tempting alternative -- detaching the picked
probabilities too, to "avoid a gradient through a hard decision" -- silently
deletes the router's only learning signal, and every agreement test still passes,
because agreement was never the problem. ``test_gradients_reach_router_and_experts``
is the test that closes that hole.

The GELU LUT index is a floor on a clipped position. Floor is a boundary, so the
index is taken detached and the interpolation weights are left in the graph; the
resulting derivative is piecewise constant in ``x``, which is the honest
description of a 256-entry lookup table.

Why ``train=False`` is the default and the tests insist on it
------------------------------------------------------------
``forward(..., train=True)`` mutates ``b`` in place via the load-bias
auto-correction and accumulates ``loads``. That is a *side effect of training*,
not part of the function: the same input gives a different answer on the second
call, because the router has been nudged. An agreement test that ran at
``train=True`` would be comparing two different functions and would pass or fail
for reasons that have nothing to do with the mirror. So every agreement and every
gradient test here runs at ``train=False``, and the side effect is tested
separately, on its own.

The active-set gather is the point of the layer
-----------------------------------------------
Computing all ``n_total`` experts would be correct and would make the byte count
identical to a dense FFN, which is the entire reason this layer exists. Gathering
only the routed rows is what makes the I2 audit show a saving. See
:meth:`MicroExpertLayerMirror.dense_ratio` for the measured figure.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from bhanox.config import BhanoxConfig
from bhanox.mixer.microexpert import MicroExpertLayer, gelu_lut

__all__ = ["MicroExpertLayerMirror", "gelu_lut_apply_torch", "top2_balanced_torch"]

#: Fixed LUT range, mirroring the constants baked into
#: ``bhanox.mixer.microexpert.gelu_lut_apply``. Duplicated rather than imported
#: because the reference keeps them as locals inside that function, and promoting
#: them to module constants would be an edit to the frozen reference for the sole
#: benefit of the mirror.
_LUT_LO = -8.0
_LUT_HI = 8.0
_LUT_N = 256


def gelu_lut_apply_torch(x: Tensor, table: Tensor) -> Tensor:
    """Apply the GELU LUT with linear interpolation, matching the reference.

    Args:
        x: Input tensor of any shape.
        table: ``(n_entries,)`` LUT, e.g. from
            :func:`bhanox.mixer.microexpert.gelu_lut`.

    Returns:
        Interpolated GELU, same shape as ``x``.

    Why the index is detached: ``floor`` is a boundary. The reference computes
    ``i0 = clip(floor(pos), 0, n-2)`` and then interpolates with ``frac = pos - i0``.
    Detaching ``i0`` and keeping ``frac`` in the graph reproduces the reference's
    *value* exactly while making the derivative piecewise constant, which is the
    truth about a lookup table. Carrying the floor's zero gradient through
    ``frac`` would instead make the layer's derivative identically zero, which is
    false -- it is merely discontinuous.
    """
    n = int(table.shape[0])
    pos = (x.clamp(_LUT_LO, _LUT_HI) - _LUT_LO) / (_LUT_HI - _LUT_LO) * (n - 1)
    i0 = torch.clamp(torch.floor(pos), 0, n - 2)
    frac = pos - i0
    lo = table[i0.detach().long()]
    hi = table[(i0.detach() + 1).long()]
    return lo * (1.0 - frac) + hi * frac


def top2_balanced_torch(scores: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """Pick the top-k experts per token and renormalise their gates.

    Args:
        scores: ``(..., n_experts)`` post-softmax scores, experts on the last axis.
        top_k: Experts to activate per token.

    Returns:
        ``(indices, weights)``, both shaped ``scores.shape[:-1] + (top_k,)``, with
        ``weights`` summing to 1 along the last axis.

    Why ``stable=True`` is required, and what it does *not* buy: the reference
    sorts with ``kind="stable"``, so tied scores keep their original expert order.
    Ties are not a rare edge case here -- an untrained router produces them as a
    matter of course, since ``E`` starts random and two experts landing on the
    same logit is entirely ordinary. An unstable sort would be free to disagree
    with the reference on exactly the inputs a fresh model sees most often, and
    the disagreement would look like a routing bug.

    Stated honestly, though: on this backend the flag is not *observably*
    load-bearing. Flipping it to ``stable=False`` leaves the whole suite green,
    because torch's CPU sort happens to be stable for these sizes. So the tie
    tests in ``test_mixer_mirror.py`` pin the required *behaviour* -- mirror
    agrees with reference on ties -- and ``stable=True`` is what turns that
    observed coincidence into a guarantee across backends and devices. A future
    backend whose sort is not incidentally stable would then be caught by the
    agreement tests rather than by a silent routing regression.

    Sorting the negation ascending is the reference's formulation and is kept
    rather than replaced with ``sort(descending=True)``: they agree on order, but
    matching the reference's expression keeps the tie semantics obvious.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k > int(scores.shape[-1]):
        raise ValueError(f"top_k={top_k} exceeds {int(scores.shape[-1])} experts")
    order = torch.argsort(-scores, dim=-1, stable=True)[..., :top_k]
    picked = torch.gather(scores, -1, order)
    total = picked.sum(dim=-1, keepdim=True)
    weights = picked / total.clamp_min(1e-6)
    return order, weights


class MicroExpertLayerMirror(nn.Module):
    """Torch mirror of one :class:`~bhanox.mixer.microexpert.MicroExpertLayer`.

    Args:
        layer: The numpy reference layer to mirror. Weights are copied, not
            shared, so training the mirror cannot mutate the reference.

    Note on ``loads`` and ``last_entropy``: both are buffers, not parameters. They
    are runtime bookkeeping that the checkpoint round-trips, so a resumed run
    reports the same load entropy as an uninterrupted one, but an optimizer must
    never see them.
    """

    # Bare annotations for the tensors registered below. ``nn.Module.__getattr__``
    # is typed as ``Tensor | Module``, so without these every ``self.gelu``,
    # ``self.b -= ...`` and ``self.loads`` would be a type error -- and the real
    # shape of the bug is invisible. Same remedy as ``mirror.py``.
    E: nn.Parameter
    b: nn.Parameter
    W1: nn.Parameter
    W2: nn.Parameter
    gelu: Tensor
    loads: Tensor
    last_entropy: Tensor

    def __init__(self, layer: MicroExpertLayer) -> None:
        super().__init__()
        self.config: BhanoxConfig = layer.config
        self.layer_index = int(layer.layer_index)
        self.n_shared = int(self.config.n_shared_experts)
        self.top_k = int(self.config.top_k)

        def _p(a: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.tensor(np.array(a), dtype=torch.float32))

        self.E = _p(layer.E)
        self.b = _p(layer.b)
        self.W1 = _p(layer.W1)
        self.W2 = _p(layer.W2)

        # The LUT is a constant, not a parameter. It is a buffer so it moves and
        # saves with the module, and deliberately not an ``nn.Parameter``: making
        # a 256-entry table trainable would add 256 gradient-carrying values per
        # layer that the reference never optimises.
        self.register_buffer(
            "gelu", torch.tensor(np.array(gelu_lut(_LUT_N)), dtype=torch.float32)
        )
        self.register_buffer(
            "loads",
            torch.tensor(np.array(layer.loads), dtype=torch.int64),
        )
        self.register_buffer("last_entropy", torch.tensor(float(layer.last_entropy)))

    # -- routing -------------------------------------------------------------

    def route(self, x: Tensor) -> Tensor:
        """Softmax routing scores, ``(n_tokens, n_experts)``.

        Mirrors ``MicroExpertLayer.route`` including its ``logits -= max`` shift
        and its ``max(sum, 1e-6)`` floor. The shift is what keeps ``exp`` from
        overflowing; the floor is the reference's own guard against a degenerate
        all-zero row and is reproduced rather than improved, because a mirror that
        "fixed" it would stop matching the thing it mirrors.
        """
        logits = x @ self.E.T + self.b
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.exp(logits)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # -- forward -------------------------------------------------------------

    def _expert(self, x: Tensor, j: int) -> Tensor:
        """Apply expert ``j`` to every token. Mirrors ``_expert``."""
        h = gelu_lut_apply_torch(x @ self.W1[j].T, self.gelu)
        return h @ self.W2[j].T

    def _routed(self, x: Tensor, idx: Tensor, w: Tensor) -> Tensor:
        """Apply each token's selected expert, gathering only active weights.

        Mirrors ``_routed``. The gather is per token -- ``(T, ...)`` -- not
        per distinct index, which is what makes the touched-byte count
        ``top_k`` experts rather than ``n_total``. Note ``idx`` is offset by
        ``n_shared`` to reach storage, since shared experts occupy slots
        ``0..n_shared-1`` and the routed pool follows.
        """
        rows = idx + self.n_shared
        w1g = self.W1[rows].transpose(1, 2)
        w2g = self.W2[rows]
        h = gelu_lut_apply_torch(torch.einsum("td,tdk->tk", x, w1g), self.gelu)
        y = torch.einsum("tk,tdk->td", h, w2g)
        return y * w[:, None]

    def forward(self, x: Tensor, *, train: bool = False) -> Tensor:
        """Mix shared and top-k routed experts. Mirrors ``MicroExpertLayer.forward``.

        Args:
            x: ``(..., d_model)`` inputs; any leading shape is accepted.
            train: Apply the load-bias update and record loads. Off by default,
                and every agreement and gradient test keeps it off, because this
                flag has a side effect on ``b`` and therefore changes the function
                being mirrored.

        Returns:
            Output of the same shape as ``x``.
        """
        arr = x.reshape(-1, x.shape[-1])
        out = torch.zeros_like(arr)
        for j in range(self.n_shared):
            out = out + self._expert(arr, j)

        scores = self.route(arr)
        indices, weights = top2_balanced_torch(scores, self.top_k)
        if train:
            self.update_load_bias(indices)
            p = scores.detach().mean(dim=0)
            nz = p[p > 0]
            # ``fill_``, not assignment: ``last_entropy`` is a registered buffer so
            # it round-trips through the checkpoint, and assigning a python float
            # to a buffer name raises.
            #
            # The ``detach()`` on ``scores`` is defensive, not load-bearing: the
            # value is immediately reduced to a Python float inside ``no_grad``,
            # so no gradient can reach the router through it either way. Verified
            # by mutation -- dropping the detach leaves the suite green. It is
            # kept because it makes the intent explicit at the point where a
            # future edit would be tempted to keep the tensor instead of the
            # scalar, which *would* start leaking a signal.
            with torch.no_grad():
                self.last_entropy.fill_(float(-(nz * torch.log(nz)).sum()))
        for slot in range(self.top_k):
            out = out + self._routed(arr, indices[:, slot], weights[:, slot])
        return out.reshape(x.shape)

    # -- load balancing ------------------------------------------------------

    def update_load_bias(self, indices: Tensor, *, gamma: float = 1e-3) -> Tensor:
        """Auto-correct the router bias toward uniform load.

        Mirrors ``MicroExpertLayer.update_load_bias``. The in-place ``b`` update
        runs under ``no_grad``: ``b`` is a leaf parameter, and mutating it inside
        the graph would either raise or silently detach the router's gradient.
        Counts are per call rather than lifetime, for the reason the reference
        gives -- a lifetime average cannot correct a one-off imbalance and gets
        weaker the longer training runs.
        """
        flat = indices.reshape(-1)
        with torch.no_grad():
            self.loads.index_add_(0, flat, torch.ones_like(flat))
        counts = torch.bincount(flat, minlength=self.config.n_experts).to(torch.float32)
        fraction = counts / max(int(flat.numel()), 1)
        with torch.no_grad():
            # ``sub_``, not ``self.b -= ...``, and the reason is the type checker
            # rather than the runtime. ``-=`` on a tensor is genuinely in-place --
            # ``__isub__`` mutates and returns ``self``, exactly like numpy's
            # ndarray, so both spellings behave identically here. But mypy reads
            # the augmented assignment as rebinding the name to the *result*,
            # whose type is a plain Tensor, and rejects it against a declared
            # ``nn.Parameter``. ``sub_`` says "mutate this object" with no
            # rebinding for mypy to misread. Verified by mutation: swapping back
            # to ``-=`` leaves the whole suite green.
            self.b.sub_(gamma * (fraction - fraction.mean()))
        return fraction

    def load_entropy(self) -> float:
        """Entropy of the per-expert load distribution, in nats.

        Mirrors the reference: only the routed experts (``0..n_experts-1``) count,
        since the shared expert is always on and would pin the entropy at its
        maximum no matter how the routing went. ``log(n_experts)`` is perfectly
        balanced.
        """
        routed = self.loads[: self.config.n_experts].to(torch.float64)
        total = float(routed.sum())
        if total == 0.0:
            return 0.0
        p = routed / total
        nz = p[p > 0]
        return float(-(nz * torch.log(nz)).sum())

    # -- byte accounting -----------------------------------------------------

    def active_nbytes(self) -> int:
        """int8 weight bytes touched per token.

        The *active* side of the claim: ``n_shared + top_k`` experts' worth of
        weights, since the shared expert always runs and ``top_k`` routed experts
        are gathered.
        """
        per_expert = 2 * self.config.d_model * self.config.d_expert
        return int((self.n_shared + self.top_k) * per_expert)

    def dense_nbytes(self) -> int:
        """int8 weight bytes a dense FFN of equal capacity would touch.

        The *dense* side. ``total_experts`` (routed **and** shared) is the
        like-for-like denominator: a dense FFN of the same capacity runs every
        expert the layer owns, including the shared one.

        Using ``n_experts`` here instead would give 16/3 = 5.33x, which is the
        5.3 figure the reference docstrings quote. That number is an artifact of
        counting the shared expert on the active side while omitting it from the
        dense side, and it understates the saving. The two sides have to be
        measured over the same set of weights or the ratio means nothing.
        """
        cfg = self.config
        return int(2 * cfg.d_model * cfg.d_expert * cfg.total_experts)

    def dense_ratio(self) -> float:
        """``dense_nbytes / active_nbytes`` -- the density saving, > 1."""
        return self.dense_nbytes() / self.active_nbytes()

    # -- convenience ---------------------------------------------------------

    def load_from_numpy(self, layer: MicroExpertLayer) -> None:
        """Copy reference weights in, e.g. after a checkpoint load."""
        with torch.no_grad():
            self.E.copy_(torch.tensor(np.array(layer.E), dtype=torch.float32))
            self.b.copy_(torch.tensor(np.array(layer.b), dtype=torch.float32))
            self.W1.copy_(torch.tensor(np.array(layer.W1), dtype=torch.float32))
            self.W2.copy_(torch.tensor(np.array(layer.W2), dtype=torch.float32))
            self.loads.copy_(torch.tensor(np.array(layer.loads), dtype=torch.int64))

    def extra_repr(self) -> str:
        return (
            f"layer_index={self.layer_index}, n_experts={self.config.n_experts}, "
            f"n_shared={self.n_shared}, top_k={self.top_k}, "
            f"dense_ratio={self.dense_ratio():.3f}x"
        )
