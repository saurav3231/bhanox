"""Torch mirror of the DeltaBank head, for training only (law C9).

Runtime stays numpy-only. This exists so a gradient can be computed, and it is
worth being precise about what it is: **not** a differentiable rewrite of the
integer recurrence, but a faithful copy of it that also happens to have a
backward pass.

The integer path
----------------
The forward pass here reproduces the reference's integer operations exactly, so
its output is bit-identical to numpy's and can be asserted to be. That part is
not a design choice, it is the requirement: a mirror that merely approximates the
reference would train a different function than the one the benchmarks measure,
and every number in ``docs/benchmarks.md`` would stop describing the model that
ships.

Why integers are a problem at all
---------------------------------
``state`` holds ``127 * S`` in int32, the read divides by ``127**2``, the write
requantises with ``// 127``, and everything saturates into int8. None of that has
a derivative. So the question is not "how do we differentiate integer arithmetic"
-- we do not, and there is nothing there to find. The question is what the
integers are *encoding*, because that is what has a derivative.

They are a fixed-point encoding of the delta rule. The reference's own write
path, read as arithmetic rather than as code::

    decayed  = qmul(state, lam_q)            # Diag(lambda) S
    scaled_e = qmul(beta_q, e8)              # beta * e,  e = v - S^T k
    update   = (k8[:, :, None] * scaled_e[:, None, :]) // INT8_MAX
    state\'   = saturate_int8(decayed + update)

is ``S <- Diag(lambda) S + k (x) beta (v - S^T k)``. The ``// INT8_MAX`` is the
requantisation every int8 GEMM performs, already documented as such in
``deltabank.py``. So the integers are a Q-format rendering of a float
recurrence, and the backward pass differentiates *that*.

Two claims about the surrogate, and a trap
-----------------------------------------
**The value is re-anchored every step, and that is deliberate.** A free running
float shadow would produce gradients describing a trajectory the deployed model
never walked. So the value used at every step is the real integer state, and the
gradient path is carried alongside it. Forward and backward agree on *which*
trajectory is being optimised and differ only in how the boundary is crossed.

**Why the state is not simply detached.** It is tempting, and it is a trap. A
detached state makes the current step's output independent of ``W_k``, ``W_v``
and ``bank_logits``, which write the state rather than read it, so those tensors
receive no gradient at all and the memory becomes permanently unwritable. Every
agreement test still passes in that arrangement, because agreement was never the
problem. ``test_gradients_reach_every_learned_tensor`` exists to close that hole.

**This is a surrogate, and the bias is real.** The true derivative through the
integer dynamics is zero almost everywhere -- the state is a piecewise constant
function of the parameters, discontinuous on a set of measure zero -- so there
is no exact gradient to recover. This is the same approximation the STE makes at
every other boundary, stated here rather than hidden. It also means the
re-anchored forward *cannot* be finite-differenced, which is a property worth
knowing before spending an afternoon on it.

**Why BPTT does not explode.** With unit-norm keys, ``beta = eta`` and the
error-correcting factor is ``I - eta k k^T``, whose eigenvalues are ``1`` and
``1 - eta``. At the reference default ``eta = 0.5`` that is ``{1, 0.5}``, and the
decay banks are in ``[0, 1)``, so the whole map is non-expansive. The delta
rule's error-correcting structure is what makes this stable, not luck.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn

from bhanox.core.deltabank import _RECIP, DeltaBankHead
from bhanox.quant.numerics import INT8_MAX, to_q
from bhanox.train.ste import (
    MAX8,
    l2_normalize,
    quantize_activation,
    sigmoid,
    ste_clip,
    ste_round,
)

__all__ = ["DeltaBankHeadMirror"]


class DeltaBankHeadMirror(nn.Module):
    """Torch mirror of one :class:`DeltaBankHead`.

    Holds three parallel representations, and keeping them separate is the point:

    - the ``nn.Parameter`` copies of ``W_k``/``W_q``/``W_v``/``W_r`` and
      ``bank_logits``, which is what an optimizer updates;
    - ``state_int``, the authoritative int32 recurrent state, advanced by the
      exact integer recurrence;
    - the float surrogate, whose *value* is re-anchored to ``state_int`` at every
      step and whose gradient is carried alongside it.

    Args:
        head: The numpy reference head to mirror. Weights are copied, not
            shared, so training the mirror cannot mutate the reference.
    """

    # Bare annotations for the buffers registered below. ``nn.Module.__getattr__``
    # is typed as ``Tensor | Module``, so without these every ``self.state_int``
    # would be a type error and the real shape of the bug is invisible.
    state_int: Tensor
    recip: Tensor

    def __init__(self, head: DeltaBankHead) -> None:
        super().__init__()
        self.head = head
        self.d_in = head.d_in
        self.d_k = head.d_k
        self.d_v = head.d_v
        self.eta = float(head.eta)
        self.normalize_keys = bool(head.normalize_keys)
        self.use_read_gate = bool(head.use_read_gate)
        self.write_mode = head.write_mode

        def _p(a: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.tensor(np.array(a), dtype=torch.float32))

        self.W_k = _p(head.W_k)
        self.W_q = _p(head.W_q)
        self.W_v = _p(head.W_v)
        self.W_r = _p(head.W_r)
        self.bank_logits = _p(head.bank_logits)

        self.register_buffer(
            "state_int", torch.tensor(np.array(head.state), dtype=torch.int32)
        )
        self.register_buffer(
            "recip", torch.tensor(np.array(_RECIP), dtype=torch.float32)
        )

    def ensure_batch(self, batch: int) -> None:
        """Grow the state to hold ``batch`` independent streams.

        Mirrors ``DeltaBankHead.ensure_batch``: new rows start at zero, which is
        exactly a fresh stream's state, and existing rows are never touched.

        This is not a convenience. A mirror that skipped it would either fail on
        a batch larger than the reference's current state, or -- worse, if it
        broadcast -- return an answer per state row for one token.
        """
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        have = int(self.state_int.shape[0])
        if batch <= have:
            return
        grown = torch.zeros((batch, self.d_k, self.d_v), dtype=torch.int32)
        grown[:have] = self.state_int
        self.state_int = grown

    # -- shared pieces -------------------------------------------------------

    def decay(self, bank_rates: Tensor) -> Tensor:
        """Learned per-channel decay, in float.

        Mirrors ``DeltaBankHead.decay``: softmax over the decay-bank prior in
        log space. This is genuinely differentiable, so no STE is needed here --
        only the Q16 rounding downstream is a boundary.
        """
        w = torch.softmax(self.bank_logits, dim=0)
        return (w * bank_rates[:, None]).sum(dim=0)

    def _beta(self, k_sq: Tensor) -> Tensor:
        """``beta = RECIP[round(||k||^2)] * eta``, with the LUT index STE'd.

        The index is a table lookup on a rounded value, which is the one piece
        of the write path that is not a fixed-point scale. Straight-through on
        the index, so the gradient reaches ``eta`` and the keys.

        Returns ``(B, 1)`` so it broadcasts against the ``(B, d_v)`` error term.
        """
        index = ste_round(k_sq).clamp(1.0, 255.0)
        # The reference indexes ``_RECIP[index - 1]``; a LUT gather is not
        # differentiable, so the gather is done on detached indices and the
        # gradient is carried by the surrounding arithmetic.
        idx = index.detach().long() - 1
        return (self.recip[idx] * self.eta)[:, None]

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k = x @ self.W_k
        q = x @ self.W_q
        v = x @ self.W_v
        if self.normalize_keys:
            k = l2_normalize(k)
            q = l2_normalize(q)
        return k, q, v

    # -- the integer path, bit-exact -----------------------------------------

    def _codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The three int8 activation codes, as int32 tensors.

        ``quantize_activation`` ends in ``.astype(np.int32)`` and the rest of the
        path is int32, so producing floats here would both lose exactness and make
        the ``>> 16`` shifts fail outright.
        """
        k, q, v = self._project(x)
        k8 = torch.clamp(torch.round(k * INT8_MAX), -MAX8, MAX8).to(torch.int32)
        q8 = torch.clamp(torch.round(q * INT8_MAX), -MAX8, MAX8).to(torch.int32)
        v8 = torch.clamp(torch.round(v * INT8_MAX), -MAX8, MAX8).to(torch.int32)
        return k8, q8, v8

    @torch.no_grad()
    def forward_int(self, x: Tensor, bank_rates: Tensor) -> Tensor:
        """Advance the integer state exactly as the reference does.

        No gradient. This is the authoritative trajectory; the surrogate is
        anchored to whatever this produces.
        """
        if x.ndim == 1:
            x = x[None, :]
            single = True
        else:
            single = False
        self.ensure_batch(int(x.shape[0]))
        state = self.state_int[: x.shape[0]]
        k8, q8, v8 = self._codes(x)

        # Integer einsum, not a float one. The magnitudes are small enough
        # (127*127*d_k ~ 2.6e5) that int64 is exact, and staying in integers is
        # the only way "bit-exact" means anything: a float accumulation would
        # agree on almost every step and disagree on the ones that matter.
        #
        # ``MAX8`` rather than ``INT8_MAX`` throughout, because ``INT8_MAX`` is
        # a Python *float* (127.0) and dividing or shifting an int64 tensor by a
        # float promotes the whole expression to float32. The values survive --
        # these magnitudes are exactly representable -- but the dtype does not,
        # and an integer path that quietly runs in float is not an integer path.
        acc_q = torch.einsum("bkv,bk->bv", state.to(torch.int64), q8.to(torch.int64))
        acc_k = torch.einsum("bkv,bk->bv", state.to(torch.int64), k8.to(torch.int64))
        # Floor division, matching numpy's ``//``. Torch's ``//`` floors on
        # integer tensors (unlike C), and the accumulators go negative.
        e8 = torch.clamp(v8 - acc_k // MAX8, -128, 127)
        state = self._write_int(state, k8, e8, v8, bank_rates)
        # float64, because the reference's ``int32 / float(127**2)`` promotes to
        # float64 and the result stays that wide through the read gate. Computing
        # this in float32 agrees to within one ULP and disagrees on every step
        # after the first, which is a worse failure than an obvious exception:
        # the first step is exactly zero and matches perfectly.
        r = acc_q.to(torch.float64) / float(MAX8**2)

        self.state_int[: x.shape[0]] = state
        out = r * sigmoid(x @ self.W_r).to(torch.float64) if self.use_read_gate else r
        return out[0] if single else out

    def _write_int(
        self, state: Tensor, k8: Tensor, e8: Tensor, v8: Tensor, bank_rates: Tensor
    ) -> Tensor:
        lam_q = torch.tensor(
            np.array(to_q(self.decay(bank_rates).detach().numpy())), dtype=torch.int32
        )
        k_sq = (k8.float() * k8.float()).sum(dim=1) / float(MAX8**2)
        index = torch.clamp(torch.round(k_sq), 1.0, 255.0).long()
        beta_q = torch.tensor(
            np.array(to_q(self.recip[index - 1].numpy().astype(np.float64) * self.eta)),
            dtype=torch.int32,
        )
        if self.write_mode == "additive":
            e8 = v8
        decayed = (state * lam_q[None, :, None]) >> 16
        scaled_e = (beta_q[:, None] * e8) >> 16
        update = (k8[:, :, None] * scaled_e[:, None, :]) // MAX8
        return torch.clamp(decayed + update, -128, 127)

    # -- the float surrogate, for gradients ----------------------------------

    def surrogate(
        self,
        x: Tensor,
        bank_rates: Tensor,
        s: Tensor,
        quantize: Callable[[Tensor], Tensor] = quantize_activation,
        saturate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Differentiate the recurrence the integers encode, from a given state.

        ``s`` is the state to read from, in real units (``state_int / 127``), and
        may be part of an autograd graph. Returns ``(output, next_state)``.

        Taking the state as an argument rather than reading ``self.state_int``
        here is deliberate. The integer path mutates ``state_int`` as it steps,
        so a surrogate that reached for it would silently read the *post*-write
        state and compute a gradient for a step that never happened. The caller
        owns the ordering, and ``step`` gets it right.

        ``quantize`` is a seam for testing only, and defaults to the real
        quantiser. It exists so the gradient claim can be checked against a
        forward that has no ``round`` in it; see :func:`_quantize_activation_smooth`.
        """
        k, q, v = self._project(x)

        k_f = quantize(k)
        q_f = quantize(q)
        v_f = quantize(v)

        k_sq = (k_f * k_f).sum(dim=1)
        beta = self._beta(k_sq)

        # Read before write, matching the reference's ordering: both the read and
        # the surprise come from the pre-write state.
        r = torch.einsum("bkv,bk->bv", s, q_f)
        e = v_f - torch.einsum("bkv,bk->bv", s, k_f)

        lam = self.decay(bank_rates)
        s_next = lam[None, :, None] * s + k_f[:, :, None] * (beta * e)[:, None, :]
        if saturate:
            s_next = ste_clip(s_next, -1.0, 1.0)  # saturation, the int8 range
        out = r * sigmoid(x @ self.W_r) if self.use_read_gate else r
        return out, s_next

    def step(
        self,
        x: Tensor,
        bank_rates: Tensor,
        shadow: Tensor | None = None,
        quantize: Callable[[Tensor], Tensor] = quantize_activation,
        reanchor: bool = True,
        saturate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """One training step: integer state advances, surrogate carries gradient.

        Returns ``(output, next_shadow)``. Feed ``next_shadow`` back in on the
        next token; leave it ``None`` to start a fresh sequence.

        The re-anchoring here is the whole design, and it is worth spelling out
        because the two obvious alternatives are both wrong:

        **Do not use the integer state as a detached constant.** That is the
        first thing that looks right, and it silently trains a model whose
        memory cannot be written. With the state detached, ``W_k``, ``W_v`` and
        ``bank_logits`` affect nothing observable in this step's output, so they
        receive no gradient at all -- ``test_gradients_reach_every_learned_tensor``
        is the test that caught it. The recurrence becomes untrainable while
        every other test still passes.

        **Do not let the float shadow run free.** Then the backward pass
        describes a trajectory the deployed model never walked, and the gradient
        optimises a different model than the one that gets benchmarked.

        So the value is re-anchored and the graph is kept:
        ``state_int + (shadow - shadow.detach())`` has the integer state as its
        value and unit gradient with respect to ``shadow``. Every forward step
        therefore reads the trajectory that actually happened, while gradients
        still flow back through the float recurrence to the step that produced
        the shadow. The two agree on value, differ only in what they carry.

        ``reanchor=False`` drops the re-anchoring and follows the float shadow
        freely. It exists only to be finite-differenced, and it is *not* a
        training mode: the re-anchored forward is a staircase in the weights --
        a 1e-3 nudge to ``W_k`` almost never changes ``k8``, so the loss does not
        move and no finite difference can see anything. Un-anchored and
        un-rounded, the same graph is smooth and its derivative exists, so that
        is the configuration a central difference can actually check. It
        verifies the autograd plumbing; it says nothing about the STE bias,
        which is documented above rather than verified.
        """
        if x.ndim == 1:
            x = x[None, :]
        self.ensure_batch(int(x.shape[0]))
        state_int = self.state_int[: x.shape[0]]

        s = state_int.to(torch.float32) / INT8_MAX
        if shadow is not None:
            # Re-anchored: value from the integer trajectory, gradient from the
            # shadow. Not re-anchored (``reanchor=False``): follow it freely.
            s = s + (shadow - shadow.detach()) if reanchor else shadow

        out, next_shadow = self.surrogate(x, bank_rates, s, quantize, saturate)

        # Advance the authoritative state last, off the pre-write copy, so the
        # surrogate above read the same state the reference would have.
        with torch.no_grad():
            # Only the write-side codes are needed here; the read happened above,
            # in the surrogate, and the two must not disagree about it.
            k8, _q8, v8 = self._codes(x)
            acc_k = torch.einsum(
                "bkv,bk->bv", state_int.to(torch.int64), k8.to(torch.int64)
            )
            e8 = torch.clamp(v8 - acc_k // MAX8, -128, 127)
            new_int = self._write_int(state_int, k8, e8, v8, bank_rates)
            self.state_int[: x.shape[0]] = new_int

        return out, next_shadow

    def forward(self, x: Tensor, bank_rates: Tensor) -> tuple[Tensor, Tensor]:
        """Debug view: the integer output and the surrogate's, side by side.

        Returns ``(integer_output, surrogate_output)``. They should agree to
        within quantisation error, which is the empirical check that the integers
        really are a fixed-point rendering of the float recurrence.

        For training use :meth:`step`, which gets the read-before-write ordering
        right. This helper is for inspecting a mirror, not for optimising one.
        """
        if x.ndim == 1:
            x = x[None, :]
        self.ensure_batch(int(x.shape[0]))
        state_int = self.state_int[: x.shape[0]].clone()
        s = state_int.to(torch.float32) / INT8_MAX
        out_f, _ = self.surrogate(x, bank_rates, s)
        out_int = self.forward_int(x, bank_rates)
        return out_int, out_f

    # -- convenience ---------------------------------------------------------

    def load_from_numpy(self, head: DeltaBankHead) -> None:
        """Copy reference weights in, e.g. after a checkpoint load."""
        with torch.no_grad():
            for name in ("W_k", "W_q", "W_v", "W_r", "bank_logits"):
                getattr(self, name).copy_(torch.tensor(np.array(getattr(head, name))))
            self.state_int.copy_(torch.tensor(np.array(head.state), dtype=torch.int32))

    def to_numpy_state(self) -> np.ndarray:
        return self.state_int.numpy().copy()

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, d_k={self.d_k}, d_v={self.d_v}, eta={self.eta}"
