"""The assembled torch mirror of :class:`bhanox.model.Bhanox`.

Purpose: wire the per-component mirrors into the same block order the
reference uses, so the thing that gets trained is the thing that gets
benchmarked.

In simple words: the other mirrors check that each part is right. This
one checks that the parts are right *in the order the reference puts
them*, which is the only place an ordering mistake can hide -- every
component test would still pass.

Split out of ``mirror.py`` for law C5. That file is the DeltaBank head
and is already at its line budget; the layer and model wiring is a
different concern (assembly, not mechanism) and lives here.

Two entry points, and the difference is the whole point:

- :meth:`BhanoxMirror.forward_int` runs the authoritative integer
  trajectory. It is what the agreement tests assert against.
- :meth:`BhanoxMirror.step` runs the training path: the same integer
  trajectory, plus surrogates that carry gradient, plus the router's
  load-balancing update.

The reference's forward is *always* the integer path -- it has no
surrogate at all. So "the mirror agrees with the reference" is a claim
about :meth:`forward_int` and nothing else. The float tolerance on the
training path is a separate, weaker claim, and it is stated separately
because it has to be: the surrogate deliberately rounds to int8 in the
middle, so it is a different function, not a worse approximation of the
same one.

What agreement actually holds, measured rather than assumed
-------------------------------------------------------

**The integer head state is bit-exact, for as long as the gate agrees.**
The DeltaBank read is float64 and the state write is int32 arithmetic, and
none of it depends on the order BLAS sums a float32 matmul in. So the
integer path is a legitimate ``array_equal`` comparison, and that is what
the tests use. The qualifier is not a hedge -- see below. It is bit-exact
until a gate mask differs, and then it is not, because the two runs stop
feeding the banks the same input. That is the whole subject of this
docstring's second half.

**The logits are not bit-exact, and cannot be.** torch's float32 GEMM
is not bit-identical to numpy's, because they are different BLAS builds.
Measured here at 1.1e-5 on a ``(2, 128) @ (128, 128)`` float32 product --
which is a property of this machine's BLAS, not a number to quote as a
spec, and it moves with library version and hardware. The part that does
not move is that it is non-zero at all. A few ULP per matmul, carried
forward by the residual stream.

**The PulseGate turns that ULP drift into a visible difference.** The
governor thresholds a float, ``delta = |a - cached|``, against
``tau_hi`` and ``tau_lo``. Nothing in between: a channel either
contributes its memory to the residual stream or it does not. So once
accumulated drift crosses a threshold, an entire channel switches on or
off, which is an O(1) change in the residual built on top of an O(1e-7)
cause.

This is a property of the architecture, not of this mirror. It was
confirmed with no mirror in the loop: perturbing the *reference's own*
activations by one to two ULPs between layers changes 9 of its 16 heads'
integer state, 1.3% of state elements, by up to 13.

``test_the_gate_has_no_dead_band`` is the direct evidence for the mechanism:
prime the cache so ``delta == tau_hi`` exactly, and one ULP upward wakes all
eight channels. There is no band to hide in, so no mirror can be held to a
tighter claim than this allows.

In the perturbation run, no single 1-ULP step actually crossed a threshold --
the closest any channel-step came to ``tau_hi`` was 3.4e-5, about 500 ULPs.
What escaped was the accumulation over the layer stack, which compounds with
depth and sequence length until it does cross. The two facts together are the
honest version: individual steps were nowhere near the boundary, and the
boundary still got crossed.

Consequences worth stating plainly rather than discovering later:

- Any two float implementations disagree this way. numpy against torch,
  CPU against GPU, two BLAS versions, two machines. Nothing in the mirror
  can prevent it, because the reference has no determinism margin built
  in.
- So a model trained in torch and benchmarked in numpy is the same model
  for the first several tokens and drifts after that. That is a real
  limit on the cross-implementation claim, and it is why the trainer has
  to be checked against the reference rather than assumed equal to it.

The measured envelope, from the trials behind ``tests/test_model_mirror.py``
(the test suite samples this envelope; it does not re-derive it):

- 2-layer config, 120 trials up to 64 tokens: integer state bit-exact
  119 times, worst logit difference 8.4e-6. The test holds this config to
  ``1e-5``.
- nano, 70 short-window trials up to ``(4, 4)``: worst logit difference
  1.25e-5, inside the documented ``1e-4``.
- Isolated nano seeds exist where an 8-token window reaches 9.8e-4 with the
  gate state itself diverged, so the ``1e-4`` claim at nano is a bound on a
  typical short window, not a guarantee.

The third row is why the claim is stated with a regime attached to it. It is
also why the tests run the converse direction -- a wrong state has to miss the
tolerance by a wide margin, or the tolerance is not evidence of anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from bhanox.core.deltabank_layer import DeltaBankLayer
from bhanox.model import Bhanox
from bhanox.train.bookend_mirror import HashBindMirror, UnembedMirror, layer_norm_torch
from bhanox.train.governor_mirror import PulseGateMirror
from bhanox.train.mirror import DeltaBankHeadMirror
from bhanox.train.mixer_mirror import MicroExpertLayerMirror
from bhanox.train.ste import quantize_activation

__all__ = ["BhanoxMirror", "DeltaBankLayerMirror"]


# ``nn.ModuleList`` is registered, so its members show up in ``parameters()`` and
# ``state_dict()``, which is what makes checkpointing and the optimizer work. It is
# not generic in torch's stubs and its ``__iter__`` is typed as yielding
# ``Tensor | Module``, so iterating one needs a narrowing cast. The objects are
# exactly the declared type; only the stub is imprecise, so the cast does no
# runtime work and loses no checking.


def _p(a: np.ndarray) -> nn.Parameter:
    return nn.Parameter(torch.tensor(np.array(a), dtype=torch.float32))


def _apply_mask(step_out: Tensor, compute: Tensor) -> Tensor:
    """``np.where(compute, step_out, 0.0)``, with the gradient kept.

    The reference's ``compute`` is a bool; the mirror's is a float that is
    *exactly* 0.0 or 1.0 at forward time, carrying a straight-through gradient.
    So the values can be multiplied -- but multiplying is the wrong operator for
    the forward pass, because ``inf * 0.0`` is ``nan`` and a single non-finite
    activation anywhere upstream would then poison the residual stream instead
    of being masked out of it. The reference's ``np.where`` puts an exact ``0.0``
    there, so the value comes from ``where`` and only the gradient is added on.

    The gradient term is ``step_out * (compute - compute.detach())``: value zero,
    derivative ``step_out`` with respect to ``compute``. Combined with the
    ``where`` above it gives the right total -- ``hard`` along the direct path
    to ``step_out``, and nothing extra, since the term's own derivative with
    respect to ``step_out`` is that same zero.
    """
    hard = compute > 0.5
    zeroed = torch.where(hard, step_out, torch.zeros_like(step_out))
    return zeroed + step_out * (compute - compute.detach())


class DeltaBankLayerMirror(nn.Module):
    """One DeltaBank *layer*: ``n_heads`` independent heads plus the shared wiring.

    ``DeltaBankHeadMirror`` mirrors a single head. The layer adds two things a
    head does not have: the read-out that stacks the per-head reads back to
    ``d_model``, and the gated bypass. Both are learned matrices, both get
    gradients, and the concatenation is load-bearing -- the heads' read-outs
    are interleaved by position, so getting the concat order wrong would produce
    a ``(B, n_heads * d_v) @ (n_heads * d_v, d_model)`` product of exactly the
    right shape and a completely different number.
    """

    # Class-level annotations for the registered buffers. ``register_buffer`` does
    # not narrow the type, and ``nn.Module.__getattr__`` is typed as returning
    # ``Tensor | Module``, so without these the buffer is a union and every call
    # through it is an error. Same pattern as the other mirrors.
    bank_rates: Tensor

    def __init__(self, layer: DeltaBankLayer) -> None:
        super().__init__()
        self.config = layer.config
        self.heads = nn.ModuleList(DeltaBankHeadMirror(h) for h in layer.heads)
        self.W_o = _p(layer.W_o)
        self.G = _p(layer.G)
        self.register_buffer(
            "bank_rates", torch.tensor(np.array(layer.bank_rates), dtype=torch.float32)
        )

    def _assemble(self, reads: Tensor, x: Tensor) -> Tensor:
        """``reads @ W_o + x @ G.T``. Mirrors ``DeltaBankLayer.forward``'s tail.

        The dtype dance is not decoration. The integer read is float64 (the
        reference divides an int64 by a python float, which promotes), while
        ``W_o`` is float32. numpy promotes the product to float64; torch raises
        ``expected m1 and m2 to have the same dtype``. So the promotion has to
        be written out.

        The bypass is left in float32 and only *then* widened, because that is
        the order numpy does it in -- ``arr @ G.T`` is a float32 product there,
        and only the final addition is float64. Widening the bypass first would
        be more accurate and would *disagree* with the reference, which is the
        wrong direction to be wrong in for a mirror.

        The result is deliberately left float64. The reference does not cast it
        either; the cast to float32 happens one level up, when the value is
        written into the preallocated float32 output array.
        """
        out = reads.to(torch.float64) @ self.W_o.to(torch.float64)
        return out + (x @ self.G.T).to(torch.float64)

    @torch.no_grad()
    def forward_int(self, x: Tensor) -> Tensor:
        """The integer path for a ``(B, d_model)`` input. Bit-exact, no gradient."""
        if x.ndim == 1:
            x = x[None, :]
            single = True
        else:
            single = False
        reads = torch.cat(
            [
                h.forward_int(x, self.bank_rates)
                for h in cast("list[DeltaBankHeadMirror]", list(self.heads))
            ],
            dim=-1,
        )
        out = self._assemble(reads, x)
        return out[0] if single else out

    def step(
        self,
        x: Tensor,
        shadows: list[Tensor] | None = None,
        *,
        quantize: Callable[[Tensor], Tensor] = quantize_activation,
        reanchor: bool = True,
        saturate: bool = True,
    ) -> tuple[Tensor, list[Tensor]]:
        """One token's training step. Returns ``(out, next_shadows)``."""
        if x.ndim == 1:
            x = x[None, :]
        reads: list[Tensor] = []
        next_shadows: list[Tensor] = []
        for j, h in enumerate(cast("list[DeltaBankHeadMirror]", list(self.heads))):
            prev = None if shadows is None else shadows[j]
            r, nxt = h.step(
                x,
                self.bank_rates,
                prev,
                quantize=quantize,
                reanchor=reanchor,
                saturate=saturate,
            )
            reads.append(r)
            next_shadows.append(nxt)
        return self._assemble(torch.cat(reads, dim=-1), x), next_shadows


class BhanoxMirror(nn.Module):
    """The full model, assembled in the reference's order.

    The block, verbatim from ``Bhanox._run_memory`` / ``Bhanox.step``::

        x = x + where(gate, bank(x), 0)     # the memory contribution
        x = x + mixer(x)
        x = layer_norm(x)

    The placement details are the ones a rewrite gets wrong, and all of them
    produce a forward that still runs:

    **The bank always computes; the gate only decides what to keep.** The
    reference runs ``bank.forward(x)`` for every channel at every step and then
    zeroes the *contribution* where the gate said skip. A mirror that skips the
    bank call when the gate says skip would stop the integer state from decaying
    on exactly the channels the governor is trying to preserve -- the opposite
    of what it is for -- and it would be invisible for as long as the gate never
    fully sleeps.

    **Where the mask is applied is not a choice, but not for the reason it looks
    like.** ``x + where(keep, s, 0)`` and ``where(keep, x + s, x)`` are the same
    value -- the obvious-looking bug here is not one, and a test that "proves" the
    first form is necessary is testing algebra rather than the mirror. The mask
    does have to land on the layer's ``(B, d_model)`` output rather than on the
    per-head read, but those widths coincide at nano, so a mask applied in the
    wrong place is a silent numerical error there and a shape error everywhere
    else.

    **The mixer sits between the residual and the layer norm.** It is token-local
    -- it mixes the feature axis only and never the time axis -- so whether it is
    written inside or outside the time loop computes the same function. What is
    load-bearing is the order around it: the memory contribution is added first,
    the mixer sees the sum, and normalisation is last. Moving the mixer before
    the residual, or the layer norm before it, is a different model.
    """

    def __init__(self, model: Bhanox) -> None:
        super().__init__()
        self.config = model.config
        self.embed = HashBindMirror(model.embedder)
        self.banks = nn.ModuleList(DeltaBankLayerMirror(b) for b in model.deltabanks)
        self.mixers = nn.ModuleList(MicroExpertLayerMirror(m) for m in model.mixers)
        self.gates = nn.ModuleList(PulseGateMirror(g) for g in model.gates)
        self.unembed = UnembedMirror(model.output)

    # -- the integer path, bit-exact -----------------------------------------

    @torch.no_grad()
    def forward_int(self, ids: np.ndarray) -> Tensor:
        """Next-token logits on the authoritative integer trajectory.

        Args:
            ids: ``(B, T)`` integer ids. A 1-D input is one sequence.

        Returns:
            ``(B, T, output_vocab)`` float32 logits.

        This is the reference's own function, in torch. In the regime the tests
        cover, the integer head state it produces is bit-exact and the tests assert
        that with ``array_equal``; the logits are compared with a float tolerance
        instead, for the reasons in the module docstring. Asserting the logits
        with ``array_equal`` would be a test that fails for a reason that has
        nothing to do with the mirror.

        "In the regime the tests cover" is doing real work in that sentence. A
        larger window, or a longer sequence, can reach a gate threshold where
        ``array_equal`` fails -- not because the mirror is wrong, but because the
        two float implementations have drifted far enough apart that the gate
        decides differently. ``test_the_gate_has_no_dead_band`` is the direct
        evidence, and the module docstring has the measurements.
        """
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.ensure_batch(int(arr.shape[0]))
        x = self.embed(arr)
        banks = cast("list[DeltaBankLayerMirror]", list(self.banks))
        mixers = cast("list[MicroExpertLayerMirror]", list(self.mixers))
        gates = cast("list[PulseGateMirror]", list(self.gates))
        for bank, mixer, gate in zip(banks, mixers, gates, strict=True):
            out = torch.zeros_like(x)
            for t in range(x.shape[1]):
                step_out = bank.forward_int(x[:, t])
                out[:, t] = _apply_mask(step_out, gate(step_out, step_out.abs()))
            x = x + out
            x = x + mixer(x)
            x = layer_norm_torch(x)
        return self.unembed(x)

    # -- the training path ---------------------------------------------------

    def ensure_batch(self, batch: int) -> None:
        """Grow every recurrent state to hold ``batch`` independent streams.

        Mirrors the reference's own ``ensure_batch``. New rows start at zero,
        which is exactly a fresh stream's state.
        """
        for bank in cast("list[DeltaBankLayerMirror]", list(self.banks)):
            for h in cast("list[DeltaBankHeadMirror]", list(bank.heads)):
                h.ensure_batch(batch)
        for gate in cast("list[PulseGateMirror]", list(self.gates)):
            gate.ensure_batch(batch)

    def step(
        self,
        ids: np.ndarray,
        *,
        shadows: list[list[Tensor]] | None = None,
        train: bool = False,
        quantize: Callable[[Tensor], Tensor] = quantize_activation,
        reanchor: bool = True,
        saturate: bool = True,
    ) -> tuple[Tensor, list[list[Tensor]]]:
        """One teacher-forced training forward over ``(B, T)`` ids.

        Args:
            ids: ``(B, T)`` integer ids.
            shadows: Per-layer, per-head float shadows from the previous call, or
                ``None`` to start fresh.
            train: Apply the router's load-balancing update.
            quantize: Activation quantiser, for the finite-difference
                configuration. Leave it alone for training.
            reanchor: See :meth:`DeltaBankHeadMirror.step`. ``False`` is for
                gradient checks only; it is not a training mode.
            saturate: Clip the shadow into the int8 range.

        Returns:
            ``(logits, next_shadows)``. Feed ``next_shadows`` back in on the
            next call.

        The integer state advances inside, one token at a time, because the
        recurrence is sequential in ``t`` and there is nothing to vectorise
        across it. The float shadows carry the graph between steps, which is
        what makes a 4,096-token window trainable without unrolling it by hand.
        """
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"step expects (B, T) ids, got shape {arr.shape}")
        # Checked before ``ensure_batch``, which is the whole point. Every other
        # guard here is a shape check, but this one also has to run before the
        # first state-mutating call: ``ensure_batch`` *grows* the recurrent
        # state, so a rejected forward would leave the mirror holding rows for a
        # window it refused to process. A rejection that mutates is not a
        # rejection. Wording mirrors ``Bhanox.forward`` so the two paths report
        # the same limit the same way.
        if arr.shape[-1] > self.config.max_context:
            raise ValueError(
                f"context {arr.shape[-1]} exceeds max_context="
                f"{self.config.max_context}. The state is O(1) in length, so "
                "this is a training-window limit, not a runtime one."
            )
        self.ensure_batch(int(arr.shape[0]))
        x = self.embed(arr)
        next_shadows: list[list[Tensor]] = []
        banks = cast("list[DeltaBankLayerMirror]", list(self.banks))
        mixers = cast("list[MicroExpertLayerMirror]", list(self.mixers))
        gates = cast("list[PulseGateMirror]", list(self.gates))
        for i, (bank, mixer, gate) in enumerate(zip(banks, mixers, gates, strict=True)):
            out = torch.zeros_like(x)
            layer_shadows = None if shadows is None else shadows[i]
            running: list[Tensor] | None = None
            for t in range(x.shape[1]):
                step_out, running = bank.step(
                    x[:, t],
                    running if running is not None else layer_shadows,
                    quantize=quantize,
                    reanchor=reanchor,
                    saturate=saturate,
                )
                out[:, t] = _apply_mask(step_out, gate(step_out, step_out.abs()))
            x = x + out
            x = x + mixer(x, train=train)
            x = layer_norm_torch(x)
            next_shadows.append(running if running is not None else [])
        return self.unembed(x), next_shadows

    # -- reporting -----------------------------------------------------------

    def param_count(self) -> int:
        """Stored parameter values, matching the reference's count exactly.

        Counts every ``nn.Parameter``, which *includes* the gate's ``salience``.
        ``salience`` is deliberately a parameter here even though it is inert: the
        reference exposes it as state that is allocated, counted and checkpointed,
        so mirroring it as a parameter is what makes this count and the checkpoint
        match. Inert is not the same as absent, and dropping it here would hide
        the real defect instead of reporting it.

        The trainer still has to exclude it from AdamW -- it never receives a
        gradient, and an optimiser group holding a permanently-``None`` gradient
        is a step that silently does nothing to that slice of the model.
        ``test_salience_is_a_parameter_and_inert`` pins both halves.
        """
        return int(sum(p.numel() for p in self.parameters()))

    def to_numpy_state(self) -> list[list[np.ndarray]]:
        """Every head's integer state, as numpy, for checkpointing."""
        return [
            [
                h.state_int.detach().cpu().numpy().copy()
                for h in cast("list[DeltaBankHeadMirror]", list(bank.heads))
            ]
            for bank in cast("list[DeltaBankLayerMirror]", list(self.banks))
        ]
