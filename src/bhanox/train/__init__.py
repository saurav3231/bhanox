"""Training path for Bhanox. The only place torch is allowed (law C9).

Runtime stays numpy-only; this package is the optional GPU training extra, and
importing it is what pulls torch in. Nothing under ``src/bhanox`` outside this
directory may import torch, which is enforced by a test rather than by
convention.

Seven pieces, in the order they have to exist:

- :mod:`bhanox.train.gradcheck` -- finite-difference gradient checks. This is
  the thing that has to be right before any weight is trained, because a wrong
  gradient produces a model that looks like it is learning and is not.
- :mod:`bhanox.train.ste` -- the straight-through estimators. Two flavours:
  rounding boundaries (:func:`~bhanox.train.ste.ste_round`,
  :func:`~bhanox.train.ste.quantize_activation`) and decision boundaries
  (:func:`~bhanox.train.ste.ste_gt`, :func:`~bhanox.train.ste.ste_ge`).
- :mod:`bhanox.train.mirror` -- the torch mirror of the DeltaBank head, checked
  against the numpy reference with bit-exact integer agreement.
- :mod:`bhanox.train.mixer_mirror` -- the torch mirror of the MicroExpert mixer,
  checked against the numpy reference. Not bit-exact, because this layer stores
  dequantized weights and its reference forward is already float; the agreement
  claim is a stated float tolerance instead.
- :mod:`bhanox.train.bookend_mirror` -- HashBind, layer norm and the unembed.
  Layer norm in particular has no learned gain or bias, so it is written out
  rather than delegated to ``nn.LayerNorm``, whose default would add both.
- :mod:`bhanox.train.governor_mirror` -- the torch mirror of PulseGate. Bit-exact
  on the mask and on every piece of integer state, which is a stronger claim than
  the other mirrors make and is the only way to test a boolean state machine.
  It also documents a spec bug it mirrors rather than fixes: the gate's
  ``salience`` array is allocated, counted and checkpointed but never read, so it
  can never receive a gradient.
- :mod:`bhanox.train.model_mirror` -- the above, assembled in the reference's
  block order. This is where the agreement claims stop being component-local, and
  where the module docstring records the one thing worth knowing before trusting
  any of it: the reference's int8 state is *not* stable under ULP-level float
  perturbation, because PulseGate thresholds a float and has no dead band. Two
  float implementations therefore agree exactly for a while and then diverge, and
  that is a property of the frozen architecture rather than of any mirror.

The prototype trainer, added on top of those seven. It trains the mirror on
synthetic in-memory bytes and nothing else: no corpus, no checkpoint, no resume.

- :mod:`bhanox.train.objective` -- position-wise next-byte cross-entropy and
  the optimizer's parameter list, which excludes the gate's inert ``salience``.
- :mod:`bhanox.train.reset` -- the document-boundary reset: zero the integer
  state, flush the gates, keep everything that belongs to the run. Plus the
  shadow-detach that makes a per-chunk schedule possible.
- :mod:`bhanox.train.trainer` -- one optimizer step per chunk, and a document
  loop. Truncated backpropagation at each chunk seam, documented as such.
- :mod:`bhanox.train.smoke` -- the synthetic run. A repeating byte cycle, chosen
  because random bytes have no learnable structure and a falling loss on them
  would mean nothing. Reports its curve; asserts no threshold.

What this milestone deliberately does not have: real training, a corpus, a
learning-rate schedule, checkpointing, or resume. The mirror already provides
``state_dict()`` through ``nn.Module``, so a checkpoint milestone needs no new
mechanism -- only a decision about what may be claimed when resuming.


The mirror is not optional bookkeeping. If it drifts from the numpy reference
even slightly, training optimizes a different function than the one the
benchmarks measure, and every number in ``docs/benchmarks.md`` stops describing
the model that ships.
"""

from __future__ import annotations
