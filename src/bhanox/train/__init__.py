"""Training path for Bhanox. The only place torch is allowed (law C9).

Runtime stays numpy-only; this package is the optional GPU training extra, and
importing it is what pulls torch in. Nothing under ``src/bhanox`` outside this
directory may import torch, which is enforced by a test rather than by
convention.

Five pieces, in the order they have to exist:

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

Not written yet: the assembled full-model mirror and the training loop itself.


The mirror is not optional bookkeeping. If it drifts from the numpy reference
even slightly, training optimizes a different function than the one the
benchmarks measure, and every number in ``docs/benchmarks.md`` stops describing
the model that ships.
"""

from __future__ import annotations
