"""Training path for Bhanox. The only place torch is allowed (law C9).

Runtime stays numpy-only; this package is the optional GPU training extra, and
importing it is what pulls torch in. Nothing under ``src/bhanox`` outside this
directory may import torch, which is enforced by a test rather than by
convention.

Two pieces, in the order they have to exist:

- :mod:`bhanox.train.gradcheck` -- finite-difference gradient checks. This is
  the thing that has to be right before any weight is trained, because a wrong
  gradient produces a model that looks like it is learning and is not.
- :mod:`bhanox.train.mirror` -- the torch mirror of the forward pass, checked
  against the numpy reference. Currently
  :class:`~bhanox.core.deltabank.DeltaBankHead` only; the rest of the model, the
  gate, the router, and the training loop are not written yet.

The mirror is not optional bookkeeping. If it drifts from the numpy reference
even slightly, training optimizes a different function than the one the
benchmarks measure, and every number in ``docs/benchmarks.md`` stops describing
the model that ships.
"""

from __future__ import annotations
