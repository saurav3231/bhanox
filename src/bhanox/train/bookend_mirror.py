"""Torch mirrors of the bookends: the HashBind front-end, layer norm, unembed.

Runtime stays numpy-only (law C9); this is the training path only, and like the
other mirrors it reproduces the reference exactly and adds a backward pass.

What "bookends" covers
----------------------
Everything from the ids to the logits that is not a repeated block:
:func:`~bhanox.frontend.hashbind.HashBind` on the way in,
:func:`~bhanox.model.layer_norm` between blocks, and the unembedding projection
on the way out. They look trivial next to DeltaBank and MicroExpert, and that is
exactly why they are worth mirroring first -- there is nothing here to hide a
mistake, so a mistake is a bug in the mirror rather than in the idea.

The hash is shared, not reimplemented
------------------------------------
:meth:`HashBindMirror.hash_rows` calls the reference's own ``hash_rows`` rather
than porting ``mix64`` to torch. That is a deliberate departure from "reproduce
the reference independently", and it is the right call for a specific reason: the
hash is pure integer index arithmetic with nothing to differentiate, so there is
no gradient to lose by sharing it, and a reimplementation would be pure downside
risk. ``mix64`` is xorshift-and-shift over uint64 with a per-function salt; two
implementations that agree on the reference test inputs can still disagree on
some other id, and the symptom would be a single token embedding that differs.
There is no aggregate to hide it in, but it would still be a silent divergence
in the one component whose entire job is to be a pure function of its input.

So the hash is a shared, non-differentiable subroutine, and everything that does
carry gradient -- the gather, the mix weights, the direct table, the norm and the
unembedding -- is mirrored here.

Layer norm has no affine parameters, and that is load-bearing
-------------------------------------------------------------
The reference's :func:`~bhanox.model.layer_norm` is a function: mean, centre,
divide by the RMS, with **no** learned gain or bias. ``torch.nn.LayerNorm``
defaults to ``elementwise_affine=True``, so wrapping the reference in it would
silently add ``2 * d_model`` parameters per call that the reference does not
have, change the parameter count, and give a model a degree of freedom the
architecture does not grant. So the norm is written out in full here rather than
delegated, and :func:`layer_norm_torch` is a function for the same reason
:class:`~bhanox.model.Bhanox` exposes ``layer_norm`` as one.

The all-zero-row guard
----------------------
``eps`` is not cosmetic. An all-zero row has zero variance, so ``0 / sqrt(0)`` is
NaN, and NaN in the residual stream is unrecoverable for the rest of the
sequence. The reference floors the variance; this does the same, and the test
asserts the output is zeros rather than merely finite.

The direct table is an addition, not a choice
---------------------------------------------
The frozen D3 equation is ``e(x) = Table[x] + sum_j g_j * Pool[h_j(x)]``. A known
id gets its dedicated row *in addition to* the hashed contribution. Substituting
the table for the hash would be a different, cheaper model, and it would look
like an optimisation rather than a bug.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from bhanox.frontend.hashbind import HashBind

__all__ = [
    "HashBindMirror",
    "UnembedMirror",
    "layer_norm_torch",
]

#: Variance floor, mirroring the reference's ``layer_norm`` default.
LAYER_NORM_EPS = 1e-5


def layer_norm_torch(x: Tensor, eps: float = LAYER_NORM_EPS) -> Tensor:
    """Unit-RMS normalisation, matching :func:`bhanox.model.layer_norm`.

    Args:
        x: ``(..., d)`` activations.
        eps: Variance floor, so an all-zero row returns zeros rather than NaN.

    Returns:
        Normalised activations of the same shape and dtype as ``x``.

    Why this is spelled out rather than delegated to ``F.layer_norm``: the
    reference defines the operation and I4 requires the mirror to agree with it
    to 1e-3, so the arithmetic is reproduced in the order the reference uses.
    ``F.layer_norm`` would very likely agree numerically, and it would also
    quietly accept a weight and a bias that the architecture does not have.

    The variance is the *biased* one, over the same axis, which is what
    ``F.layer_norm`` computes too -- the unbiased correction is a ``LayerNorm``
    with a different meaning entirely, and a plausible-looking thing to get
    wrong.
    """
    mu = x.mean(dim=-1, keepdim=True)
    centred = x - mu
    var = (centred * centred).mean(dim=-1, keepdim=True)
    return centred / torch.sqrt(var + eps)


class UnembedMirror(nn.Module):
    """Torch mirror of the unembedding projection ``x @ output``.

    Args:
        output: The reference's ``(d_model, output_vocab)`` array, as produced by
            ``bhanox.model._init_output`` or restored from a checkpoint.
    """

    output: nn.Parameter

    def __init__(self, output: np.ndarray) -> None:
        super().__init__()
        arr = np.asarray(output)
        self.d_model = int(arr.shape[0])
        self.output_vocab = int(arr.shape[1])
        self.output = nn.Parameter(torch.tensor(np.array(arr), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Project activations to logits.

        Args:
            x: ``(..., d_model)`` activations, normally the final normalised
                residual stream.

        Returns:
            ``(..., output_vocab)`` logits.

        No bias: the reference is a bare matmul, ``(x @ self.output)``. Adding one
        would be a free parameter the architecture does not have, and at
        ``output_vocab=256`` it would be 256 values per model that nothing else
        in the parameter accounting would notice.
        """
        return x @ self.output

    def load_from_numpy(self, output: np.ndarray) -> None:
        """Copy reference weights in, e.g. after a checkpoint load."""
        with torch.no_grad():
            self.output.copy_(torch.tensor(np.array(output), dtype=torch.float32))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, output_vocab={self.output_vocab}, bias=False"


class HashBindMirror(nn.Module):
    """Torch mirror of :class:`~bhanox.frontend.hashbind.HashBind`.

    Args:
        gate: The numpy reference front-end. Arrays are copied, not shared, so
            training the mirror cannot mutate the reference.

    Note on the gather: ``pool`` is looked up with an index tensor, so its
    backward is a scatter-add over the rows that were actually touched. That is
    the correct and cheap behaviour here -- an embedding lookup updates the rows
    it read -- and it is worth knowing when reading the gradient tests: most of
    the 8,192 pool rows get exactly zero gradient in any one step, and a test
    that asserts "the pool has a gradient" has to check the norm over the whole
    tensor rather than elementwise, or it will be asserting that every row is
    used, which is not true and is not supposed to be.
    """

    pool: nn.Parameter
    table: nn.Parameter
    g: nn.Parameter

    def __init__(self, gate: HashBind) -> None:
        super().__init__()
        self.gate = gate
        self.d_model = int(gate.d_model)
        self.pool_size = int(gate.pool_size)
        self.vocab_table = int(gate.vocab_table)
        self.n_hashes = int(gate.n_hashes)

        def _p(a: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.tensor(np.array(a), dtype=torch.float32))

        self.pool = _p(gate.pool)
        self.table = _p(gate.table)
        self.g = _p(gate.g)

    def hash_rows(self, ids: np.ndarray | Tensor) -> np.ndarray:
        """Pool rows per id, one column per hash function.

        Delegates to the reference. See the module docstring: the hash is integer
        index arithmetic with no gradient, so sharing it costs nothing and
        removes a whole class of silent divergence.
        """
        raw = ids.detach().cpu().numpy() if isinstance(ids, Tensor) else np.asarray(ids)
        return self.gate.hash_rows(raw)

    def forward(self, ids: Tensor) -> Tensor:
        """Embed ids, matching :meth:`HashBind.embed`.

        Args:
            ids: Integer ids, last axis the token axis. ``(B, T)`` gives
                ``(B, T, d_model)``.

        Returns:
            ``float32`` embeddings of shape ``ids.shape + (d_model,)``.
        """
        arr = torch.as_tensor(ids).to(torch.int64)
        if arr.ndim == 0:
            raise ValueError("ids must have at least one axis (the token axis)")

        rows = torch.from_numpy(self.hash_rows(arr))
        if rows.shape[-1] != self.n_hashes:
            raise ValueError(
                f"expected {self.n_hashes} hash columns, got {rows.shape[-1]}"
            )
        # ``(..., n_hashes, d_model)`` -- one gathered row per hash.
        gathered = self.pool[rows]
        hashed = torch.einsum("...hk,h->...k", gathered, self.g)

        known = (arr >= 0) & (arr < self.vocab_table)
        # The index is clamped *and* the result is re-masked. Clamping alone is
        # not enough: a negative id would index the table from the end, so id -1
        # would silently read the last row, and the mask would then add it. Both
        # halves are reproduced from the reference, which has the same two
        # defences for the same reason.
        direct = self.table[torch.where(known, arr, torch.zeros_like(arr))]
        return hashed + torch.where(
            known[..., None], direct, torch.zeros((), dtype=hashed.dtype)
        )

    def embed(self, ids: Tensor) -> Tensor:
        """Alias for :meth:`forward`, so the module matches the reference's API."""
        return self.forward(ids)

    def nbytes(self) -> int:
        """int8 bytes resident, mirroring ``HashBind.nbytes``.

        A method here rather than the reference's ``@property`` because a
        property and an ``nn.Module`` attribute collide: reading ``self.nbytes``
        would go through ``nn.Module.__getattr__`` machinery and a property on
        the class is not a buffer or a parameter. Renaming the shape is the
        smaller surprise than a mirror that raises on attribute access.
        """
        return (self.pool.numel() + self.table.numel()) * 1

    def param_count(self) -> int:
        """Total stored values, mirroring ``HashBind.param_count``."""
        return int(self.pool.numel() + self.table.numel() + self.g.numel())

    def load_from_numpy(self, gate: HashBind) -> None:
        """Copy reference arrays in, e.g. after a checkpoint load."""
        with torch.no_grad():
            self.pool.copy_(torch.tensor(np.array(gate.pool), dtype=torch.float32))
            self.table.copy_(torch.tensor(np.array(gate.table), dtype=torch.float32))
            self.g.copy_(torch.tensor(np.array(gate.g), dtype=torch.float32))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, pool_size={self.pool_size}, "
            f"vocab_table={self.vocab_table}, n_hashes={self.n_hashes}"
        )
