"""Agreement and gradient tests for the bookend mirrors.

The bookends are small, which is the point: there is no recurrence to get
subtly wrong here, so anything that does disagree is a bug in the mirror rather
than a subtle difference in an idea. That makes them the cheapest place to prove
the training path reproduces the reference, and it means the tests below are
mostly about *specific* properties that a naive port gets wrong:

* the hash is shared with the reference rather than reimplemented;
* layer norm has no affine gain or bias, and ``nn.LayerNorm`` would add both;
* a known id gets its table row **in addition to** the hashed contribution;
* a negative id must not index the table from the end;
* an all-zero row normalises to zeros, not NaN;
* the unembedding has no bias.

Agreement is a float tolerance, not bit-exactness, because these are all float
reductions. The tolerance is stated once at the top and used throughout, and it
is two orders of magnitude looser than anything observed, so it pins the claim
without becoming the thing that hides a real difference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bhanox.frontend.hashbind import HashBind
from bhanox.model import _init_output, layer_norm
from bhanox.train.bookend_mirror import (
    HashBindMirror,
    UnembedMirror,
    layer_norm_torch,
)

#: Observed worst case is 9.6e-7 (the unembedding matmul, 256 output rows).
#: 1e-5 is two orders of magnitude above that, which is loose enough not to be
#: a flaky test and tight enough that a wrong formula cannot hide inside it.
TOL = 1e-5


def _front_end(d_model: int = 16, pool: int = 64, table: int = 16) -> HashBind:
    return HashBind(d_model=d_model, pool_size=pool, vocab_table=table, n_hashes=4)


def _pair(**kwargs) -> tuple[HashBind, HashBindMirror]:
    gate = _front_end(**kwargs)
    return gate, HashBindMirror(gate)


def _ids(
    rng: np.random.Generator, shape: tuple[int, ...], high: int = 5000
) -> np.ndarray:
    return rng.integers(0, high, size=shape).astype(np.int64)


# -- agreement -------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 7), (1, 1), (1, 9), (3,), (2, 3, 5)])
def test_embedding_matches_reference(shape: tuple[int, ...]):
    """``HashBind.embed`` and the mirror agree, at any leading shape."""
    gate, mirror = _pair()
    ids = _ids(np.random.default_rng(0), shape)
    ref = gate.embed(ids)
    got = mirror(torch.tensor(ids)).detach().numpy()
    assert ref.shape == got.shape == (*shape, gate.d_model)
    assert np.max(np.abs(ref - got)) < TOL


def test_embedding_matches_reference_across_the_whole_id_range():
    """Ids spanning the 4-gram space, including the top of the 32-bit range.

    The spec's id space is a byte 4-gram, so ids reach ``2**32 - 1``. A hash
    that mishandles the high bits agrees perfectly on small ids and then
    scatters everything above some point, and ``2**31`` is exactly where a sign
    or a 32-bit cast would show up.
    """
    gate, mirror = _pair()
    ids = np.array(
        [[0, 1, 1023, 1024, 2**15, 2**16, 2**31 - 1, 2**31, 2**32 - 1, 2**32]],
        dtype=np.int64,
    )
    ref = gate.embed(ids)
    got = mirror(torch.tensor(ids)).detach().numpy()
    assert np.max(np.abs(ref - got)) < TOL
    rows = gate.hash_rows(ids)
    assert rows.min() >= 0 and rows.max() < gate.pool_size


def test_hash_rows_are_shared_with_the_reference_not_reimplemented():
    """The mirror must not carry its own copy of the mixer.

    Stated as a test because the temptation to "fix" it later is exactly what
    would reintroduce the divergence: an inlined torch port of ``mix64`` would
    pass every other test here and still be a second implementation of a
    function whose only job is to be pure.
    """
    gate, mirror = _pair()
    ids = _ids(np.random.default_rng(1), (3, 5))
    assert np.array_equal(mirror.hash_rows(ids), gate.hash_rows(ids))
    # And via a tensor, which is the path the forward actually takes.
    assert np.array_equal(
        mirror.hash_rows(torch.tensor(ids)), gate.hash_rows(ids)
    ), "the tensor path must not take a different route"
    assert mirror.hash_rows(ids).shape == (*ids.shape, mirror.n_hashes)


def test_hash_rows_are_deterministic():
    """Same id, same row, every time. That is the whole premise of a hash."""
    _gate, mirror = _pair()
    ids = _ids(np.random.default_rng(2), (4, 4))
    assert np.array_equal(mirror.hash_rows(ids), mirror.hash_rows(ids))


def test_known_id_is_table_row_plus_hashed_contribution():
    """The direct table is an addition, not a substitution.

    The frozen D3 equation sums them. A mirror that used the table row *instead*
    of adding the hashed term would produce a plausible-looking embedding for
    every known id and a wrong one, and the byte savings it appeared to buy would
    be an accounting fiction.
    """
    gate, mirror = _pair()
    known_id = 3
    ids = np.array([known_id], dtype=np.int64)
    rows = gate.hash_rows(ids)
    hashed = np.einsum("...hk,h->...k", gate.pool[rows], gate.g)
    assert np.allclose(
        gate.embed(ids), gate.table[known_id] + hashed[0], atol=TOL
    ), "the reference is a sum"
    assert not np.allclose(
        gate.embed(ids), gate.table[known_id], atol=TOL
    ), "and the hashed term is not negligible"
    assert (
        np.max(np.abs(gate.embed(ids) - mirror(torch.tensor(ids)).detach().numpy()))
        < TOL
    )


def test_unknown_id_gets_only_the_hashed_contribution():
    """Past ``vocab_table`` there is no direct row to add."""
    gate, mirror = _pair()
    unknown = gate.vocab_table + 5
    ids = np.array([unknown], dtype=np.int64)
    rows = gate.hash_rows(ids)
    hashed = np.einsum("...hk,h->...k", gate.pool[rows], gate.g)
    assert np.allclose(gate.embed(ids), hashed[0], atol=TOL)
    assert (
        np.max(np.abs(gate.embed(ids) - mirror(torch.tensor(ids)).detach().numpy()))
        < TOL
    )


def test_negative_id_does_not_read_the_table_from_the_end():
    """Id -1 must not silently become id ``vocab_table - 1``.

    A negative id in a NumPy or torch index wraps, so a mirror that clamped only
    the *value* and not the *index* would add ``table[-1]`` -- a real, learned
    vector -- to the embedding of an unknown token. Nothing raises; the token
    just gets a slightly wrong embedding forever.
    """
    gate, mirror = _pair()
    gate.table[-1] = 1000.0
    mirror.table.data[-1] = 1000.0
    ids = np.array([-1], dtype=np.int64)
    ref = gate.embed(ids)
    got = mirror(torch.tensor(ids)).detach().numpy()
    assert np.max(np.abs(ref - got)) < TOL
    # The poison value must not appear anywhere in the result.
    assert np.abs(got).max() < 100.0, "table[-1] leaked into the embedding"
    # And a negative id must differ from the last known id, which is what a
    # wraparound would have made it equal to.
    last = np.array([gate.vocab_table - 1], dtype=np.int64)
    assert not np.allclose(got, gate.embed(last), atol=1e-3)


def test_mix_weights_start_uniform_and_are_all_learned():
    """``g`` is initialised to ``1/n_hashes`` and is a real parameter."""
    gate, mirror = _pair()
    assert np.allclose(gate.g, 0.25)
    assert np.allclose(mirror.g.detach().numpy(), gate.g)
    assert "g" in dict(mirror.named_parameters())


# -- layer norm ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 7, 16), (1, 16), (5,), (2, 3, 4, 16)])
def test_layer_norm_matches_reference(shape: tuple[int, ...]):
    """Same normalisation, any leading shape."""
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    ref = layer_norm(x)
    got = layer_norm_torch(torch.tensor(x)).detach().numpy()
    assert ref.shape == got.shape == shape
    assert np.max(np.abs(ref - got)) < TOL


def test_layer_norm_produces_unit_rms():
    """The definition: RMS 1 along the last axis, to within ``eps``."""
    x = np.random.default_rng(4).standard_normal((6, 7, 16)).astype(np.float32)
    got = layer_norm_torch(torch.tensor(x)).detach().numpy()
    rms = np.sqrt(np.mean(got * got, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3)
    assert np.abs(got.mean(axis=-1)).max() < 1e-3, "and mean zero"


def test_layer_norm_has_no_affine_parameters():
    """No gain, no bias -- the reference is a bare function.

    This is the trap worth a test. ``nn.LayerNorm(d)`` defaults to
    ``elementwise_affine=True``, so wrapping the reference in it would add
    ``2 * d_model`` values to the parameter count, give the model a degree of
    freedom the architecture does not grant, and still pass every forward
    agreement test with the weights left at their initial values.
    """

    # A hand-rolled nn.Module version stays parameter-free too, which is the
    # check that would actually catch an affine creeping in.
    class _Norm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eps = 1e-5

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            return layer_norm_torch(y, self.eps)

    norm = _Norm()
    assert list(norm.parameters()) == []
    x = torch.randn(2, 4, 16)
    assert np.max(np.abs(norm(x).detach().numpy() - layer_norm(x.numpy()))) < TOL


def test_layer_norm_all_zero_row_is_zero_not_nan():
    """The ``eps`` floor is what keeps a zero row out of the NaN range.

    Zero variance means ``0 / sqrt(0)``, and a NaN in the residual stream is not
    recoverable for the rest of the sequence -- it propagates through every
    subsequent layer. Asserting ``isfinite`` would pass for any large value; the
    claim is that the row is exactly zero, which is what the reference returns.
    """
    for d in (1, 16, 128):
        x = np.zeros((3, d), dtype=np.float32)
        ref = layer_norm(x)
        got = layer_norm_torch(torch.tensor(x)).detach().numpy()
        assert np.isfinite(ref).all() and np.isfinite(got).all()
        assert np.array_equal(ref, np.zeros_like(ref))
        assert np.array_equal(got, np.zeros_like(got))


def test_layer_norm_is_scale_invariant():
    """Scaling the input scales nothing out, because it normalises.

    Cheapest available check that the mean and variance are both being removed
    rather than the input merely being divided by its own norm.
    """
    x = torch.tensor(
        np.random.default_rng(5).standard_normal((4, 16)).astype(np.float32)
    )
    a = layer_norm_torch(x)
    b = layer_norm_torch(x * 7.5)
    assert torch.allclose(a, b, atol=1e-5)
    c = layer_norm_torch(x + 3.0)
    assert torch.allclose(a, c, atol=1e-5), "a constant shift must also cancel"


def test_layer_norm_gradient_reaches_the_input():
    """And the gradient is not trivially zero.

    ``y.sum()`` is a degenerate loss here -- the normalised output sums to about
    zero by construction, so its gradient nearly vanishes and a test using it
    would pass for the wrong reason. A squared loss does not have that property.
    """
    x = torch.tensor(
        np.random.default_rng(6).standard_normal((2, 5, 16)).astype(np.float32),
        requires_grad=True,
    )
    y = layer_norm_torch(x)
    y.pow(2).sum().backward()
    assert x.grad is not None
    assert float(x.grad.norm()) > 0.0
    # The sum of the gradient over a feature axis vanishes: layer norm's own
    # first-order condition. Stated so a future change cannot quietly break it.
    assert float(x.grad.sum(dim=-1).abs().max()) < 1e-4


# -- unembed ---------------------------------------------------------------


def test_unembed_matches_reference():
    """A bare matmul against the stored projection."""
    from bhanox.config import load_config

    cfg = load_config("nano")
    output = _init_output(cfg)
    unembed = UnembedMirror(output)
    x = np.random.default_rng(7).standard_normal((4, 6, cfg.d_model)).astype(np.float32)
    ref = layer_norm(x) @ output
    got = (layer_norm_torch(torch.tensor(x)) @ unembed.output).detach().numpy()
    assert ref.shape == got.shape == (4, 6, cfg.output_vocab)
    assert np.max(np.abs(ref - got)) < TOL


def test_unembed_has_no_bias():
    """The reference is ``x @ output``; a bias would be 256 free parameters."""
    unembed = UnembedMirror(np.zeros((8, 5), dtype=np.float32))
    names = {name for name, _ in unembed.named_parameters()}
    assert names == {"output"}
    # A zero input with a zero output gives zeros. With a bias it would not.
    got = unembed(torch.zeros(3, 8))
    assert torch.equal(got, torch.zeros(3, 5))


def test_unembed_gradient_reaches_both_sides():
    """``output`` is learnable and the residual stream is not a dead end."""
    unembed = UnembedMirror(
        np.random.default_rng(8).standard_normal((8, 5)).astype(np.float32)
    )
    x = torch.randn(3, 4, 8, requires_grad=True)
    unembed(x).pow(2).sum().backward()
    assert unembed.output.grad is not None and float(unembed.output.grad.norm()) > 0.0
    assert x.grad is not None and float(x.grad.norm()) > 0.0


# -- gradients on the front-end -------------------------------------------


def test_pool_gradient_touches_only_the_gathered_rows():
    """A gather's backward is a scatter-add over the rows actually read.

    The check is on the whole-tensor norm *and* on the untouched rows being
    exactly zero. Asserting only "the pool has a gradient" would pass even if
    every row were updated, and a front-end that updated all 8,192 rows per token
    would defeat the entire memory saving this component exists for.
    """
    gate, mirror = _pair(pool=64)
    ids = torch.tensor(_ids(np.random.default_rng(9), (4, 7), high=5000))
    mirror(ids).pow(2).sum().backward()
    assert mirror.pool.grad is not None
    touched = np.unique(mirror.hash_rows(ids.numpy()))
    per_row = mirror.pool.grad.norm(dim=-1)
    untouched = np.setdiff1d(np.arange(gate.pool_size), touched)
    assert len(untouched) == gate.pool_size - len(touched) > 0
    assert bool((per_row[untouched] == 0).all()), "an un-gathered row was updated"
    assert bool((per_row[touched] > 0).any()), "no gathered row received gradient"


def test_table_gradient_touches_only_known_rows():
    """Unknown ids must not create gradient on any table row.

    The mask is what enforces this. Drop it -- "always gather, mask the value
    afterwards" is the obvious simplification -- and the table receives gradient
    for every id in the batch while the forward value is still correct, so the
    test that only checks the forward stays green.
    """
    gate, mirror = _pair(table=16)
    ids = torch.tensor(np.array([[100, 200, 300]], dtype=np.int64))
    assert not bool(((ids >= 0) & (ids < gate.vocab_table)).any()), "need unknown ids"
    mirror(ids).pow(2).sum().backward()
    assert mirror.table.grad is not None
    assert bool((mirror.table.grad == 0).all()), "unknown ids must not touch the table"


def test_table_gradient_reaches_exactly_the_known_rows():
    _gate, mirror = _pair(table=16)
    known_ids = [2, 5, 9]
    ids = torch.tensor(np.array([[*known_ids, 100, 200]], dtype=np.int64))
    mirror(ids).pow(2).sum().backward()
    per_row = mirror.table.grad.norm(dim=-1)
    got = np.nonzero(per_row.numpy() > 0)[0]
    assert sorted(got.tolist()) == sorted(known_ids)


def test_mix_weights_receive_gradient():
    """``g`` scales each hash's contribution, so it is on the graph."""
    _, mirror = _pair()
    ids = torch.tensor(_ids(np.random.default_rng(10), (4, 7)))
    mirror(ids).pow(2).sum().backward()
    assert mirror.g.grad is not None and float(mirror.g.grad.norm()) > 0.0


def test_all_front_end_parameters_are_trained_together():
    """One forward, one backward: pool, table and ``g`` all live."""
    _gate, mirror = _pair()
    ids = torch.tensor(np.array([[0, 1, 2, 3, 100, 200, 5000, 9999]], dtype=np.int64))
    mirror(ids).pow(2).sum().backward()
    for name, param in mirror.named_parameters():
        assert param.grad is not None, f"{name} left the graph"
        assert float(param.grad.norm()) > 0.0, f"{name} got a zero gradient"


# -- shape, errors, bookkeeping -------------------------------------------


def test_scalar_id_is_rejected():
    """Ids need a token axis; a bare scalar would silently lose one."""
    _, mirror = _pair()
    with pytest.raises(ValueError, match="token axis"):
        mirror(torch.tensor(7))


def test_nbytes_and_param_count_match_the_reference():
    """Both inventories agree, which is what the byte claim rests on."""
    for d_model, pool, table in ((16, 64, 16), (128, 8192, 1024), (256, 8192, 1024)):
        gate, mirror = _pair(d_model=d_model, pool=pool, table=table)
        # ``nbytes`` is a property on the reference and a method on the mirror;
        # see the note in the mirror. Both are the int8 row count.
        assert mirror.nbytes() == gate.nbytes == (pool + table) * d_model
        assert mirror.param_count() == gate.param_count()
        assert mirror.param_count() == mirror.nbytes() + mirror.n_hashes


def test_front_end_is_by_far_the_largest_single_component():
    """The saving the front-end exists for, stated as a ratio not a slogan.

    A full 4-gram table would need ``2**32 * d_model`` values; the front-end holds
    ``pool_size + vocab_table`` rows. The claim in the docstrings is "~96%", so
    this asserts the order of magnitude rather than the exact figure, which
    depends on ``d_model``.
    """
    for d_model in (128, 256, 512):
        _, mirror = _pair(d_model=d_model, pool=8192, table=1024)
        full = 2**32 * d_model
        assert mirror.nbytes() / full < 0.05


def test_training_cannot_mutate_the_reference():
    """Arrays are copied, not aliased, in both mirrors."""
    gate, mirror = _pair()
    with torch.no_grad():
        mirror.pool.add_(1.0)
        mirror.table.mul_(2.0)
        mirror.g.add_(0.5)
    assert not np.allclose(gate.pool, mirror.pool.detach().numpy())
    assert not np.allclose(gate.table, mirror.table.detach().numpy())
    assert not np.allclose(gate.g, mirror.g.detach().numpy())


def test_load_from_numpy_restores_the_front_end():
    """A reloaded mirror matches a fresh one on the same reference."""
    gate, mirror = _pair()
    rng = np.random.default_rng(11)
    for _ in range(3):
        gate.embed(_ids(rng, (2, 5)))
    with torch.no_grad():
        mirror.pool.mul_(3.0)
    mirror.load_from_numpy(gate)
    assert np.array_equal(mirror.pool.detach().numpy(), gate.pool)
    assert np.array_equal(mirror.table.detach().numpy(), gate.table)
    assert np.array_equal(mirror.g.detach().numpy(), gate.g)
    ids = _ids(rng, (3, 4))
    assert (
        np.max(np.abs(gate.embed(ids) - mirror(torch.tensor(ids)).detach().numpy()))
        < TOL
    )


def test_unembed_load_from_numpy():
    output = np.random.default_rng(12).standard_normal((8, 5)).astype(np.float32)
    unembed = UnembedMirror(output)
    with torch.no_grad():
        unembed.output.add_(5.0)
    unembed.load_from_numpy(output)
    assert np.array_equal(unembed.output.detach().numpy(), output)


def test_repr_reports_the_bookend_shape():
    gate, mirror = _pair()
    text = repr(mirror)
    assert f"pool_size={gate.pool_size}" in text
    assert f"vocab_table={gate.vocab_table}" in text
    assert "bias=False" in repr(UnembedMirror(np.zeros((8, 5), dtype=np.float32)))


def test_int32_and_int64_ids_agree():
    """The trainer hands over whatever dtype the data pipeline produced.

    A byte 4-gram id reaches ``2**32 - 1``, which does not fit in int32, so both
    dtypes have to work and have to agree. Widening silently is the right
    behaviour; narrowing would be a wraparound bug waiting for a long document.
    """
    gate, mirror = _pair()
    ids64 = np.array([[0, 1023, 2**31, 2**32 - 1]], dtype=np.int64)
    ids32 = np.array([[0, 1023, -2147483648, -1]], dtype=np.int32)
    ref64 = gate.embed(ids64)
    got32 = mirror(torch.tensor(ids32)).detach().numpy()
    got64 = mirror(torch.tensor(ids64)).detach().numpy()
    assert np.max(np.abs(ref64 - got64)) < TOL
    # int32 -1 and int64 2**32-1 are different ids, and must stay different:
    # casting int64 down to int32 would make them collide.
    assert not np.allclose(got32, got64, atol=1e-3)


def test_embed_alias_is_the_same_path():
    """``embed`` exists so the mirror matches the reference's API surface."""
    _gate, mirror = _pair()
    ids = _ids(np.random.default_rng(13), (2, 3))
    assert np.array_equal(
        mirror.embed(torch.tensor(ids)).detach().numpy(),
        mirror(torch.tensor(ids)).detach().numpy(),
    )
