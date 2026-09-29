"""DeltaBank: the recurrent memory at the centre of Bhanox.

Purpose: replace attention's growing KV cache with a fixed-size state that is
*corrected* on write instead of appended to, and read in constant time.

In simple words: a notebook with a fixed number of pages. Writing a fact
doesn't add a page — it erases the wrong answer already on that page and writes
the right one. Each page forgets on its own schedule, and there are pages that
forget fast (details) and pages that forget slowly (gist).

Module layout (law C5, one concern per module). This module is the head: the
recurrence and its projections. The two related concerns live next door:

- :mod:`bhanox.core.deltabank_numerics` -- the int8 grid the rule runs on
  (``l2_normalize``, ``recip_lut``, the shared ``_RECIP`` table, the logistic).
  :mod:`bhanox.train.mirror` needs the same table and the same normalisation as
  the reference head, or the two disagree about what an int8 code means.
- :mod:`bhanox.core.deltabank_layer` -- the multi-head wiring for one layer.

The numerics are re-exported here, so ``from bhanox.core.deltabank import
l2_normalize`` keeps working exactly as before.

Architecture (spec D3, frozen)::

    k_t, v_t, q_t   = projections of x_t
    k_t <- k_t / ||k_t||_2      and   q_t <- q_t / ||q_t||_2   # CRITICAL
    lambda_t       = per-channel decay, a learned mix over B decay banks
    r_t            = S_{t-1}^T q_t                  # READ  (before write!)
    e_t            = v_t - S_{t-1}^T k_t            # surprise
    beta_t         = LUT_recip[||k_t||^2] * eta_head  ~= eta_head
    S_t            = Diag(lambda_t) S_{t-1} + beta_t e_t k_t^T
    y_t            = W_o (r_t * rho_t) + G * x_t
    rho_t          = sigmoid(W_r x_t)                # read gate

Three findings from the design-phase tournament are load-bearing and are
regression-tested in ``tests/core/test_deltabank.py``:

1. **K and Q L2 normalization is mandatory.** Without it ``beta`` explodes
   (~31), the delta rule's contraction condition fails, and training cannot
   start at all. This is not a tuning knob.
2. **Read before write.** Read-after-write lets the fresh write overwrite the
   very association being retrieved: 3.1623 vs 3.1251 BPC, and it destroys
   memory accuracy (0.981 is the best achievable).
3. **The read gate is not optional.** +0.037 BPC without it.

Bytes touched per token: state is read once and written once, so
``2 * d_k * d_v`` bytes of int8 state, plus ``3 * d_model * (d_k + d_v)`` of
projection weights. Constant in context length -- this is the O(1) guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bhanox.core.deltabank_numerics import (
    _RECIP,
    _sigmoid,
    l2_normalize,
    recip_lut,
)
from bhanox.quant.numerics import (
    INT8_MAX,
    absmax_quantize,
    qmul,
    quantize_activation,
    saturate_int8,
    to_q,
)
from bhanox.seeding import init_rng

__all__ = ["DeltaBankHead", "l2_normalize", "recip_lut"]


@dataclass
class DeltaBankHead:
    """One independent memory: a ``d_k x d_v`` state with its own projections.

    Attributes:
        d_k: Key/query width, and the number of state rows.
        d_v: Value width, and the number of state columns.
        W_k: Input key projection, shape ``(d_in, d_k)``.
        W_q: Input query projection, shape ``(d_in, d_k)``.
        W_v: Input value projection, shape ``(d_in, d_v)``.
        W_r: Read gate, shape ``(d_in, d_v)``.
        bank_logits: Learned log-space weights over the ``B`` decay banks,
            shape ``(n_banks, d_k)``.
        eta: Per-head step size, scalar. Kept below 1 so the update is a
            contraction (a projection, not an amplifier).
        state: The recurrent state, shape ``(d_k, d_v)``. Held as int32 in
            this reference purely as the arithmetic carrier for the
            rank-1 update; the state is logically int8 and every value is an
            int8 code. See :attr:`state_nbytes` for the honest accounting.
        reads: Lifetime counter of read operations, for tests and the audit.
        writes: Lifetime counter of write operations.
        normalize_keys: Whether to L2-normalise K and Q. Mandatory: without it
            the delta rule's ``beta = 1/||k||^2`` diverges.
        read_before_write: Read the state before writing it. Mandatory: the
            tournament measured 3.1623 vs 3.1251 BPC the other way.
        use_read_gate: Whether to apply the read gate. The tournament measured
            +0.037 BPC without it, so it is on by default.
        write_mode: ``"delta"`` (error-correcting, the architecture) or
            ``"additive"`` (plain accumulation, kept as the ablation baseline
            the delta rule is measured against).
    """

    d_k: int
    d_v: int
    d_in: int
    n_banks: int
    W_k: NDArray[np.floating] = field(init=False)
    W_q: NDArray[np.floating] = field(init=False)
    W_v: NDArray[np.floating] = field(init=False)
    W_r: NDArray[np.floating] = field(init=False)
    bank_logits: NDArray[np.floating] = field(init=False)
    eta: float = 0.5
    state: NDArray[np.integer] = field(
        # int32, not the np.zeros float64 default: the field is declared integer
        # and the state is int8-valued but carried in int32 for arithmetic.
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32),
        init=False,
    )
    reads: int = 0
    writes: int = 0
    normalize_keys: bool = True
    read_before_write: bool = True
    use_read_gate: bool = True
    write_mode: str = "delta"
    seed: int = 0
    name: str = "custom"
    head_index: int = 0

    def __post_init__(self) -> None:
        """Allocate projections and the state.

        Why the projections are stored dequantized: the int8 regime is a
        property of the deployed op sequence, not a licence to run the
        reference on unscaled int8 codes. The per-column absmax scale is
        applied here, once, so that ``X @ W_v`` lands on the unit grid that
        :func:`quantize_activation` and the read-gate sigmoid both expect.
        See the note in :meth:`proj` and ``docs/architecture.md``.

        The stream is keyed by ``head_index`` as well as the shape. Keying on
        shape alone gave every head of a layer the same stream, so the four
        "independent" memories were byte-identical and stayed that way for the
        whole forward pass. See :mod:`bhanox.seeding`.
        """
        rng = init_rng(self.seed, "head", self.name, self.head_index)
        d = self.d_in

        def proj(rows: int, cols: int) -> NDArray[np.floating]:
            w = rng.standard_normal((rows, cols)) / np.sqrt(rows)
            # Dequantized, not the raw int8 codes. docs/architecture.md records
            # this exact failure for MicroExpert ("a float matmul against a
            # 127x-scaled matrix gives router logits with a spread of ~500,
            # which saturates the softmax") and it applies here unchanged: the
            # codes are +/-127, so `X @ W_v` lands at a median |v| of ~430
            # against a quantize_activation grid that assumes |x| <= 1.
            #
            # Measured at nano before this fix: 99.9% of value projections
            # clipped, `v8` reduced to 2-6 distinct codes out of 255, and the
            # read gate 99.3% saturated -- the gate the spec credits with
            # +0.037 BPC was a hard on/off switch with no usable gradient. The
            # L2 normalisation hid it for W_k and W_q only; W_v reaches the
            # quantiser and W_r the sigmoid unscaled.
            #
            # The int8 regime is a property of the deployed op sequence, not a
            # licence to run the reference on unscaled codes.
            #
            # Measured on the real residual stream at nano, this init is already
            # well placed: median |X @ W_v| 0.677 (grid assumes |x| <= 1) with
            # 250-254 distinct `v8` codes across heads. Rescaling W_v to fill
            # the grid further was tried and reverted: it was sized against the
            # synthetic items in tests/core/test_deltabank.py, which are ~11x
            # smaller than real activations, and on the real stream it pushed
            # the median to 3.835, cut `v8` to 159-169 distinct codes and
            # clipped 86.4% of value codes. Do not re-tune this without
            # measuring on _residual_stream, not on a test fixture.
            return absmax_quantize(w, axis=0).dequantize().astype(np.float32)

        self.W_k = proj(d, self.d_k)
        self.W_q = proj(d, self.d_k)
        self.W_v = proj(d, self.d_v)
        self.W_r = proj(d, self.d_v)
        # Bank logits initialised to a near-uniform prior over decay rates, so
        # every timescale starts represented and learning refines the mixture.
        self.bank_logits = np.zeros((self.n_banks, self.d_k), dtype=np.float32)
        # One sample to start. forward() grows this to the batch it is handed,
        # so batch-1 callers (step, generate, every invariant check) allocate
        # exactly what they did before per-sample state existed.
        self.state = np.zeros((1, self.d_k, self.d_v), dtype=np.int32)

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Zero the state. O(1): the state has no per-token component."""
        self.state.fill(0.0)
        self.reads = 0
        self.writes = 0

    @property
    def state_nbytes(self) -> int:
        """int8 bytes of state *per sample*, constant in context length.

        One byte per element, not ``state.itemsize``: the state holds int8 codes
        and the int32 carrier exists only so the reference's intermediate
        products cannot overflow. The native M4 kernel stores these as int8, and
        this is the number invariant I2 audits.

        Per sample, deliberately: I2 asks what one stream costs, and a batch of
        ``B`` streams costs ``B`` times this. Reporting the allocation would
        make the invariant drift with whatever batch size the model last saw,
        which is not a property of the architecture. Use
        :attr:`batch_state_nbytes` for what is actually resident.
        """
        return int(self.d_k * self.d_v)

    @property
    def batch_state_nbytes(self) -> int:
        """int8 bytes the state actually occupies, across every live sample."""
        return int(self.state.size)

    def ensure_batch(self, batch: int) -> None:
        """Grow the state to hold ``batch`` independent streams.

        Args:
            batch: Number of concurrent samples. Must be at least one.

        Raises:
            ValueError: If ``batch`` is not positive.

        Why growth is safe: the state is O(1) in context length and O(batch) in
        batch, and new samples start at zero, which is exactly what a fresh
        stream's state is. Nothing is copied but the existing rows.
        """
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        have = self.state.shape[0]
        if batch <= have:
            return
        grown = np.zeros((batch, self.d_k, self.d_v), dtype=np.int32)
        grown[:have] = self.state
        self.state = grown

    # -- decay ---------------------------------------------------------------

    def decay(self, bank_rates: NDArray[np.floating]) -> NDArray[np.floating]:
        """Per-channel decay ``lambda_t`` as a learned mix over decay banks.

        Args:
            bank_rates: The frozen bank prior ``[1 - 2**-b for b in 1..B]``.

        Returns:
            ``(d_k,)`` array of decay rates in ``[0, 1)``.

        Why log-space weights over a convex mix: the prior is a fixed geometric
        ladder, and a softmax mixture over it can represent any rate in the
        range without ever being able to emit a non-decaying or negative rate.
        That constraint is what makes the state provably bounded.
        """
        z = self.bank_logits - self.bank_logits.max(axis=0, keepdims=True)
        w = np.exp(z)
        w /= w.sum(axis=0, keepdims=True)
        return (w * bank_rates[:, None]).sum(axis=0)

    def decay_q(self, bank_rates: NDArray[np.floating]) -> NDArray[np.int32]:
        """Per-channel decay in Q16 fixed point, ready for the int8 recurrence.

        Args:
            bank_rates: The frozen bank prior.

        Returns:
            ``(d_k,)`` int32 array of ``round(lambda * 2**16)``.

        Why the rounding happens once, here: a decay rate is not a whole
        number, so storing it as a float would put a float multiply on the
        hottest path in the model. Rounding to Q16 once, then using only
        multiply-shift-saturate, is what keeps the recurrence genuinely int8.
        """
        return to_q(self.decay(bank_rates))

    # -- the recurrence ------------------------------------------------------

    def forward(
        self, x: NDArray[np.floating], bank_rates: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Advance every sample's state by one token and return their outputs.

        Args:
            x: ``(B, d_in)`` activations, or a single ``(d_in,)`` vector, which
                is treated as one sample.
            bank_rates: Decay bank prior, shape ``(n_banks,)``.

        Returns:
            ``(B, d_v)``, or ``(d_v,)`` if ``x`` was a single vector.

        Raises:
            ValueError: If ``x`` has the wrong width.

        Per-sample state: sample ``b`` only ever touches row ``b``. That is the
        whole point -- before this, ``B`` rows shared one state, so row 1 read
        what row 0 had written and a batched loss was measuring the wrong thing.
        Every operation below is integer, and integer addition is associative,
        so batching cannot change a single bit of the batch-1 result.
        """
        arr = np.asarray(x, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[-1] != self.d_in:
            raise ValueError(
                f"DeltaBankHead expected d_in={self.d_in} as (B, {self.d_in}) or "
                f"a single ({self.d_in},) vector, got shape {arr.shape}"
            )
        batch = arr.shape[0]
        self.ensure_batch(batch)
        # Slice, don't just ensure. A model that ran a batch of 32 and is now
        # handed a batch of 1 still holds 32 state rows, and einsum would
        # happily broadcast the single query against all of them and hand back
        # 32 answers for 1 token.
        state = self.state[:batch]
        # Project, then L2-normalise K and Q (finding 1: mandatory -- without
        # unit-norm keys the delta rule's beta = 1/||k||^2 diverges).
        k, q, v = arr @ self.W_k, arr @ self.W_q, arr @ self.W_v
        if self.normalize_keys:
            k = l2_normalize(k)
            q = l2_normalize(q)

        # Quantise to the int8 activation grid. Everything below is integer.
        k8 = quantize_activation(k)
        q8 = quantize_activation(q)
        v8 = quantize_activation(v)

        if self.read_before_write:
            # READ FIRST (finding 2). Both the read and the surprise come from
            # the pre-write state, so the fresh write cannot clobber the very
            # association being retrieved.
            #
            # Two different dot products, and they must stay that way: the read
            # is S^T q (what the model gets to see) while the surprise is
            # S^T k (what the write is about to overwrite). Collapsing them to
            # one is the kind of "simplification" that makes the recurrence
            # diverge, because the error term then never points along the key
            # the write actually applies.
            #
            # Units: `state` holds 127 * S and q8/k8 hold 127 * q / 127 * k, so
            # each int32 dot product is in 127^2 * r. One integer divide by
            # INT8_MAX brings the surprise back to code units; the real-valued
            # read needs the full 127^2. In the native kernel both are shifts.
            acc_q = np.einsum("bkv,bk->bv", state, q8)
            acc_k = np.einsum("bkv,bk->bv", state, k8)
            e8 = saturate_int8(v8 - acc_k // INT8_MAX)
            state = self._write(state, k8, e8, v8, bank_rates=bank_rates)
            r = acc_q / float(INT8_MAX**2)
        else:  # pragma: no cover - reached only by the ordering regression test
            # Write first, then read. Kept because the difference is a
            # regression test, not a configuration anyone should use.
            state = self._write(state, k8, v8, v8, bank_rates=bank_rates)
            r = np.einsum("bkv,bk->bv", state, q8) / float(INT8_MAX**2)
        self.state[:batch] = state
        self.reads += batch
        self.writes += batch

        # Read gate (finding 3): +0.037 BPC without it. Per read channel,
        # before the read-out, so it can suppress a channel the model does not
        # currently want.
        out = r * _sigmoid(arr @ self.W_r) if self.use_read_gate else r
        return out[0] if single else out

    def _write(
        self,
        state: NDArray[np.integer],
        k8: NDArray[np.integer],
        e8: NDArray[np.integer],
        v8: NDArray[np.integer],
        *,
        bank_rates: NDArray[np.floating],
    ) -> NDArray[np.integer]:
        """Apply the delta-rule write plus per-channel decay, in int8 fixed point.

        Args:
            state: The active state rows, ``(B, d_k, d_v)``.
            k8: Unit-norm keys on the int8 grid, ``(B, d_k)``.
            e8: Surprise ``v - S^T k`` already in code units, ``(B, d_v)``.
            v8: Values on the int8 grid, ``(B, d_v)``.
            bank_rates: Decay bank prior.

        Returns:
            The new state rows. Written back by the caller so a narrower batch
            cannot clobber rows belonging to other samples.

        Every step is one of MUL / SHIFT / SAT / ADD, all whitelisted by I3.
        """
        lam_q = self.decay_q(bank_rates)  # (d_k,) Q16, shared across samples
        # beta = LUT_recip[||k||^2] * eta, in Q16, per sample. The table is
        # 1-indexed by the spec's LUT_recip[||k||^2], so ||k||^2 == 1 reads
        # entry 1 == 1.0 and beta is exactly eta. This is why K L2-normalisation
        # is mandatory rather than merely tidy.
        k_sq = np.einsum("bk,bk->b", k8, k8).astype(np.float64) / float(INT8_MAX**2)
        index = np.clip(np.rint(k_sq), 1, 255).astype(np.int64)
        # float64 before the Q16 rounding, deliberately. `_RECIP` is float32 and
        # NumPy 2 keeps `float32 * python_float` in float32, so the batched form
        # silently rounds beta at 24 bits and then again at 16. The single-sample
        # form went through Python's `float()`, which widens first. The two
        # disagree by 1 in Q16 often enough to flip the gate's hysteresis, and
        # batch-1 then stops reproducing.
        beta_q = to_q(_RECIP[index - 1].astype(np.float64) * float(self.eta))[:, None]
        # Diag(lambda) S: decay scales state ROWS (key channels), so it is a
        # left multiplication. Q16 rate against a unit-range state: one shift.
        decayed = qmul(state, lam_q[None, :, None])
        # "additive" is the ablation baseline: write v k^T with no error term,
        # i.e. plain accumulation. The delta rule's whole argument is that this
        # is worse, and the architecture's T3 measurement (recall 1.000 vs
        # 0.815) is a claim about exactly this comparison, so the baseline has
        # to be runnable rather than described.
        if self.write_mode == "additive":
            e8 = v8
        # beta * e, in code units.
        scaled_e = qmul(beta_q, e8)
        # The rank-1 term is an int8 x int8 product accumulated in int32 -- the
        # MUL_I8 a VNNI instruction performs. Its magnitude (~8000) far exceeds
        # the int8 state range, so it needs the requantisation step every int8
        # GEMM has: divide by INT8_MAX to bring the product back to code units.
        # Without it the state slams into saturation and recall goes to zero.
        update = (k8[:, :, None] * scaled_e[:, None, :]) // INT8_MAX
        return saturate_int8(decayed + update)

    # -- introspection -------------------------------------------------------

    def recall(self, k: NDArray[np.floating]) -> NDArray[np.floating]:
        """Read the state with a query key, without writing.

        Args:
            k: Query key of width ``d_k``, normalised internally. A ``(d_k,)``
                vector reads sample 0; a ``(B, d_k)`` block reads one key per
                sample, each from its own state row.

        Returns:
            ``(d_v,)`` for a single key, else ``(B, d_v)``, in real units.

        Why the quantise-then-divide: this has to agree with the read inside
        :meth:`forward` to the last bit, and that read is int8 codes against
        int8 codes. Returning ``state^T q`` with a raw unit vector would hand
        back codes at 127x the intended scale, which looks like a working
        recall and is not one.
        """
        arr = np.asarray(k, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        qn = l2_normalize(arr)
        q8 = quantize_activation(qn)
        rows = self.state[: arr.shape[0]]
        out = np.einsum("bkv,bk->bv", rows, q8) / float(INT8_MAX**2)
        return out[0] if single else out

    def memory_accuracy(
        self, keys: NDArray[np.floating], values: NDArray[np.floating]
    ) -> float:
        """Fraction of key/value pairs retrieved within a tolerance.

        Args:
            keys: ``(n, d_k)`` probe keys.
            values: ``(n, d_v)`` target values.

        Returns:
            Accuracy in ``[0, 1]``. The design phase's target is > 0.9 for an
            underloaded head; this is the metric that makes invariant I1
            observable rather than merely asserted.
        """
        tol = 0.05 * max(1e-6, float(np.abs(values).max()))
        hits = 0
        for k, v in zip(keys, values, strict=True):
            if np.all(np.abs(self.recall(k) - v) <= tol):
                hits += 1
        return hits / max(1, len(keys))
