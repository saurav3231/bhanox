# Architecture

The frozen design, and where each piece lives in the code. Everything here is
checked by a test; where a test is missing, that is called out.

## The problem

A Transformer spends memory and energy on a cache that grows with every token
you read. Bhanox replaces that cache with a **fixed-size** state that is
*corrected* on write rather than appended to. Fixed size is the whole point: it
is what makes the per-token cost O(1) and the energy bill flat.

## The layer

Per layer, from `src/bhanox/model.py:forward`:

```
x <- x + DeltaBank(x)      # recurrent memory, constant cost
x <- x + MicroExpert(x)    # sparse capacity
x <- LayerNorm(x)          # keeps the residual stream bounded
```

Layer norm is a per-column absmax rescale, which stays in the integer domain
under the default regime. The reference uses the standard mean/variance form
because that is the definition the native runtime and the PyTorch mirror must
agree with to 1e-3 (I4).

## The five components

### HashBind — `src/bhanox/frontend/hashbind.py`

Embeds any byte 4-gram. No vocabulary, no `<UNK>`. The frozen equation is a
**sum**, not a choice: a known id gets its dedicated table row *in addition to*
its hashed contribution, and an unknown id gets only the hashed one.

The one sharp edge: the known-id test is `(arr >= 0) & (arr < vocab_table)`. The
lower bound is load-bearing. Without it a negative id passes and numpy then
indexes the table from the end, so `-1` silently reads the last row.

### DeltaBank — `src/bhanox/core/deltabank.py`

Multi-head, multi-timescale recurrent memory with a delta-rule (error-correcting)
write. The write *repairs* the existing association instead of adding a second,
slightly different copy, which is what stops a fact from smearing across the
state.

The read is a convex mix of the bank logits, so `decay_q` and `decay` agree by
construction rather than by coincidence. State is int8 logically, carried in
int32 for arithmetic.

Measured: the delta rule reaches 1.000 recall against 0.815 for additive writes
under the same load; decay banks retain 0.94 in the worst bin against 0.16 for a
single fast decay.

### MicroExpert — `src/bhanox/mixer/microexpert.py`

16–128 routed experts plus shared, top-2 routing. Per token it touches
`n_shared + top_k` experts out of `n_experts + n_shared`, which is 5.7x fewer
bytes than the dense equivalent at nano scale.

One thing worth knowing before you touch the initialisation: the weights are
stored **dequantized**, not as raw int8 codes. Codes are ±127, and a float
matmul against a 127x-scaled matrix gives router logits with a spread of ~500,
which saturates the softmax so the router picks the same two experts forever and
the load-balancing bias cannot move it. The int8 regime is a property of the
deployed op sequence, not a licence to run the reference on unscaled codes.

The load-balancing bias is updated from **per-call** counts, not lifetime
counts. A lifetime average cannot correct an imbalance that happened once — it
is permanently baked in — and it makes the correction weaker the longer training
runs, which is backwards.

### PulseGate — `src/bhanox/governor/pulsegate.py`

Skips any channel whose input did not meaningfully change. Hysteresis
(`tau_lo`/`tau_hi`) plus a learned salience threshold, with the top
`salience_frac` of channels protected so the most important ones can never be
skipped, ties included.

Measured: 75% of channel-steps skipped on steady input, 0% on random input.

### VectorVault — `src/bhanox/memory/vectorvault.py`

Optional HDC episodic store, off for nano and mini. Bipolar hypervectors, XOR
plus popcount.

Measured: near/far Hamming distance 158 vs 4435 of 8192, 0.9 recall with 10%
noise. The reference caches a random projection for bipolar encoding, which is
about 16 MB at d=512; the M4 native runtime should use a counter-based hash
instead of materialising it.

## Memory budgets

Both memories are explicit, bounded, user-settable byte budgets. This is the one
thing a Transformer cannot do to its KV cache and a Mamba cannot do to its
state, so it is a feature rather than a convenience.

```python
cfg = bhanox.load_config("mini", temp_mem="64MB", perm_mem="2GB")
model.set_perm_budget("4GB")        # live; the vault admits or stops admitting
print(bhanox.memory_report(model))  # bytes in use against each budget
```

Sizing, as the code actually measures it:

| memory | bytes | notes |
|---|---|---|
| TEMP (DeltaBank state) | `n_layers * n_heads * d_k * d_v` | int8, 1 B per cell. nano 8,192 B; mini 131,072 B |
| PERM (VectorVault) | `n_slots * (bits // 8 + 4 * d_value + 12)` | the hypervector key dominates |

The design-phase spec wrote these as `L * B * 2d` and `E * (2d + 16)`. Both
undercount the built layout, the second by 14x, because they assume a key is `d`
wide when it is a hypervector of `bits` bits. A budget check that undercounts is
worse than no budget, so the formulas above are the ones the code measures and
`test_temp_matches_the_measured_state` holds them against the real arrays.

What is tied to the trained model, and what is a free knob:

| piece | tied to params | user-settable |
|---|---|---|
| `d_k`, `d_v` (state shape) | yes | no |
| `n_heads` (state area) | yes | no |
| `n_banks` (decay schedule) | no | config-time, **costs no bytes** |
| vault entry cap `E` | no | yes, live |

`n_banks` is the one the spec got wrong in the other direction: `L * B * 2d`
implies buying memory by widening the bank mix, but B only selects decay rates.
Widen B and the state does not move.

Budgets bound **bytes**, not admissions. The vault allocates its slot table up
front, so a budget that merely stopped writes would still hold 1.5 MB resident
while reporting 64 KB. `set_budget` resizes the table, so a 64 KB budget really
costs 64 KB. Lowering one truncates by the same salience x recency score the
vault evicts with, so shrinking a budget drops exactly the entries the vault
would have overwritten anyway. Oversubscribing is legal and warns once;
undersubscribing never raises, because running out of room must cost recall
rather than the process.

A budget is a ceiling, not a purchase order. `set_budget("4GB")` does not
allocate 4 GB: the table is `min(entry_cap, budget limit)`, so the entry cap `E`
governs how much is really reserved and a loose budget is simply not binding.
Growing `E` is a separate, deliberate call (`set_entry_cap`), and it refuses to
drop entries already stored. Widening `n_banks` costs nothing, because it only
selects decay rates.

## Batch semantics

The recurrent state carries a **sample axis**: `(B, d_k, d_v)` for the
DeltaBank, `(B, n_channels)` for the PulseGate. Sample `b` reads only what
sample `b` wrote, so `forward(stack([a, b]))` is no longer
`forward(concatenate([a, b]))`. A batched loss measures what it claims to.

Two claims, deliberately kept apart:

- **State is exact.** The recurrence is int32, so a batched row and the
  equivalent single-stream run leave states that compare equal with `==`.
- **Logits are approximate.** Logits are float32 out of a matmul over a
  `(B, T)` block, and BLAS sums a batched product in a different order than a
  single-row one. Compare logits with `allclose`, state with `array_equal`.

The time loop stays outermost in `_run_memory`, because the recurrence is
sequential in `t` and there is nothing to vectorise across it. The batch is
real rather than a throughput knob for the mixer alone.

Batch-1 behaviour is unchanged by this work, and that was measured rather than
assumed: on the nano config, the same prompt gives the same int32 state, the
same gate decisions per token, and float32 logits bit-identical to the
pre-M2 path (`max|diff| == 0.0`). See the note on `Bhanox.forward`.

## Assembly

`src/bhanox/model.py` is not in the frozen architecture table. It exists because
the public API needs one place that owns the layer stack, and the alternative is
the API living in `__init__.py`, which is worse (ADR-001).
