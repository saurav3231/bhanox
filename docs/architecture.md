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

Representation, which is worth knowing before you touch `quantize()` or
`embed()`. `pool` and `table` hold int8 **codes**; the per-column absmax scale
lives beside them in `pool_scale`/`table_scale`, and `embed` applies it. This is
the opposite convention to `DeltaBank` and `MicroExpert` below, and on purpose:
the front-end's arrays *are* the int8 grid that gets deployed, so storing
dequantized floats in a field documented as codes would be a rescale wearing a
quantization's name. Both halves have to be named, because either one alone is a
silent 283x-795x error — the codes read as real values are that far too large,
and the scales are checkpointed state rather than derived data. `nbytes` counts
the scales; at nano they are 1,024 B of 1,180,672 B (0.087%), and a residency
number that omits an array it holds stops being true as the model grows.

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

`n_heads` buys *distinct* memories only as of `d0c6724`. Before it, every head
of a layer drew the same init stream and stayed byte-identical through the whole
forward pass, so `n_heads` was `n_heads` copies of one memory paying `n_heads`
times the arithmetic. The streams are now keyed per head
(`bhanox.seeding.init_rng`). The mechanism is real; the capacity benefit is
measured: a head with `d_k=16` holds 16–24 items at ≥50% retrieval accuracy
(median 20) and 4 heads hold 48–64 (median 48), ratio 2.40x–4.00x (median
2.67x) across five item draws. The ideal 4x is not reached because the heads
share one input space. See `TestHeadCapacity` in
`tests/core/test_deltabank.py`.

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

The same split applies to the NumPy reference against the Torch mirror, and for a
sharper reason. The two stacks differ by ~1e-6 in float32, which is usually
invisible — but if that difference lands within 1e-6 of a half-integer, the two
`round` calls select different int8 codes, and one flipped code moves the read by
a full quantisation step. So the mirror claim has **two regimes**:

- **Same int8 codes.** The integer recurrence is fed bit-identical input, so the
  int32 states compare with `array_equal` and the logits stay within the tight
  bound (1.79e-6 measured on the two-layer config, 3.81e-6 on nano).
- **A code flips at a quantisation boundary.** The states are still bit-exact up
  to the first flip, but the logits get a wider, separately measured bound,
  because a single int8 step genuinely moves them. This is a property of
  comparing two float32 implementations across a quantiser, not a mirror defect:
  forcing agreement would mean reproducing NumPy's exact accumulator order.

Both bounds are empirical and scoped to the config and sequence lengths they were
measured on. The head-side `1/127` is *not* a valid end-to-end limit — it bounds
the read, not what the mixer, layer norm and unembed do to it. Full grid, measured
worst cases, and the structural check that every code difference really is a
half-integer straddle are in `ROADMAP.md` and `tests/test_model_mirror.py`.

The time loop stays outermost in `_run_memory`, because the recurrence is
sequential in `t` and there is nothing to vectorise across it. The batch is
real rather than a throughput knob for the mixer alone.

Batch-1 behaviour is unchanged by this work, and that was measured rather than
assumed: on the nano config, the same prompt gives the same int32 state, the
same gate decisions per token, and float32 logits bit-identical to the
pre-M2 path (`max|diff| == 0.0`). See the note on `Bhanox.forward`.

## Checkpoints

`src/bhanox/checkpoint.py`. Versioned, atomic, NumPy-only, and bit-exact — 124
tensors round-trip with `max|diff| == 0.0` and identical logits.

Three decisions, each of which is a way this could have gone wrong:

| decision | why |
|---|---|
| tensors discovered by walking the model | a hand-written field list silently drops any parameter added later, and training resumes from a checkpoint that quietly lost it |
| write to a temp file, then `os.replace` | a crash mid-write must not destroy the checkpoint that was already there |
| parameters only, no recurrent state | keeps a checkpoint proportional to parameters rather than context, and stops a file's contents from depending on the batch size of the run that wrote it |

The walk found a discrepancy the hand-written count had missed, which is the
argument for it arriving at this point rather than later. `param_count()`
reported 2,107,460 values and omitted the gates' `tau_hi`/`tau_lo`/`salience`
— 1,536 learned, trained parameters. The count is now 2,108,996, and
`test_param_count_agrees_with_the_checkpoint_walk` pins the two together so a
parameter cannot be counted in one place and missed in the other. What the
checkpoint holds that the count does not is 32 values of `bank_rates`, a cached
broadcast of `cfg.decay_rates`; it is pinned by name, so "derived from config"
cannot become a place to hide a forgotten parameter.

Note the two figures that are easy to confuse. The **packed** size in the
benchmark tables is 1 byte per value — the int8 budget the model is designed
against, and what `packed_nbytes()` means. A checkpoint **file** is float32 and
is about 4x that. Both are honest; they answer different questions.

The third decision above has a consequence worth stating plainly:
`from_pretrained` returns a model with an **empty** state, and resuming
mid-sequence without carrying the state is a different computation. `forward`
continues from existing state rather than resetting, so a trainer that keeps the
recurrent buffers alongside the checkpoint gets a faithful resume. Both
directions are pinned in `tests/test_checkpoint.py` — carrying it matches the
uninterrupted run, and dropping it does not, so neither test can pass for the
wrong reason. The matching assertion is on the **weights** (`== 0.0`); the logits
are compared with a documented float32 tolerance, because the two paths reach the
same answer by summing in a different order and BLAS picks that order per build.

### The walk carries arrays, so scalars are a separate block

The same walk that cannot lose a parameter also cannot see a `filled` that is a
plain `int`. The VectorVault is the case where that matters: restoring its slot
tables while `filled` and `clock` came back at `0` produced a vault that
answered no query — `query` short-circuits on `filled == 0` — and that handed out
slot 0 on every write, overwriting one recovered entry per write and orphaning
the rest, with no exception anywhere on the path.

So the vault's `filled`, `clock`, `theta_s` and `perm_budget_bytes` travel in a
namespaced `runtime` block in the metadata, versioned separately, and `load()`
restores them automatically and validates them against the arrays beside them
(`n_slots` is recorded as a cross-check, not restored). Two properties follow:

- **A resume is consistent or refused.** Validation runs *before* any array is
  written, so a bad block leaves the target untouched. A checkpoint holding vault
  arrays with no such block is refused, because restoring it is the bug.
  `filled=0` next to populated arrays is no longer reachable.
- **The version stays 1.** The tensor layout did not change, and nano and mini
  checkpoints — which have no vault — read exactly as before. The block's own
  `version` field is what a future layout moves.

Pinned in `tests/test_checkpoint_vault.py`, including the behavioural claim: two
models differing only in how many times they were loaded agree slot-for-slot and
clock-for-clock on the next ten writes.

This fixes one item only. **Exact training resume is still BLOCKED** — there is no
optimizer, scheduler, RNG state or data position to restore. See *ROADMAP*.

## Generation: two id spaces

The model does not use one "token". `HashBind` embeds one packed byte 4-gram per
position, and the unembedding is `(d_model, output_vocab)` with
`output_vocab == 256`. So a model *input* is a 32-bit id naming four bytes, and a
model *output class* is a single byte value in `0..255`. The two spaces are not
interchangeable, and nothing about the types says so: a byte value is a valid
`int64` and therefore a valid input id.

Generation bridges them by rolling. After sampling byte `b`, the next context is
the previous three bytes plus `b`, re-encoded by `encode_bytes` — never `b`
itself, which would be the unrelated 4-gram `0x000000XX`. The public API is
byte-oriented (`model.generate(prompt: bytes, ...) -> bytes`) so the ids stay
internal, and `generate_ids` exists for callers who already hold them. It returns
*only* the generated byte values, as `uint8`, because the caller still has the
prompt ids and the two must not end up in one array.

Three consequences worth stating:

- `str` prompts are refused rather than encoded. The output is bytes and is not
  guaranteed to be valid UTF-8, so an API that quietly converted text in would
  invite reading the result as text.
- `output_vocab != 256` is refused before sampling. `BhanoxConfig` only asks for
  `output_vocab > 0`, so a 1000-way head is constructible, and its sampled class
  would not be a byte.
- The loop feeds one context at a time via `step`, not a window via `forward`, so
  generation is not bounded by `max_context`. That limit exists for training
  windows; the model is O(1) in context at runtime and can be prompted with
  anything that fits in memory.

`tests/test_gram_contract.py` pins the rolling with a recording stub that
substitutes `step` and captures the ids the model is actually fed, then asserts
those ids are exactly the 4-gram windows of the returned bytes. A bare byte feed
is a valid `int64`, so this is checked by observation rather than by type.

## Assembly

`src/bhanox/model.py` is not in the frozen architecture table. It exists because
the public API needs one place that owns the layer stack, and the alternative is
the API living in `__init__.py`, which is worse (ADR-001).
