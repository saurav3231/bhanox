# Benchmarks

Every number here is either **measured** (a script in this repo produced it) or
**modeled** (a component model, not an end-to-end measurement). Nothing is
estimated silently — that is the honesty law, A3.

Regenerate the reference figures with:

```bash
python scripts/benchmark.py --config nano --tokens 512
python examples/audit_demo.py
```

---

## Audit: which numbers survived the d0c6724 init fix

`d0c6724` made weight init reproducible and gave each head and layer its own
stream. Before it, the read-out, bypass, unembed and MoE weights were redrawn
every process (`hash()` on a str), and all `n_heads` heads of a layer shared one
stream — byte-identical, and identical through the whole forward pass.

That is a behaviour change, so every measured figure was re-checked rather than
assumed. **None turned out to be stale**, for reasons worth recording:

| figure | verdict | why |
|---|---|---|
| params, packed size, state B, B/token, L2 %, I2 | unaffected | computed from shapes, not weight values. `param_count` re-checked at 2,107,460. |
| reference speed, ~51 tok/s | unaffected | timing does not depend on weight values; `n_heads` still does the same arithmetic. Re-measured 4x: 45.4 / 45.8 / 56.5 / 55.6 — the spread is machine variance, and ~51 sits inside it. |
| delta rule vs additive, 0.213 vs 0.375 | unaffected | re-measured identical. The test injects its own weights (`default_rng(11)`) in `make_head`, so it never depended on the init key. |
| decay banks, 5,430 vs 2,456 (2.21x) | unaffected | re-measured identical, same reason. Also pinned by `pytest.approx(2.21, abs=0.01)`, so it cannot drift silently. |
| PulseGate 75% / 0% | unaffected | the gate has no weights; it is driven by synthetic input. |
| MicroExpert 5.7x, VectorVault, HashBind | unaffected | byte counts and self-seeded components, untouched by the fix. |

**What is still unmeasured is the multi-head capacity claim.** The architecture
table advertises `n_heads` as the sanctioned way to add independent memories.
Until `d0c6724` that could not have been true — the heads were copies — so no
figure here ever demonstrated a benefit from `n_heads > 1`. The claim is now
measured: a head with `d_k=16` holds 16–24 items at ≥50% retrieval accuracy
(median 20) and 4 heads hold 48–64 (median 48), ratio 2.40x–4.00x (median
2.67x) across five item draws. The ideal 4x is not reached because the heads
share one input space and interference is not zero. See the regression guard at
`tests/core/test_deltabank.py::TestHeadCapacity`.

---

## The one number that is not measured

**P1: `>= 20,000 tokens/s`.** Status: **NOT MEASURED**, pending M4.

P1 is a claim about the *native* runtime. The NumPy reference cannot settle it,
and quoting the reference figure as though it did would be the exact failure this
project is trying to avoid. The reference pays for float32 temporaries and a
Python-level loop per token per layer; the native runtime does int8 work straight
out of a fixed buffer. Expect one to two orders of magnitude between them.

---

## Measured: cost and residency

Batch 1, greedy, `scripts/benchmark.py`. L2 budget 1,048,576 B.

| preset | params | packed (int8) | state | B/token/layer | of L2 | I2 |
|---|---|---|---|---|---|---|
| `nano` | 2,107,460 | 2.11 MB | 8,192 B | 110,592 | 10.5% | PASS |
| `mini` | 24,338,948 | 24.34 MB | 131,072 B | 720,896 | 68.8% | PASS |
| `small` | 324,929,540 | 324.93 MB | 1,048,576 B | 3,538,944 | 337.5% | **FAIL** |

`small` is 325 M parameters, which is roughly 17 MB of dense int8 expert weights
against a 1 MiB cache. The mixer is not the problem — it touches 524,288 B, 5.7x
less than dense. The DeltaBank projections are, at 3,014,656 B per layer.

Nano's per-layer breakdown:

| component | B/token | note |
|---|---|---|
| `deltabank` | 86,016 | 4 heads, state read + written |
| `microexpert` | 24,576 | 1+2 of 17 experts |
| `hashbind` (touched) | 512 | 4 rows of an 8.2 MB resident pool, whole model |

The front-end is charged once for the model, not once per layer.

## Measured: reference speed

| figure | value |
|---|---|
| tokens/s, nano, batch 1, greedy | ~51 (32 tokens, 0.63 s) |
| tokens/s, nano, batch 1, greedy | ~51 (reference, **not** P1) |

Machine-dependent. The harness prints the interpreter and platform with every
run so a number is never quoted without its context.

## Measured: component mechanisms

Every value below is what the NumPy reference produces today, on an untrained
model. The cited test is the regression guard.

| claim | value | test |
|---|---|---|
| State is O(1) in context | 8,192 B after 64 and after 256 steps | `test_state_is_constant_over_a_long_context` |
| Delta rule vs additive recall | 0.213 vs 0.375 | `test_delta_rule_beats_additive_writes` |
| Decay banks vs single fast decay | 5,430 vs 2,456 state energy (2.2x) | `test_decay_banks_outlast_a_single_fast_decay` |
| PulseGate skip, steady input | 75% | `test_steady_input_is_mostly_skipped` |
| PulseGate skip, random input | 0% | `test_noisy_channel_never_sleeps` |
| MicroExpert sparsity | 5.7x fewer bytes than dense | `test_touches_far_fewer_bytes_than_dense` |
| VectorVault near vs far distance | 84 vs 399 of 1024 bits | `test_near_vectors_are_closer_than_distant_ones` |
| VectorVault recall, 10% noise | 1.000 | `test_noise_still_recovers_most_items` |
| HashBind unknown ids | embed with no table row | `test_unseen_token_still_embeds` |
| HashBind negative ids | no table row, not the last one | `test_negative_id_gets_no_direct_row` |
| 4 heads vs 1 head capacity | 48–64 vs 16–24 items at ≥50% (median 2.67x) | `TestHeadCapacity.test_four_heads_retain_more_than_one` |

The absolute recall and distance numbers are low because the model is
untrained; the claims are about the *sign* and the ratio, which is what the
tests assert. The design-phase tournament scored some of these higher on its
own harness (1.000 vs 0.815 for the delta rule, for example). Those are
**design-phase figures, not reference output**, and are not quoted as
measurements anywhere in this project.

## Not measured

| claim | why not |
|---|---|
| P1 `>= 20,000 t/s` | needs the M4 native runtime |
| I4 backend agreement | needs the M4 native runtime |
| Quality (ppl, BPC) | no model trained until M2 |
| Energy vs a quality-matched Transformer | needs RAPL on bare metal |

## Modeled

The energy claim targets `>= 95%` reduction against a quality-matched
Transformer, and ships as **modeled**. It is a compound of the component
mechanisms above — O(1) state means no cache growth, int8 means 4x less state
traffic, sparse activation means a fraction of the weights — each of which is
individually verified, but no end-to-end measurement exists. The individual
component numbers are measured; the product is not.

Two things to keep in mind when reading it:

- Energy per token *rises* with context for a Transformer and stays flat for
  Bhanox, so the advantage grows with sequence length. The crossover point is a
  model, not a measurement.
- The reference implementation's own energy profile is irrelevant to the target.
  It measures the architecture, not the reference.

See `docs/invariants.md` for what is enforced versus what is still open.
