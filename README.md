# Bhanox

**A CPU-native neural architecture substrate.** O(1) cost per generated token,
int8 arithmetic only, no data center required.

> In simple words: language models today need a room full of GPUs. Bhanox is
> being built so that a useful one runs on the laptop in your bag — and so that
> the energy bill is a rounding error instead of the line item.

**Status: pre-alpha.** The architecture is frozen, the reference
implementation exists, and nothing has been trained yet. The table below says
exactly which numbers are real. Numbers that have not been measured are marked
`modeled` and are not claims.

---

## Quickstart

```bash
git clone https://github.com/saurav3231/bhanox
cd bhanox
pip install -e ".[dev]"
```

```python
import bhanox

model = bhanox.Bhanox(bhanox.load_config("nano"))   # untrained reference model
ids = bhanox.frontend.encode("hello world")          # byte 4-grams, open vocab
print(model.generate(ids, max_new=32, temperature=0.8))
```

Cost and residency audit (invariant I2):

```python
report = bhanox.audit_bytes_per_token(model)
print(bhanox.report(model))
```

Memory budgets (spec D7) — cap either memory in bytes you choose:

```python
cfg = bhanox.load_config("mini", temp_mem="64MB", perm_mem="2GB")
model = bhanox.Bhanox(cfg)
model.set_perm_budget("4GB")        # live resize; the vault admits or stops
print(bhanox.memory_report(model))
```

Verify the invariants and the code doctrine:

```bash
pytest                                  # full suite
pytest tests/test_invariants.py -v      # I1-I4, the release blockers
pytest tests/test_codebase_limits.py    # size doctrine (law C5)
python scripts/benchmark.py --config nano --tokens 512
python examples/generate_nano.py        # public API smoke test
python examples/audit_demo.py           # I2/I3 for every preset
```

---

## What it is

Five components, each independently verifiable:

| Component | One-line job |
|---|---|
| **HashBind** | Embeds *any* byte 4-gram. No vocabulary, no `<UNK>`, ~96% less embedding memory. |
| **DeltaBank** | Multi-head, multi-timescale recurrent memory with a delta-rule (error-correcting) write. O(1) per token, no growing KV cache. |
| **MicroExpert** | Fine-grained mixture of experts: 16–128 routed experts plus shared, top-2 routing. Active weights fit in L1. |
| **PulseGate** | Skips any channel whose input did not meaningfully change (hysteresis + learned thresholds). |
| **VectorVault** | Optional HDC episodic store (bipolar hypervectors, XOR + popcount) for exact long-range recall. |

The design in one paragraph: replace attention's ever-growing KV cache with a
**fixed-size** state that is *corrected* on write rather than appended to. The
delta rule means a write does not add a second, slightly different copy of a
fact — it repairs the existing one. Multi-timescale decay banks then give that
state a short memory and a long memory at once, which is what lets a
constant-size state behave like it remembers.

---

## Current measured status

Labels are the project's honesty law: `measured` (a script in this repo produced
it) or `modeled` (a component model, not an end-to-end measurement).

| Claim | Value | Label | Produced by |
|---|---|---|---|
| Generation cost per token | constant, O(1) in context | **measured** | `test_state_stays_bounded_while_generating` |
| Inference arithmetic | integer only, no float ops | **measured** | `tests/test_invariants.py` (I3) |
| State size | fixed, independent of context | **measured** | `test_state_is_constant_over_a_long_context` |
| Packed params (nano) | 2.11 MB (2,108,996 values) | **measured** | `bhanox.report(model)` |
| Packed params (mini / small) | 24.35 MB / 324.95 MB | **measured** | `bhanox.report(model)` |
| Delta rule vs additive writes (recall, untrained) | 0.213 vs 0.375 | **measured** | `test_delta_rule_beats_additive_writes` |
| Decay banks vs single decay (state energy) | 5,430 vs 2,456 (2.2x) | **measured** | `test_decay_banks_outlast_a_single_fast_decay` |
| PulseGate compute skipped, steady input | 75% | **measured** | `test_steady_input_is_mostly_skipped` |
| PulseGate compute skipped, random input | 0% | **measured** | `test_noisy_channel_never_sleeps` |
| MicroExpert bytes/token vs dense | 5.7x fewer | **measured** | `test_touches_far_fewer_bytes_than_dense` |
| Batch-1 tokens/sec, NumPy reference | ~51 t/s (32 tokens, greedy) | **measured** | `scripts/benchmark.py` |
| P1 speed gate (>= 20,000 t/s) | **NOT MEASURED** — native runtime is M4 | — | — |
| Quality (perplexity / BPC) | **NOT MEASURED** — no model trained yet | — | — |
| Energy vs quality-matched Transformer | **modeled >= 95% target** | modeled | `docs/benchmarks.md` |

The energy claim ships as `modeled` and stays that way until a RAPL
measurement exists on bare metal. Component mechanisms are individually
verified; the end-to-end number is a compound of those models, not a
measurement. See `docs/benchmarks.md` for the full table and the caveats.

The reference speed figure is *not* the P1 claim. The reference pays for
float32 temporaries and a Python-level loop per token per layer; the M4 native
runtime does int8 work straight out of a fixed buffer. Expect one to two orders
of magnitude between them, which is exactly why P1 cannot be settled here.

---

## The four invariants

These are release blockers, enforced in CI, not aspirations in prose.

| | Invariant | Enforced by |
|---|---|---|
| **I1** | Head-load: `concurrent_writes < d_k` | `BhanoxConfig.check_head_load` |
| **I2** | Residency: bytes touched per token per layer <= L2 | `bhanox.audit_bytes_per_token` |
| **I3** | Op whitelist: no float, no int multiply wider than 8 bits | `bhanox.ir.verifier` |
| **I4** | Numerical: backends agree with the reference to < 1e-3 | `tests/test_invariants.py` — **skipped until M4** |

I2 currently **fails on the `small` preset** at 3,538,944 B per layer against a
1 MiB budget (337.5%). The cause is the DeltaBank's projections at 3,014,656 B,
not the experts: the mixer touches 524,288 B, 5.7x less than its dense
equivalent. That is reported honestly rather than tuned away; see ADR-003.

I4 has no native runtime to compare against until M4, so the test exists and is
skipped with that reason rather than being quietly omitted.

---

## Repository layout

```
src/bhanox/
  config.py          shapes, presets, invariant I1
  model.py           the assembled Bhanox network
  frontend/          HashBind  (open-vocabulary embeddings)
  core/              DeltaBank (the recurrent memory)
  mixer/             MicroExpert (top-2 sparse FFN)
  governor/          PulseGate (event-driven compute)
  memory/            VectorVault (episodic HDC store)
  quant/             int8 + ternary numerics
  ir/                BIR op whitelist and graph verifier (invariant I3)
  audit.py           bytes-per-token residency audit (invariant I2)
  generate.py        O(1) generation loop
  train/             trainer + PyTorch mirror (M2)
tests/               mirrors src/ exactly
docs/                architecture, decisions, benchmarks, backlog
```

Runtime dependencies: **numpy only.** The import-and-generate path works with
nothing else installed. PyTorch appears only in `bhanox/train/` for Kaggle GPU
training, as an optional extra.

---

## Contributing

Read `CONTRIBUTING.md`. The short version: Black formatting, ruff, mypy, and
`pytest` green before any commit. One concern per commit. No new runtime
dependency without an ADR.

---

## Origin

Bhanox is a personal project by **Saurav Bhandari** (Pokhara, Gandaki Pradesh,
Nepal), built in the open and released under Apache-2.0. The name is Nepali —
*bhanox* is what the mountain calls you when the air is thin and the work is
worth it. The codebase carries no Nepali component names; the heritage lives
here and in the release codenames.

## License

Apache-2.0. See `LICENSE`.
