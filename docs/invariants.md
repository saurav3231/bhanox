# Invariants

Four release blockers. Each is a claim about the architecture that a test
enforces. A claim nobody checks is a wish, so each section below names the
enforcing code and the current measured verdict.

The gate is `tests/test_invariants.py`.

---

## I1 — Head load

**`concurrent_writes < d_k`**, where
`concurrent_writes = write_rate * 1 / (1 - lambda_max)`.

Each layer writes one association per token. The shallowest decay bank is
`lambda = 0.5`, which halves its effective capacity. If more associations are in
flight than the state has key rows, the delta rule has nowhere to put them and
later writes evict the association currently being retrieved. This is a hardware
capacity limit, not a tuning knob.

Enforced by `BhanoxConfig.check_head_load`, called from `__post_init__`, so a
config that breaks I1 cannot be constructed at all.

The bank prior is frozen at `[1 - 2**-b for b in 1..n_banks]`: the first bank
halves, and each subsequent one forgets more slowly.

**Verdict: holds.** nano 8.0 < 16, mini 16.0 < d_k, small 32.0 < d_k, all with
at least 1.5x headroom so the check cannot pass by a single row.

---

## I2 — L2 residency

**`bytes_touched_per_token_per_layer <= l2_bytes`.**

Energy follows bytes moved, not FLOPs (Horowitz, ISSCC 2014), so bytes per token
is this project's primary cost metric and fitting in L2 is the primary
architectural constraint. Measured against a 1 MiB budget:

| preset | B / token / layer | of budget | verdict |
|---|---|---|---|
| `nano` | 110,592 | 10.5% | **PASS** |
| `mini` | 720,896 | 68.8% | **PASS** |
| `small` | 3,538,944 | 337.5% | **FAIL** |

`small` fails, and the cause is not the experts. Its DeltaBank projections alone
are 3,014,656 B per layer; the mixer touches 524,288 B, 5.7x less than the dense
equivalent. The sparsity is working, the projections are simply larger than a
1 MiB cache. Reported rather than tuned away (ADR-003).

The front-end is charged **once for the whole model** and reported separately
from `layer_total`, because charging it in both would inflate every per-layer
figure. That accounting bug is exactly what first made the nano and mini numbers
look better than they were.

Enforced by `bhanox.audit.audit_bytes_per_token`, and every reported figure is
counted from real arrays at call time.

---

## I3 — No float in inference

Every op in the deployed inference graph is on the BIR whitelist, and no integer
multiply may be wider than 8 bits.

The whitelist is 11 ops: `LOAD_I8`, `MUL_I8:8`, `ADD`, `SUB`, `SAT`, `SHIFT`,
`CMP`, `LUT`, `POPCOUNT`, `PERMUTE`, `STORE_I8`. No op begins with `F`.

The width limit lives on the graph, not on a bare op list, so a wide multiply is
caught by constructing a `Node` with `width_bits > 8`.

Enforced by `bhanox.ir.verifier`, and checked on **every** audit run — so a graph
that is not whitelist-clean fails before it is measured.

A note on what this does and does not mean: the reference stores dequantized
float32 weights and applies the int8 regime through an explicit `quantize()`
call. I3 is a property of the op sequence, not of the reference's storage dtype.
See `docs/architecture.md` for why running the reference on raw int8 codes
breaks the router.

---

## I4 — Backend agreement

**Every backend agrees with this reference to < 1e-3.**

**Verdict: not evaluable.** The native runtime lands in M4, so there is nothing
to compare against yet. The gate exists in `tests/test_invariants.py` and is
skipped with that reason, rather than being quietly omitted.

What *is* checked now is the precondition: the reference is deterministic, since
comparing against a non-reproducible baseline would be meaningless.

---

## What is not an invariant

**P1, `>= 20,000 tokens/s`,** is a performance target, not an invariant. It is
`NOT MEASURED` until the M4 native runtime exists. The NumPy reference measures
around 51 tokens/s for nano at batch 1; that number is a regression guard and
explicitly *not* the P1 claim. See `docs/benchmarks.md`.

**Quality** (perplexity, bits per character) is `NOT MEASURED` until a model is
trained in M2.
