# Kaggle CUDA smoke: setup and run

A private, manual **correctness-only** check that runs Bhanox's real trainer on a
Kaggle GPU, on synthetic data. Nothing is published.

**Status.** The short Gate A smoke has been run on a Kaggle GPU and **passed**:
a real T4 completed forward, backward, and AdamW for two short-context steps
(context 32, batch 1) with finite losses. That verified result is at **context
32 only** and says nothing about any other context.

There is now a second, **opt-in** probe for the full planned context of 4,096.
It has **not** been run. There is no result to quote for it until you run it.

## What this is for

Answering one question: **does the existing `bhanox.train` trainer run on a
Kaggle GPU?** It is a correctness-only check on synthetic data, not a benchmark,
not a training run, and not a speed test.

It makes **no GPU speed claim and no energy claim**. In particular, a Kaggle GPU
is not a way to learn anything about the product: the stated goal is
**CPU-native** operation, so GPU throughput is not evidence of fast or
low-energy CPU inference, in either direction.

## Which gate is this?

This README covers **Gate A** only. There are two distinct activities with
different requirements. Do not let Gate B's requirements block Gate A.

**Gate A -- short private synthetic CUDA correctness smoke.** Requires a Kaggle
account, a private notebook, **Internet enabled**, and a GPU that is actually
attached. It uses synthetic bytes generated in the notebook, runs a couple of
steps under a wall-clock cap, and is **not resumable and not a long run** -- so
it does **not** require a working checkpoint/resume path and does **not**
require an approved real corpus. If CUDA is absent it stops immediately with no
CPU fallback.

**Gate B -- long or real-data training.** A separate activity with a higher bar,
none of which this notebook satisfies or pre-authorizes:

- a checkpoint/resume path covering model, optimizer, step/scheduler state, and
  data position, as applicable, and **tested** -- a long run that cannot resume
  cannot survive a session ending;
- prior approval of an **English-only** corpus and of its **license and terms**;
- realistic-context throughput and memory measurements, not a small smoke;
- no unsupported quality or energy claims.

## Two ways to run it

Both call the same `bhanox.train.smoke.cuda_smoke` in the cloned repository, so
they cannot drift apart. Pick one.

### A. Paste one cell (recommended)

Create a new Kaggle notebook, set a GPU, and paste the single cell from the
handoff. It clones the public repo, checks CUDA, runs the smoke, and prints
everything. This is the shortest path and needs no dataset at all.

### B. The committed notebook

`bhanox_kaggle_smoke.ipynb` in this directory does the same thing cell by cell,
with the honest limits in markdown. To use it, upload it as a private Kaggle
notebook, or open it in the Kaggle editor and paste the cells.

**Neither path needs a private source dataset.** An earlier version of this file
told you to zip `src/` and upload it as a private Kaggle dataset. That step is
gone: the source now comes from a **public `git clone`**, which is simpler and
keeps the notebook pointed at one specific commit.

## The two runs: quick smoke, and the expensive opt-in probe

Both call into the same cloned repository, and both use synthetic in-memory bytes.
Neither needs a corpus, an approved licence, or a checkpoint/resume path, because
neither is a long run and neither reads anything from disk. They differ only in
what they ask and what they cost.

| | **quick smoke (default)** | **T=4096 probe (opt-in)** |
|---|---|---|
| entry point | `cuda_smoke` | `t4096_probe` |
| context | 32 (configurable) | **4,096**, fixed per call |
| updates | 2 | **exactly 1** |
| wall-clock | bounded by an explicit cap | **no cap; runtime unknown** |
| data | repeating 5-byte cycle (learnable) | deterministic varied bytes (noise) |
| answers | does training run on this GPU? | does the full window fit and run? |
| cost | seconds to a couple of minutes | **unknown; may be substantial** |

The quick smoke is the default and stays that way. The probe is gated behind a
flag you have to set yourself (`RUN_T4096_PROBE = True` in the handoff cell, or
`--t4096-probe` on the CLI). **It does not run unless you flip that flag.**

### About the probe's cost, stated plainly

The probe is **not cheap** and this file will not pretend otherwise. The verified
T=32 run above took about 34 seconds for *two* steps; the probe is a single step
at 128x that context. **No runtime prediction is offered here, and none should
be inferred** -- a linear extrapolation from the T=32 number is not a runtime,
because the recurrence's per-token host/device synchronisation does not scale
linearly with anything.

The probe has **no timeout, by design**. A wall-clock cap cannot interrupt a
blocked `train_chunk`; it would only be checked after the update returned, by
which point the GPU has already done the work. A timeout that cannot fire is
worse than none, because it reads like a safety rail while providing none. So the
function prints a loud warning before it starts, and **the caller is expected to
watch the cell and interrupt manually**.

On failure -- an out-of-memory error, or any runtime error -- the probe reports
the exact failure, the elapsed time, and the peak memory reached, then re-raises.
It does **not** shrink the context, retry, or switch devices. An OOM at full
context *is* the measurement; papering over it would destroy the only thing the
probe exists to find out.

### What a probe pass does and does not prove

Proves, and only this: **one synthetic T=4096 update completed on that device.**
It fit in memory, the full context ran, and the loss, gradients, and post-update
parameters were finite.

Does **not** prove: sustained training; throughput; learning on a real corpus;
language quality; energy use; or that checkpointing and resume would work in a
long run. One update cannot demonstrate any of those, and none of them may be
inferred from a pass.

The probe's loss is labelled a **meaningless diagnostic** in its own output. The
bytes are uniform random, so the next byte is genuinely unpredictable and the loss
sits near `ln(256) = 5.545`. **A value at or above that floor is the expected
result, not a failure** -- unlike the quick smoke, a low loss here would mean
nothing either.

**The T=4,096 probe has not been run and has not passed.** Until you run it and
share the output, the full-context claim is unverified.

## Setup

1. **Create a private notebook.** `Code -> New Notebook`. Do not add a title --
   a titled notebook can become public. Leave it untitled and private.
2. **Turn Internet ON.** This is the one thing the run downloads: the public
   Bhanox source, by public `git clone`. No token, no credential, no Kaggle API.
   It does **not** download a corpus, a model, or a package -- `torch` comes from
   Kaggle's own preinstalled environment, and nothing is pip-installed.
3. **Select a GPU.** Open the right-hand **Settings** panel and set
   **Accelerator** to a GPU (usually `T4 x2` or `P100 x1`). This is the step
   people miss; without it the run fails immediately, on purpose.
4. Run the cell top to bottom.

## What a healthy run prints

The smoke prints the resolved commit, the environment, the run configuration,
a per-step table, and the observations. What to look for:

```
all losses finite      True
completed all steps    True
argmax accuracy        <rising> ... (chance floor 0.0039)
peak CUDA memory       <MiB> MiB (this context only)
```

`all losses finite True` and `completed all steps True` is the pass condition.
Accuracy above the `1/256` chance floor is the signal that the gradient path
reaches the weights.

Note the `(this context only)` on the memory line. It is load-bearing: a peak
measured at context 32 is a floor for context 4,096, not an estimate of it, and
the probe exists precisely because that number was unknown.

If the run reports more than one visible GPU, note that it used **exactly one**,
`cuda:0`. There is no DataParallel and no DDP.

## What a pass does and does not prove

Proves: the mirror builds, moves to the device as one unit, and runs forward,
backward and an AdamW step there; the loss is finite; the loss falls on data
with real structure; peak memory at this context is measurable.

Does **not** prove: that Bhanox learns anything about language; that it is fast
on a GPU; that it is energy-efficient; that full-context memory fits; that a
checkpoint can be saved or resumed; or anything about real text. The data is a
repeating byte cycle generated in memory -- a gradient-path check, nothing more.

## Why the training path is expensive

Stated as **source facts, not predictions**, and with no claim about any
particular device:

- `BhanoxMirror.step` advances the state with a **sequential Python loop over
  every token position**. The recurrence is genuinely sequential in the token
  index, so there is no parallelism across the context to exploit.
- The reported 4-layer call path performs **roughly 48 host/device
  synchronization points per token** -- host round-trips in `_write_int`, scalar
  `int()`/`float()` reads in `PulseGateMirror.step`, and an entropy read in
  `MicroExpertLayerMirror.forward`. On CPU these are cheap no-op views; on a
  CUDA device each is a stall. **The exact count and the end-to-end cost depend
  on the actual device path and have not been measured.**
- `int8_codes` and the per-token unembed matmul are **float64 because fidelity
  to the numpy reference requires it**, not by accident. **FP64 is not
  established as the dominant cost**, and a local CPU profile showed it is not
  the dominant cost on CPU.

### Local CPU diagnostic (a different machine, CPU only)

A bounded profile of one training step at `nano`, batch 1, `T=256`, on synthetic
in-memory bytes: forward + backward + AdamW, no data loading, no profiler, 1
warm-up then 3 timed repeats.

| quantity | value |
|---|---|
| median step | 46.811 s |
| spread (3 repeats) | 44.891 - 59.402 s (31% of median) |
| per byte-position | 182.9 ms |
| throughput | 5.47 positions/s |

Host: Intel Core i5-3337U @ 1.80 GHz, 2 cores / 4 threads, Windows 11,
`torch 2.13.0+cpu`, `numpy 2.5.3`, 2 torch threads. This is a slow 2013 mobile
CPU and the 31% spread reflects that.

Phase breakdown, from a separate instrumented run at `T=64`. **Instrumentation
overhead is included**, so read the percentages as phase indications rather than
precise benchmark numbers:

| phase | share |
|---|---|
| forward total | 66.6% |
| &nbsp;&nbsp;DeltaBank recurrence | 60.0% |
| &nbsp;&nbsp;PulseGate | 5.3% |
| &nbsp;&nbsp;mixer (MoE) | 0.3% |
| &nbsp;&nbsp;embed / unembed | ~0.0% |
| backward | 32.1% |
| optimizer step | 1.4% |

Two things follow, and one does not:

- On **CPU**, the **DeltaBank recurrence is about 60% of a training step** and
  the mixture-of-experts mixer is negligible despite being the most prominent
  part of the architecture by name.
- The ~48 synchronization points do **not** explain the CPU number: on CPU there
  is no device transfer at all. They remain a separate, unmeasured CUDA-path
  question.
- Nothing here predicts a GPU number, and nothing here measures energy. A CPU
  profile cannot predict a GPU profile.

An earlier version of this file quoted "about 185 ms per token" as if it were a
result. It is not: it was a **single, un-warmed, CPU-only probe at `T=64`,
batch 1** (forward + backward + AdamW, no data loading) with **no repeats, no
variance, and no phase breakdown** -- not a steady-state benchmark. The
256-position median above supersedes it. **A 4096-position chunk has never been
measured**; any figure for it is an extrapolation, not a runtime. The opt-in
probe is the first thing that can actually measure it.

## Regenerating the notebook

`bhanox_kaggle_smoke.ipynb` is generated, not hand-edited:

```
python kaggle/build_notebook.py
```

Edit `kaggle/build_notebook.py` and re-run it. The notebook is a thin wrapper;
all of the training logic lives in `bhanox.train.smoke.cuda_smoke`, which is
covered by `tests/test_trainer.py`.

**The GPU path has been executed for the short smoke**, and passed, at context
32 on a real T4. Everything wider than that -- the full 4,096 context, throughput,
sustained training -- is still unproven, and the opt-in probe exists to test one
step of it and nothing more. Neither smoke makes a GPU speed claim or an energy
claim, and a Kaggle GPU is not a way to learn anything about the product: the
stated goal is **CPU-native** operation, so GPU results say nothing about fast or
low-energy CPU inference, in either direction.
