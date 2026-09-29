# Kaggle CUDA smoke: setup and run

A private, manual **correctness-only** check that runs Bhanox's real trainer on a
Kaggle GPU, on synthetic data. Nothing is published.

**This has never been run on a Kaggle GPU.** There is no result to quote until
you run it.

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
measured**; any figure for it is an extrapolation, not a runtime.

## Regenerating the notebook

`bhanox_kaggle_smoke.ipynb` is generated, not hand-edited:

```
python kaggle/build_notebook.py
```

Edit `kaggle/build_notebook.py` and re-run it. The notebook is a thin wrapper;
all of the training logic lives in `bhanox.train.smoke.cuda_smoke`, which is
covered by `tests/test_trainer.py`.

**The GPU path has never been executed.** It is proven only by a real run on
Kaggle, and until that happens neither this file nor the notebook can claim
otherwise.
