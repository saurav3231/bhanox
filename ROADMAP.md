# Roadmap

Six milestones. The architecture is frozen; what changes is how much of it
actually runs, and how much of the story is measured rather than modelled.

Status legend: **done** / **in progress** / **not started**.

Shipped as `v0.1.0`.

---

## M0 — Repository skeleton

**Status: done.**

Licence, `pyproject.toml`, CI, the directory layout, the law references, and a
README that says which numbers are real. Nothing in `src/` yet beyond the
packaging.

## M1 — Reference implementation

**Status: done.** Shipped in `v0.1.0` together with M0: the skeleton commit
cannot be green on its own, because `pip install -e .` needs the package that
M1 is, so there was only ever one shippable state.

Every component of the frozen architecture, in NumPy, with a test per
mechanism, and the I1–I3 gates enforced.

- HashBind, DeltaBank, MicroExpert, PulseGate, VectorVault
- the BIR op whitelist and graph verifier (I3)
- the bytes-per-token residency audit (I2)
- the assembled model, the O(1) generation loop, the public API
- the discipline gates: module size, numpy-only runtime, no undocumented public
  function, required files present

Exit criterion: `pytest` green, and every number in the README traceable to a
test.

## M2 — Training

**Status: not started.**

The trainer, the checkpoint format, and the first real weights. This is where
perplexity becomes a number instead of a `NOT MEASURED`.

It also resolves the open question left by M1: the recurrent state is currently a
single instance, so `forward(B, T)` is one concatenated stream. Per-sample state
is required before a batched loss means anything.

**Done ahead of M2 — spec D7, memory budgets.** Both memories are now explicit,
bounded, user-settable byte ceilings: `load_config(..., temp_mem=, perm_mem=)`,
`model.set_perm_budget()`, `bhanox.memory_report()`, and `set_entry_cap()` for
the count knob. It went in early because the trainer needs a memory budget to
behave against, and because the spec's sizing formulas were wrong in a way worth
settling against real arrays: they undercounted the built layout by 2x on the
temporary state and 14x on the vault, so implementing them literally would have
violated D7's own "never OOM" rule on the first entry. See
`docs/architecture.md` for the corrected formulas and what each knob is tied to.

**Deferred to M4: the D7 CLI.** The spec's example is
`bhanox serve mini.nx --temp-mem 64MB --perm-mem 2GB --perm-storage disk`, and
none of it exists — the repo has no entry point and no `serve` command. A serve
loop and a disk-overflow store are runtime concerns, and M4 is where a runtime
gets built, so the CLI belongs there rather than growing a second surface now
and rewriting it later. D7 ships as a library API, which is complete and tested.

## M3 — Model zoo, on Kaggle free tier

**Status: not started.**

Every heavy job runs on Kaggle through the hand-off loop in `CONTRIBUTING.md`.
Nothing in this milestone runs on a local machine, and Saurav is never asked to
run anything locally either.

Write `kaggle/run_mini_train.py` and `kaggle/bootstrap.py`: Mini on
TinyStories-class data, GPU-T4 track, budgeted per the timing table. Optional CPU
thesis track: `kaggle/run_nano_cpu.py`, the full Nano lifecycle on Kaggle CPU
only.

Baseline runners on identical data, splits and seeds: an equal-parameter
Transformer, an equal-parameter GRU, and a 4x-parameter Transformer.

Every run is handed over as the one-cell bootstrap snippet, with the accelerator
and expected wall-time stated. Results Saurav returns are committed to
`docs/benchmarks.md` and `results/` the same day, labelled `measured`, whatever
they say — including a result where Bhanox loses.

Planning numbers, from measured hardware rates (T4 ~3 TFLOPS effective, Kaggle
CPU 4 cores ~30-50 GFLOPS effective, cost ~6 x params x tokens per step):

| job | FLOPs/step | T4 | CPU (4 cores) |
|---|---|---|---|
| Nano 1.1M, batch 32x256 | ~54 GFLOPs | ~10-30 s / 1000 steps | ~15-30 min / 1000 steps |
| Mini 13M, batch 64x512 | ~2.6 TFLOPs | ~15 min / 1000 steps, ~7-8 h full | ~18 h / 1000 steps — do not |
| 20M Transformer baseline | ~123 GFLOPs (batch 8x128) | <1 min / 1000 steps | ~40-70 min / 1000 steps |

So Mini and all baselines go on the GPU-T4 track. Quota is ~30 GPU-hours a week
and sessions die without warning, which is why every runner checkpoints and
resumes. Corpus goes up once as a private Kaggle Dataset, with a download
fallback for sessions that have network.

`from_pretrained` and reproducible evaluation are also part of this milestone;
the function is declared and raises today so callers can be written against the
final API shape.

## M4 — Native runtime

**Status: not started, needs explicit approval.**

The C or Rust inner loop doing int8 work out of a fixed buffer. This is the
milestone that settles the two open claims:

- **P1**, `>= 20,000 tokens/s`
- **I4**, agreement with the reference to `< 1e-3`

Expect one to two orders of magnitude over the NumPy reference. This is also
where the VectorVault should stop caching a 16 MB random projection at d=512 and
use a counter-based hash instead.

**This milestone is not authorised to begin without explicit approval.**

## M5 — Energy measurement

**Status: not started, needs explicit approval.**

RAPL on bare metal, a quality-matched baseline, and turning the `modeled` energy
claim into a `measured` one. Until this exists the project publishes no
end-to-end energy number.

**This milestone is not authorised to begin without explicit approval.**

---

## Beyond

Sequences: long-context recall beyond what the state can hold, and the
VectorVault as the answer. Then whatever the measurements say, including the
possibility that some of this does not work.
