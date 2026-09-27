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

**Status: in progress.** The checkpoint format is done. The trainer and the first
real weights are not.

The trainer, the checkpoint format, and the first real weights. This is where
perplexity becomes a number instead of a `NOT MEASURED`.

**Done ahead of M2 — the checkpoint format.** `bhanox.checkpoint.save()` /
`load()`, plus `bhanox.from_pretrained()`, which was declared as a frozen API
hook and raised until this landed. Versioned, atomic, NumPy-only, and
bit-exact: 124 tensors round-trip with `max|diff| == 0.0` and the restored model
produces identical logits. Three decisions worth knowing:

- The tensor list is **discovered by walking the model**, not hand-written, so a
  new parameter cannot be silently left out of a checkpoint. A hand-maintained
  list is where that bug hides.
- Writes go to a sibling temp file, fsync, then `os.replace`. A crash mid-save
  leaves the *previous* checkpoint intact, which is tested by making the write
  fail.
- Only parameters are saved, not recurrent state. That is what keeps a
  checkpoint proportional to parameters rather than context length, and it
  stops a file's contents from depending on the batch size of whichever run
  wrote it. Because `forward` continues from existing state rather than
  resetting, a caller that carries the state separately still gets a bit-exact
  mid-sequence resume; both directions are pinned in `tests/test_checkpoint.py`.

**Done ahead of M2 — per-sample state.** The question M1 left open is closed.
The recurrent state carries a sample axis, so a batched loss means what it
says. Batch-1 behaviour is unchanged, measured rather than assumed: same int32
state, same gate decisions, float32 logits bit-identical. See *Batch semantics*
in `docs/architecture.md`.

**Done ahead of M2 — spec D7, memory budgets.** Both memories are now explicit,
bounded, user-settable byte ceilings: `load_config(..., temp_mem=, perm_mem=)`,
`model.set_perm_budget()`, `bhanox.memory_report()`, and `set_entry_cap()` for
the count knob. It went in early because the trainer needs a memory budget to
behave against, and because the spec's sizing formulas were wrong in a way worth
settling against real arrays: they undercounted the built layout by 2x on the
temporary state and 14x on the vault, so implementing them literally would have
violated D7's own "never OOM" rule on the first entry. See
`docs/architecture.md` for the corrected formulas and what each knob is tied to.

**Settled: how training interacts with the int8 grid.** This had to be decided
before the update rule, because it decides what the gradient even *is*.

**Straight-through estimator, on a schedule.** The forward pass uses the
quantized weights and the gradient passes through the quantizer as if it were
identity, so what is optimized is what ships.

That was not an open choice so much as a re-reading. The numerics layer already
implements it — `ste_round`, `ste_quantize`, `ternary_quantize` in
`src/bhanox/quant/numerics.py` — and `ternary_quantize` carries the reason: the
sign function has zero derivative almost everywhere, so without the STE that
regime has no gradient at all. The reference also already stores *dequantized*
float32 weights and applies the int8 regime through an explicit `quantize()`
call, so "on-grid" is the model's native representation rather than an extra
approximation laid on top of it. Training in float and quantizing at save would
mean optimizing a function the model never ships.

The schedule is float warmup first, then quantized STE for the remainder. Pure
STE from step 0 is the purest option and was the runner-up, but its gradient is
a biased surrogate of a hard nonlinearity, and early steps tend to barely move.
The cost of the schedule is that the loss reported during warmup is not the loss
that ships, so it is labelled as such rather than being quietly charted beside
post-quantization loss.

One implementation trap, recorded because it fails silently: the NumPy form
`x + (rint(x) - x)` has no graph to preserve, but the same expression in Torch
has gradient **0**, not 1, and certainly not the **2** an earlier draft of this
file claimed. `round` has no derivative, so the two terms cancel and the STE
becomes a no-op that raises nothing. The mirror needs
`x + (round(x) - x).detach()`. `d(ste_round)/dx == 1` is a claim, so it is
checked by `src/bhanox/train/gradcheck.py` rather than asserted in a comment.

A second trap, found by finite-differencing the mirror and much harder to see:
the activation quantiser must be a *single* straight-through, not
`ste_requantise(ste_round(x * 127))`. That composition yields `d k_f / dk == 127`,
because the round passes the multiply's 127 through and the requantise adds
another 1. It is wrong because in the float recurrence the quantiser has already
been rounded away, so `k_f` stands in for `k` and its derivative is 1. A 127x
error on every activation gradient rescales `W_k`, `W_q` and `W_v` identically,
so gradient directions stay plausible and the loss still falls — the only symptom
is an effective learning rate nobody chose. It now has its own test.

**Deferred to M4: the D7 CLI.** The spec's example is`bhanox serve mini.nx --temp-mem 64MB --perm-mem 2GB --perm-storage disk`, and
none of it exists — the repo has no entry point and no `serve` command. A serve
loop and a disk-overflow store are runtime concerns, and M4 is where a runtime
gets built, so the CLI belongs there rather than growing a second surface now
and rewriting it later. D7 ships as a library API, which is complete and tested.

**Found while mirroring: `PulseGate.salience` is dead weight.** The gate
allocates, counts and checkpoints a per-channel `salience` array, and never reads
it. `step()` takes a `gate_magnitude` argument, ranks *that* in `_protected`, and
ignores `self.salience` entirely — and both call sites in `model.py` pass
`np.abs(step_out)`. So the array cannot change any output and can never receive a
gradient, while the optimizer would dutifully update all of it forever. At nano
that is 512 of the gate's 1,536 counted values, 0.024% of the model's 2,108,996
parameters.

The mirror reproduces the bug rather than fixing it, which is the only defensible
option here. Reading `salience` in `_protected` is an architecture change to a
frozen spec; it would change which channels are protected, and it would silently
invalidate the measured 0.54 skip rate at 0.34% error. So the bug is pinned by
`test_salience_is_inert` (changing it must not change any output, and it must
never receive a gradient), the mirror's `repr` says `salience_inert=True`, and the
trainer will exclude it from AdamW. If that test ever fails, the ROADMAP, the byte
accounting and the measured claims all have to be revisited together.

**Done ahead of M2 — the PulseGate mirror.** `src/bhanox/train/governor_mirror.py`,
agreed against the numpy reference bit-exactly on the mask *and* on every piece
of integer state — the cache, the awake mask, the quiet counter, `_has_run`, and
the four event counters. That is a stronger claim than the DeltaBank and
MicroExpert mirrors make, and it is the only one available: a boolean state
machine has no float to be within a tolerance of, so "bit-exact" and "a plausible
skip rate" are otherwise indistinguishable.

Writing it turned up two bugs in my own first draft that the aggregate skip rate
hid completely, which is why they are worth writing down:

- The quiet counter is `np.where(quiet, count + 1, 0)` — a loud step **resets**
  it. Read as `count + quiet` ("increment when quiet"), a loud step instead
  *leaves the count alone*, so a channel that slept twice, took one big kick, and
  should now be wide awake still reads 2 and falls straight back to sleep. It then
  skips every other step forever while the overall skip rate looks perfectly
  plausible throughout. 124 of 384 channels diverged at step 2.
- `protected` belongs in the wake condition, not only in the sleep condition: the
  reference wakes on `(delta > tau_hi) | protected`. A mirror that omits it looks
  correct for the loud channels — which is to say for most protected channels,
  since the largest gate magnitudes usually also move the most — and quietly
  strands the small protected ones. It surfaced as 9 of 384 channels, one step
  later than the first bug and only because the first one was fixed.

The third trap is a gradient one, and it is why the counter is rebuilt rather than
detached. `_quiet` is integer state, so detaching it before comparing against
`sleep_after` gives a correct forward and a **permanently zero `tau_lo` gradient**
— a threshold that looks trained and never moves. The counter is therefore rebuilt
as a hard value and re-anchored (`q_value + (ste - ste.detach())`), so the forward
is the reference's and the gradient to `tau_lo` survives. Both thresholds now get
non-zero gradient, checked over a sequence because the first step legitimately has
none: `compute = np.where(has_run, awake, True)` makes the mask the constant 1.0
until a sample has run once.

`ste_gt` and `ste_ge` are kept separate because the reference uses both and they
are not interchangeable at the boundary: the gate wakes on `delta > tau_hi` but
sleeps on `_quiet >= sleep_after`. Collapsing them to one strict comparison makes
the gate sleep a step late, and the error is invisible in aggregate — it just
looks like a slightly different skip rate.

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
