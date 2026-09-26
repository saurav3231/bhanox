# Roadmap

Six milestones. The architecture is frozen; what changes is how much of it
actually runs, and how much of the story is measured rather than modelled.

Status legend: **done** / **in progress** / **not started**.

---

## M0 — Repository skeleton

**Status: done.**

Licence, `pyproject.toml`, CI, the directory layout, the law references, and a
README that says which numbers are real. Nothing in `src/` yet beyond the
packaging.

## M1 — Reference implementation

**Status: in progress.**

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

## M3 — Model zoo

**Status: not started.**

Published checkpoints, `from_pretrained`, and reproducible evaluation. The
function is declared and raises today so callers can be written against the final
API shape.

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
