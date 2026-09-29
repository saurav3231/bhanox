"""Generate kaggle/bhanox_kaggle_smoke.ipynb.

The notebook is committed as ``.ipynb`` (Kaggle needs that format) and is
generated from here so the JSON is not hand-edited. Run from the repo root:

    python kaggle/build_notebook.py

It is a thin wrapper. The training path, the synthetic bytes, the assertions and
every number all come from :func:`bhanox.train.smoke.cuda_smoke` in the cloned
repository -- this file contributes setup, the environment gate, and the honest
limits. An earlier draft re-implemented the corpus and the training loop inline;
that was a second copy of the same code that could drift from the real one, so
it was deleted rather than maintained.

**This notebook has never been run on a Kaggle GPU.** Its code cells are
exercised locally, and the clone step is the only thing that has never run at
all. There is no result to quote until someone runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The public branch to clone. This notebook lives in that same branch, so the
#: clone resolves to the code you are reading. The resolved commit is printed at
#: runtime so the run can be tied to an exact SHA.
BRANCH = "kaggle-cuda-smoke"
REPO = "https://github.com/saurav3231/bhanox.git"
WORKDIR = "/kaggle/working"

#: Bounded on purpose. The smallest context and step count that exercise forward,
#: backward and an optimizer update, with a wall-clock cap in case the device is
#: slower than expected. Not a benchmark, so a small number is the honest choice.
MAX_EXAMPLES = 32
STEPS = 2
TIME_CAP_S = 240

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------
md(r"""
# Bhanox -- private CUDA correctness smoke

**What this is.** A *correctness-only* check on **synthetic data**: does the real
`bhanox.train` trainer run on a Kaggle GPU, unmodified in its training logic?

The training code, the synthetic bytes, and the checks are not in this notebook.
They come from `bhanox.train.smoke.cuda_smoke` in the cloned repository, so the
notebook cannot drift from the code it is testing. There is no reimplementation
here.

**What it is not.** Not a training run and not a benchmark. It makes **no GPU
speed claim and no energy claim**, and it is not evidence of language quality --
see "Honest limits" at the bottom.

**Status: the Kaggle CUDA path has not been run.** As of writing this notebook
has never executed on a Kaggle GPU. The clone step below has not run either.
Until you run it, there is no result to quote.

**No dataset required.** Nothing is uploaded, nothing is attached, and there is
no Kaggle dataset step. An earlier version of this notebook told you to zip
`src/` and attach it as a private dataset; that is gone. The source arrives by
public `git clone` below, pinned to one branch, and the resolved commit is
printed so the run can be tied to an exact SHA.

**Keep it private.** Do not add a title, which can make a notebook public. This
is a smoke test for you, not a demo.
""")

# --------------------------------------------------------------------------
md(r"""
## 1. Clone the public repository

This needs **Internet ON** -- that is the one thing it downloads: the public
Bhanox source, by public `git clone`. There is no API token, no credential, and
no Kaggle API here. It clones *source code*; it does not download a corpus, a
model, or a package. `torch` comes from Kaggle's own preinstalled environment.
""")

code(f"""
import os
import subprocess
import sys

REPO = {REPO!r}
BRANCH = {BRANCH!r}
WORKDIR = {WORKDIR!r}

os.makedirs(WORKDIR, exist_ok=True)
dest = os.path.join(WORKDIR, "bhanox")

if os.path.isdir(os.path.join(dest, ".git")):
    print(f"reusing existing clone at {{dest}}")
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, dest],
        check=True,
    )

# The exact commit this run is testing, resolved at runtime rather than trusted.
commit = subprocess.run(
    ["git", "-C", dest, "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
print(f"cloned  {{REPO}}")
print(f"branch  {{BRANCH}}")
print(f"commit  {{commit}}")
""")

# --------------------------------------------------------------------------
md(r"""
## 2. Put the source on the import path

`bhanox` is numpy-only by design, so importing `bhanox.train` is what pulls in
torch. Nothing is pip-installed: installation would need the internet, and
there is no wheel to install.
""")

code(r"""
SRC = os.path.join(WORKDIR, "bhanox", "src")
if not os.path.isdir(os.path.join(SRC, "bhanox")):
    raise FileNotFoundError(f"expected the bhanox package under {SRC}")
sys.path.insert(0, SRC)

import bhanox

print(f"bhanox    {bhanox.__file__}")
print(f"version   {getattr(bhanox, '__version__', '(none declared)')}")
""")

# --------------------------------------------------------------------------
md(r"""
## 3. The gate -- hard fail without a GPU

No CPU fallback, on purpose. A run that silently fell back to CPU would print a
green result for a question it never asked, and "it ran" would mean two
different things depending on the session.
""")

code(r"""
import torch

print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}")
print(f"CUDA build  {torch.version.cuda}")
print(f"available   {torch.cuda.is_available()}")
print(f"device count {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Open the right-hand Settings panel, set "
        "Accelerator to a GPU (T4 x2 or P100 x1 are the usual options), and "
        "re-run from this cell. There is no CPU fallback by design."
    )
if torch.cuda.device_count() < 1:
    raise RuntimeError("CUDA reports available but device_count() is 0.")

for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
""")

# --------------------------------------------------------------------------
md(r"""
## 4. Run the bounded smoke

`cuda_smoke` builds the mirror, moves it to the device, and runs the real
`run_documents` / AdamW / `bhanox.data` windowing over a deterministic in-memory
byte cycle. It requires the device explicitly, so it cannot fall back to CPU
even if this cell is edited.

It prints the resolved commit, the device, the configuration, per-step loss and
argmax accuracy, elapsed time, and peak CUDA memory for **this** context.
""")

code(f"""
from bhanox.train.smoke import cuda_smoke

result = cuda_smoke(
    device=torch.device("cuda:0"),
    config="nano",
    steps={STEPS},
    max_examples={MAX_EXAMPLES},
    time_cap_s={TIME_CAP_S},
)

print()
print("raw result:", result)
""")

# --------------------------------------------------------------------------
md(r"""
## 5. Result

Read the console output above. It is the whole result; there is nothing to infer
from this cell.

- `all losses finite True` and `completed all steps True` is a **pass** for a
  correctness smoke.
- Argmax accuracy above the `1/256` chance floor is the signal that the gradient
  path is connected to the weights.
- Elapsed time and peak memory describe **this run on this device at this
  context**. They are not a throughput, energy, or full-context figure, and they
  say nothing about how fast this would be on different hardware.
""")

code(r"""
ok = bool(result["all_finite"]) and bool(result["finished"])
print(f"pass: {ok}")
print()
if not ok:
    print("See the console output above for the reason. This cell only")
    print("summarises; the real diagnosis is in the per-step table.")
else:
    print("Forward, backward and an optimizer update all completed on")
    print(f"{result['device']} at context {result['max_examples']}.")
    print("That is the whole claim. It is not a speed result, an energy")
    print("result, a quality result, or a T=4096 result.")
""")

# --------------------------------------------------------------------------
md(r"""
## Honest limits

Read this before quoting any number from this run.

1. **This is a correctness smoke, not a training run.** A few steps on a
   repeating byte cycle establish that the gradient path is wired up. They
   establish nothing about whether Bhanox learns anything.

2. **The data is synthetic and is not language.** It is a repeating byte cycle
   generated in memory. A falling loss there is a statement about the
   optimizer and the backward pass, not about text. No real corpus is read,
   downloaded, or attached.

3. **No speed, energy, or quality claim is made here, and none should be read
   into the output.** The product goal is **CPU-native** operation, so Kaggle
   throughput is not evidence about the product in either direction. Energy is
   not measured anywhere in this notebook. FP64 appears in a few operations for
   fidelity to the numpy reference; that is a correctness reason, and it is
   **not** established as the dominant cost.

4. **The context under test is tiny** -- a few dozen positions, not
   `max_context` (4096). The peak-memory figure is for this context only. The
   trainer documents the retained shadow graph at roughly 32 KiB per token, so
   it scales with the number of positions held live: a 4096-position chunk would
   hold substantially more, and is **unmeasured**.

5. **Checkpointing the mirror is not implemented, and is not tested here.**
   `bhanox.checkpoint.save()` requires a numpy dataclass and raises
   `TypeError` on a `BhanoxMirror`, which is an `nn.Module`, and
   `bhanox.train.trainer` has no checkpoint or resume code at all. This is why
   the smoke is short: a two-step smoke needs no resume, whereas a long run
   without one cannot survive a session ending. The mirror does expose
   `state_dict()` (186 tensors at `nano`, verified to round-trip with
   `strict=True`), so a checkpoint milestone needs a mechanism written and a
   decision about what may be claimed when resuming.

6. **No Kaggle result exists yet.** This notebook has not been run on Kaggle
   hardware. Everything above describes what it *checks*, not what it found.
""")


def build() -> Path:
    cells = []
    for kind, text in CELLS:
        lines = text.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).with_name("bhanox_kaggle_smoke.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({len(CELLS)} cells)")
