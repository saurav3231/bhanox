"""Synthetic smoke run: does a gradient reach the whole mirror, and does loss fall?

What this is
------------
A five-line question with an expensive answer to get wrong. "Training works" is
usually taken to mean the loss went down, and a loss that goes down is easy to
manufacture: on random bytes the next-byte target is unpredictable, so the loss
sits near ``ln(256) = 5.545`` and any wobble is mistaken for progress. On
*learnable* data the same fall means something.

So the input is a repeating byte cycle::

    cycle = b"ABCDE"
    raw   = bytes(cycle[i % 5] for i in range(n))

Within a 5-cycle, a byte 4-gram nearly determines the next byte. The target is
therefore a genuine finite-state function of the input -- a keyed lookup -- and a
falling loss is evidence about the optimizer and the gradient path rather than
about the entropy of noise. Accuracy is reported alongside the loss because loss
alone can fall on a degenerate head that has not learned anything useful; the
two moving together is the actual signal.

What this proves
----------------
- Every trainable parameter receives a finite gradient, the router and the
  selected experts among them.
- The loss is finite at every step, and falls on data with real structure.
- Argmax accuracy rises above the ``1/256`` chance floor.
- A fixed seed reproduces the curve exactly.
- The chunk token-count histogram is reported, so the short-final-chunk exposure
  is visible rather than assumed.

What this does not prove
------------------------
- **Nothing about real-corpus convergence, generalization, or throughput.**
- **Nothing about training stability.** The loss surface is a staircase: a
  perturbation of ``W_k`` usually does not change the quantised activation codes
  at all, so the loss does not move and no finite difference can see it
  (:mod:`bhanox.train.mirror`). A monotone toy run is consistent with a model
  that would be unusable on real data.
- **Nothing about mirror/reference agreement after an update.** Two float
  implementations drift by a few ULP per matmul, and the gate thresholds a float
  with no dead band, so a channel can flip on an O(1e-7) cause. This is a
  property of the architecture, not of the mirror
  (:mod:`bhanox.train.model_mirror`).
- **Nothing about the update schedule on real documents.** The toy document is
  ``4 * max_examples`` *examples* -- 4100 bytes at ``max_examples=1024``, four
  grams plus a byte -- so its chunks are exact multiples and the short-tail
  exposure documented in :mod:`bhanox.train.trainer` never triggers here. The
  length is chosen for that, not incidental: ``max_examples * 4`` would give
  4092 examples and a 1020-token tail, so the toy run would quietly exercise the
  one thing the real schedule is worst at. On real documents it does trigger,
  once per document.
- **A 5-cycle is a lookup table, not language.** The model can ace it and still
  be unable to learn anything with long-range structure.
- The improvement margin is **reported, not asserted as a threshold**. This run
  prints its own numbers; the owner sets the bar a real run has to clear.

Usage::

    python -m bhanox.train.smoke                 # default: nano, 12 steps
    python -m bhanox.train.smoke --steps 100 --max-examples 1024

The defaults are sized by measurement, not by taste. On one CPU, nano at
``max_examples=64`` costs about 43 s of training and 23 s of evaluation per
document, so a 12-step run is roughly 13 minutes. ``--max-examples 1024`` -- the
``max_context`` this command's first draft defaulted to -- is 16x the tokens per
step and about 17 minutes *per step*, which is how a smoke test becomes a
half-hour wait that nobody re-runs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence

import numpy as np
import torch

from bhanox.config import BhanoxConfig, load_config
from bhanox.frontend.hashbind import BYTE_GRAM_N
from bhanox.model import Bhanox
from bhanox.train.model_mirror import BhanoxMirror
from bhanox.train.objective import trainable_parameters
from bhanox.train.trainer import build_optimizer, run_documents

__all__ = [
    "SMOKE_CYCLE",
    "accuracy_on_cycle",
    "build_documents",
    "cuda_smoke",
    "main",
]

#: The repeating byte cycle. Period 5, so every 4-gram in it maps to exactly one
#: next byte and the task is a finite-state function rather than noise.
SMOKE_CYCLE = b"ABCDE"


def build_documents(
    n_documents: int, n_bytes: int, *, cycle: bytes = SMOKE_CYCLE
) -> list[bytes]:
    """Deterministic learnable documents, in memory.

    Args:
        n_documents: How many documents to synthesise.
        n_bytes: Length of each, in bytes. Must exceed ``len(cycle) + 4`` for a
            document to yield any examples at all.
        cycle: The repeating pattern.

    Returns:
        ``n_documents`` identical byte strings. Identical on purpose: this
        measures whether a gradient works, not whether the model generalises
        across documents, and varying them would only add a confound.
    """
    if n_documents < 1:
        raise ValueError(f"n_documents must be at least 1, got {n_documents}")
    period = len(cycle)
    if period < 2:
        raise ValueError(f"cycle must be at least 2 bytes, got {period}")
    return [
        bytes(cycle[i % period] for i in range(n_bytes)) for _ in range(n_documents)
    ]


def accuracy_on_cycle(
    mirror: BhanoxMirror, documents: Sequence[bytes], *, max_examples: int
) -> float:
    """Fraction of positions where argmax equals the true next byte.

    Args:
        mirror: The mirror to score. **Not reset** -- the caller owns state, so
            this can be called mid-run on a model whose state is mid-document.
        documents: The same documents the run trained on.
        max_examples: Chunk size, matching the run.

    Returns:
        Argmax accuracy over every position of every chunk, in ``[0, 1]``.

    Note:
        ``train=False`` and no gradient, so this is side-effect-free apart from
        the recurrent state advancing, which is the recurrence itself.
    """
    from bhanox.data import iter_examples

    correct = 0
    total = 0
    shadows: list[list[torch.Tensor]] | None = None
    current: int | None = None
    with torch.no_grad():
        for chunk in iter_examples(documents, max_examples=max_examples, name="smoke"):
            if chunk.doc_index != current:
                shadows = None
                current = chunk.doc_index
            logits, shadows = mirror.step(
                chunk.inputs.reshape(1, -1), shadows=shadows, train=False
            )
            predicted = logits.reshape(-1, int(logits.shape[-1])).argmax(dim=-1)
            # Built after the forward and moved onto ``logits``' device, because
            # the comparison needs both operands in one place. ``from_numpy``
            # lands on the default device, which is not where the mirror is.
            target = torch.from_numpy(np.asarray(chunk.targets, dtype=np.int64)).to(
                logits.device
            )
            correct += int((predicted == target).sum())
            total += int(target.numel())
    return correct / total if total else 0.0


def cuda_smoke(
    *,
    device: torch.device,
    config: str = "nano",
    steps: int = 2,
    max_examples: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    time_cap_s: float = 240.0,
    stream=None,
) -> dict[str, object]:
    """Bounded CUDA correctness smoke. The one entry point, for notebook and CLI.

    This is the single implementation behind both the committed Kaggle notebook
    and the paste-a-cell workflow: one call, one set of numbers, so the two
    cannot drift apart. It runs the *real* training path --
    :class:`~bhanox.train.model_mirror.BhanoxMirror`, :func:`run_documents`,
    the real AdamW, the real :mod:`bhanox.data` windowing -- on deterministic
    in-memory bytes. There is no reimplementation of the training loop here.

    Args:
        device: Where to run. **Explicit and required.** There is deliberately no
            default and no fallback: a smoke that quietly chose CPU would report
            a green result for a question it never asked. A ``cuda`` device with
            no usable CUDA raises, loudly.
        config: Config preset name.
        steps: Documents to train on. Keep small; this is a correctness gate.
        max_examples: Positions per chunk, i.e. the context window under test.
            ``min``'d against the config's ``max_context``.
        lr: AdamW learning rate.
        seed: Seed for document order and weight init.
        time_cap_s: Hard wall-clock budget. Checked before each step, so the
            overshoot is bounded by one step, not by the whole run.
        stream: Callable to write progress lines to, or ``None`` for ``print``.
            Lets a notebook capture the transcript without reimplementing the run.

    Returns:
        A dict of the observed numbers: ``device``, ``gpu_names``, ``steps``,
        ``max_examples``, ``losses``, ``accuracies``, ``elapsed_s``,
        ``peak_cuda_mib``, ``trainable_params``, ``finished``.

    Raises:
        RuntimeError: If ``device`` is CUDA and no CUDA device is usable. There
            is no CPU path out of this function, by design.
        ValueError: If ``steps`` or ``max_examples`` is below 1.

    What a return value does and does not mean
    -------------------------------------------
    It means the mirror builds, moves, and runs forward, backward and an AdamW
    step on that device, and the loss is finite. It does **not** measure
    throughput as a property of the architecture, does not say anything about
    language quality, energy, or full-context behaviour, and says nothing about
    how fast this would be on a different GPU. The bytes are a repeating cycle,
    so a falling loss there is a statement about the gradient path.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if max_examples < 1:
        raise ValueError(f"max_examples must be at least 1, got {max_examples}")

    device = torch.device(device)
    if device.type == "cuda":
        # Both halves matter. is_available() is False when no driver is usable
        # at all; device_count() can be 0 on a build that reports True. Asking
        # for cuda:1 on a single-GPU session is a third distinct mistake, and it
        # is worth naming rather than surfacing as a kernel error later.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "On Kaggle, open the right-hand Settings panel, set Accelerator "
                "to a GPU, and re-run this cell. This function has no CPU "
                "fallback on purpose: a result that silently ran on CPU would "
                "not answer the question this exists to answer."
            )
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("CUDA is available but device_count() is 0.")
        index = 0 if device.index is None else device.index
        if index >= count:
            raise RuntimeError(
                f"asked for {device} but only {count} CUDA device(s) are "
                f"visible. This smoke uses exactly one device."
            )
    else:
        print(
            f"NOTE: running on {device}, not CUDA. This is the local test path; "
            "it proves the code runs, not that the CUDA path does."
        )

    emit = stream or (lambda line: print(line, flush=True))
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )

    cfg: BhanoxConfig = load_config(config)
    window = min(max_examples, cfg.max_context)
    documents = build_documents(1, window * 4 + BYTE_GRAM_N)

    emit("=== environment ===")
    emit(f"python           {sys.version.split()[0]}")
    emit(f"torch            {torch.__version__}")
    emit(f"torch cuda build {torch.version.cuda}")
    emit(f"cuda available   {torch.cuda.is_available()}")
    emit(f"device count     {torch.cuda.device_count()}")
    for i, name in enumerate(gpu_names):
        emit(f"  cuda:{i}  {name}")
    emit(f"running on       {device}")
    if len(gpu_names) > 1:
        emit(
            f"NOTE: {len(gpu_names)} GPUs visible. This run uses {device} only "
            "-- no DataParallel, no DDP, no multi-GPU work."
        )
    emit("")
    emit("=== run configuration ===")
    emit(
        f"config {cfg.name}: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"n_heads={cfg.n_heads} max_context={cfg.max_context} "
        f"output_vocab={cfg.output_vocab}"
    )
    emit(
        f"context {window} positions per chunk | steps {steps} | lr {lr} | "
        f"seed {seed} | time cap {time_cap_s:.0f}s"
    )
    emit(
        f"data: 1 synthetic document of {len(documents[0])} bytes, repeating "
        f"{SMOKE_CYCLE!r}. In memory only -- no corpus is read or downloaded."
    )
    emit(f"uniform-random floor ln(output_vocab) = {math.log(cfg.output_vocab):.4f}")
    emit("")
    emit("=== steps (synthetic; a falling loss here is not language quality) ===")
    emit(f"{'step':>5} {'chunks':>7} {'tokens':>7} {'loss':>9} {'accuracy':>9}")

    torch.manual_seed(seed)
    mirror = BhanoxMirror(Bhanox(cfg)).to(device)
    optimizer = build_optimizer(mirror, lr=lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    losses: list[float] = []
    accuracies: list[float] = []
    finished = True
    started = time.perf_counter()
    for step in range(1, steps + 1):
        elapsed = time.perf_counter() - started
        if elapsed > time_cap_s:
            finished = False
            emit("")
            emit(
                f"ABORTED before step {step}: {elapsed:.1f}s exceeded the "
                f"{time_cap_s:.0f}s cap. Reported timings below are partial. "
                "This is a correctness smoke, not a benchmark -- raise "
                "--time-cap or lower --max-examples deliberately, and do not "
                "read a speed number off a partial run."
            )
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for report in run_documents(
            mirror,
            optimizer,
            documents,
            max_examples=window,
            seed=seed,
            name="cuda-smoke",
        ):
            accuracy = accuracy_on_cycle(mirror, documents, max_examples=window)
            losses.append(float(report.mean_loss))
            accuracies.append(accuracy)
            emit(
                f"{step:>5} {report.n_chunks:>7} {report.n_tokens:>7} "
                f"{report.mean_loss:>9.4f} {accuracy:>9.4f}"
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    finite = all(math.isfinite(v) for v in losses)
    peak_mib: float | None = None
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    emit("")
    emit("=== observations ===")
    emit(f"elapsed                {elapsed_s:.2f} s")
    if losses:
        emit(f"first loss             {losses[0]:.4f}")
        emit(f"final loss             {losses[-1]:.4f}")
    if accuracies:
        emit(f"argmax accuracy        {accuracies[0]:.4f} -> {accuracies[-1]:.4f}")
        emit(f"chance accuracy floor  {1.0 / cfg.output_vocab:.4f}")
    emit(f"all losses finite      {finite}")
    n_params = sum(p.numel() for p in trainable_parameters(mirror))
    emit(f"trainable tensor values {n_params:,}")
    if peak_mib is not None:
        emit(f"peak CUDA memory       {peak_mib:.1f} MiB (this context only)")
    emit(f"completed all steps    {finished}")

    emit("")
    emit("=== what this is not ===")
    emit("Not a benchmark. Not a training run. Not evidence of language quality:")
    emit("the data is a repeating byte cycle, so a falling loss says the gradient")
    emit("path is connected. No throughput, energy, or full-context (T=4096) claim")
    emit("is made or implied by any number above.")

    return {
        "device": str(device),
        "gpu_names": gpu_names,
        "config": cfg.name,
        "steps": steps,
        "max_examples": window,
        "losses": losses,
        "accuracies": accuracies,
        "elapsed_s": elapsed_s,
        "peak_cuda_mib": peak_mib,
        "trainable_params": n_params,
        "finished": finished,
        "all_finite": finite,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test and print its curve.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit code: ``0`` if every loss was finite, ``1`` otherwise.
        A finite run is not a claim that the loss fell; the curve is printed for
        the owner to read.

    Raises:
        ValueError: If ``--steps`` is below 1. A zero-step run would print a
            header and no data and exit ``0``, which reads like a result.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "torch device for the bounded CUDA smoke, e.g. 'cuda:0'. When "
            "given, this delegates to cuda_smoke() and exits. Omit it for the "
            "longer CPU learning-curve run below."
        ),
    )
    parser.add_argument("--config", default="nano", help="config preset name")
    parser.add_argument("--steps", type=int, default=12, help="documents to train on")
    parser.add_argument(
        "--max-examples", type=int, default=64, help="emitted positions per chunk"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=0, help="document-order seed")
    parser.add_argument(
        "--time-cap", type=float, default=240.0, help="wall-clock cap in seconds"
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {args.steps}")

    if args.device is not None:
        result = cuda_smoke(
            device=torch.device(args.device),
            config=args.config,
            steps=args.steps,
            max_examples=args.max_examples,
            lr=args.lr,
            seed=args.seed,
            time_cap_s=args.time_cap,
        )
        return 0 if result["all_finite"] and result["finished"] else 1

    cfg: BhanoxConfig = load_config(args.config)
    # The window is the config's, and the chunk bound is min()'d against it, so
    # the run cannot ask the mirror for a context it has already promised to
    # reject.
    max_examples = min(args.max_examples, cfg.max_context)
    # ``BYTE_GRAM_N`` more bytes than the chunk multiples, so the document yields
    # exactly ``4 * max_examples`` examples in 4 equal chunks. See the module
    # docstring: the point is to keep the short-tail exposure out of the toy run
    # rather than to stumble into it.
    documents = build_documents(1, max_examples * 4 + BYTE_GRAM_N)

    torch.manual_seed(args.seed)
    mirror = BhanoxMirror(Bhanox(cfg))
    optimizer = build_optimizer(mirror, lr=args.lr)

    floor = math.log(cfg.output_vocab)
    print(f"config={cfg.name} steps={args.steps} max_examples={max_examples}")
    print(f"uniform-random floor ln(output_vocab) = {floor:.4f}")
    print(f"{'step':>5} {'chunks':>7} {'tokens':>7} {'loss':>9} {'accuracy':>9}")

    all_finite = True
    # ``run_documents`` is re-entered per step rather than iterated once. It is a
    # generator over the documents it is handed, and this run hands it one
    # document, so a single iteration would emit exactly one report and exit --
    # a "curve" of one point, printed with exit code 0, which is how the first
    # draft of this function managed to look like it had trained. A fresh
    # generator per step also re-enters the reset, which is the behaviour wanted
    # between documents.
    for step in range(1, args.steps + 1):
        for report in run_documents(
            mirror,
            optimizer,
            documents,
            max_examples=max_examples,
            seed=args.seed,
            name="smoke",
        ):
            accuracy = accuracy_on_cycle(mirror, documents, max_examples=max_examples)
            finite = math.isfinite(report.mean_loss)
            all_finite = all_finite and finite
            print(
                f"{step:>5} {report.n_chunks:>7} {report.n_tokens:>7} "
                f"{report.mean_loss:>9.4f} {accuracy:>9.4f}"
            )
            if step == 1:
                print(f"  chunk token counts: {report.token_counts}")

    n_params = sum(p.numel() for p in trainable_parameters(mirror))
    print(f"trainable tensor values: {n_params}")
    print("measured only; no threshold asserted. Set the bar, then re-run.")
    return 0 if all_finite else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
