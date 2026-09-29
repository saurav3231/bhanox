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
from bhanox.data import examples_for, iter_examples
from bhanox.frontend.hashbind import BYTE_GRAM_N
from bhanox.model import Bhanox
from bhanox.train.model_mirror import BhanoxMirror
from bhanox.train.objective import trainable_parameters
from bhanox.train.trainer import build_optimizer, run_documents, train_chunk

__all__ = [
    "SMOKE_CYCLE",
    "T4096_CONTEXT",
    "accuracy_on_cycle",
    "build_documents",
    "build_probe_document",
    "cuda_smoke",
    "main",
    "t4096_probe",
]

#: The repeating byte cycle. Period 5, so every 4-gram in it maps to exactly one
#: next byte and the task is a finite-state function rather than noise.
SMOKE_CYCLE = b"ABCDE"

#: The full planned context. ``nano``'s ``max_context``, and the window the
#: opt-in capacity probe runs at. Named here so the number lives in one place:
#: it is the single most expensive thing in the smoke suite, and a probe that
#: quietly ran at a smaller window would be a different, much cheaper experiment
#: wearing this one's name.
T4096_CONTEXT = 4096


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


def build_probe_document(n_bytes: int, *, seed: int = 0) -> bytes:
    """Deterministic, *varied* synthetic bytes for the full-context probe.

    Distinct from :func:`build_documents` on purpose. The short smoke wants a
    learnable 5-cycle, so a falling loss says something about the gradient path.
    The full-context probe wants the opposite: bytes that are not a short
    repeating pattern, so the loss sits near the ``ln(256)`` noise floor and the
    number is reported as a *diagnostic only* -- never as learning, and never as
    evidence the model is good. A cycle at T=4096 would be 4096/5 = 819
    repetitions of the same five bytes, which invites exactly the wrong reading.

    The bytes are drawn from a seeded :class:`numpy.random.Generator`, so the
    same ``seed`` gives the same document and a failed run can be reproduced.
    They are uniform over the full byte range rather than over a restricted
    alphabet, so no 4-gram is degenerate and the input distribution is the
    densest one available -- the hardest case for a lookup-style embedder, and
    therefore the least flattering capacity test.

    Args:
        n_bytes: Length in bytes. Must be at least ``context + BYTE_GRAM_N`` for
            the document to yield ``context`` examples.
        seed: Seed for the byte draw. Draws the *bytes* only; it does not touch
            the model initialisation, which the caller seeds separately.

    Returns:
        One ``bytes`` object of length ``n_bytes``.

    Note:
        This is in-memory only. Nothing is read from disk, downloaded, or
        cached, and the bytes are not a corpus: they are not text, carry no
        meaning, and are discarded when the process exits.
    """
    if n_bytes < 1:
        raise ValueError(f"n_bytes must be at least 1, got {n_bytes}")
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, n_bytes, dtype=np.uint8).tobytes()


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


def t4096_probe(
    *,
    device: torch.device,
    config: str = "nano",
    context: int = T4096_CONTEXT,
    lr: float = 1e-3,
    seed: int = 0,
    stream=None,
) -> dict[str, object]:
    """Opt-in capacity probe: exactly one real update at the full context.

    Separate from :func:`cuda_smoke` on purpose, and **not** a default. The short
    smoke is the quick gate and stays that way. This one answers a different
    question -- *does a full-context window fit and run on this device at all* --
    and the two must not be conflated, because a pass here is a statement about
    capacity and nothing else.

    What it does, exactly:

    - Builds one synthetic document of ``context + BYTE_GRAM_N`` bytes from
      :func:`build_probe_document`: deterministic, varied, uniform over all 256
      byte values. **Not** a repeating cycle, so the loss is a meaningless
      diagnostic and is labelled as one. Nothing is read, downloaded, or cached.
    - Asserts the generated example count is exactly ``context``, that ids are
      below ``2**(8 * BYTE_GRAM_N)`` and targets lie in ``0..output_vocab - 1``,
      before any tensor is built. A probe that silently ran on 4092 examples
      because of an off-by-one would report a green capacity result for a
      context it never touched.
    - Runs **one** chunk through :func:`~bhanox.train.trainer.train_chunk`: the
      real mirror, the real forward, the real ``cross_entropy``, the real
      backward, the real AdamW step. Not a reimplementation.
    - Resets peak-memory stats *after* building the model and optimizer and
      *immediately before* the update, so the reported peak is the update's, not
      the setup's. Synchronises on both sides of the timed region, so the elapsed
      time is the update's and not the queue's.

    Args:
        device: Where to run. Required and explicit, like :func:`cuda_smoke`.
        config: Config preset name. ``max_context`` must be at least ``context``.
        context: Positions in the single update. Defaults to :data:`T4096_CONTEXT`.
        lr: AdamW learning rate.
        seed: Seeds the byte draw and weight init.
        stream: Callable to write progress lines to, or ``None`` for ``print``.

    Returns:
        A dict with ``completed``, ``context``, ``loss``, ``loss_finite``,
        ``grads_finite``, ``params_finite``, ``elapsed_s``, ``peak_alloc_mib``,
        ``peak_reserved_mib``, and the free/total memory seen before the update.

    Raises:
        RuntimeError: If CUDA is unavailable or ``device_count()`` is 0. There is
            no CPU fallback: a capacity result measured on CPU would answer a
            question nobody asked.
        ValueError: If ``context`` is below 1, or exceeds the config's
            ``max_context``.
        torch.cuda.OutOfMemoryError: Propagated unchanged, after the exact
            failure has been emitted. The probe does **not** silently shrink the
            context, retry on another device, or fall back -- an OOM at full
            context is the answer, and papering over it would destroy the only
            thing this function exists to measure.

    Note:
        There is deliberately **no timeout**. A wall-clock cap cannot interrupt a
        blocked ``train_chunk`` -- it would only be checked *after* the update
        returned, by which point the GPU has already done the work. A fake
        timeout that cannot fire is worse than none: it reads like a safety rail
        in the signature while providing none. The cost is unknown and can be
        large; the caller is expected to watch the cell and interrupt it by hand.

    What a pass does and does not mean
    ----------------------------------
    One synthetic T=4096 update completed on that device: it fits in memory, the
    full context runs, and loss, gradients and post-update parameters are finite.
    It does **not** establish sustained training, throughput, learning on a real
    corpus, quality, energy use, or that checkpointing would work in a long run.
    One update cannot show any of those, and none of them may be inferred from a
    pass.
    """
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"t4096_probe is a CUDA capacity probe and was given {device}. It has "
            "no CPU path on purpose: a peak-memory number from CPU RAM would say "
            "nothing about whether the full context fits in a GPU, which is the "
            "only question this function exists to answer. Use cuda_smoke() for "
            "the quick CPU-suitable smoke."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "t4096_probe requires CUDA and torch.cuda.is_available() is False. On "
            "Kaggle, set Accelerator to a GPU in the right-hand Settings panel. "
            "There is no CPU fallback here by design."
        )
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("CUDA is available but device_count() is 0.")
    index = 0 if device.index is None else device.index
    if index >= count:
        raise RuntimeError(
            f"asked for {device} but only {count} CUDA device(s) are visible. "
            "This probe uses exactly one device."
        )

    emit = stream or (lambda line: print(line, flush=True))

    emit("!! FULL-CONTEXT CAPACITY PROBE -- OPT-IN, AND NOT CHEAP !!")
    emit(
        f"One real optimizer update at context {context}. Runtime is UNKNOWN and "
        "may take substantial GPU time and quota:"
    )
    emit(
        "the T=32 smoke above took ~34 s for TWO steps, and this is 128x the "
        "context in a single update. No linear extrapolation is offered, because "
        "none is trustworthy. There is no timeout in this function: a wall-clock "
        "cap cannot interrupt a blocked train_chunk."
    )
    emit("Watch the cell. If it must be stopped, interrupt it manually.")
    emit("")

    cfg: BhanoxConfig = load_config(config)
    if context > cfg.max_context:
        raise ValueError(
            f"context {context} exceeds the {cfg.name} config's max_context="
            f"{cfg.max_context}. Raise the config or lower the context; do not "
            "ask the mirror for a window it has already promised to reject."
        )

    gpu_names = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

    emit("=== environment ===")
    emit(f"python           {sys.version.split()[0]}")
    emit(f"torch            {torch.__version__}")
    emit(f"torch cuda build {torch.version.cuda}")
    emit(f"cuda available   {torch.cuda.is_available()}")
    emit(f"device count     {count}")
    for i, name in enumerate(gpu_names):
        emit(f"  cuda:{i}  {name}")
    emit(f"running on       {device}")
    if len(gpu_names) > 1:
        emit(
            f"NOTE: {len(gpu_names)} GPUs visible. This run uses {device} only "
            "-- no DataParallel, no DDP, no multi-GPU work."
        )
    emit("")
    emit("=== probe configuration ===")
    emit(
        f"config {cfg.name}: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"n_heads={cfg.n_heads} max_context={cfg.max_context} "
        f"output_vocab={cfg.output_vocab}"
    )
    emit(f"batch 1 | context {context} positions | steps 1 | lr {lr} | seed {seed}")

    # The data contract is checked before any tensor exists, so a bad document
    # cannot be reported as an OOM or a capacity result.
    n_bytes = context + BYTE_GRAM_N
    document = build_probe_document(n_bytes, seed=seed)
    examples, targets = examples_for(document)
    assert len(examples) == context, (
        f"expected exactly {context} examples from {n_bytes} bytes, got "
        f"{len(examples)}; the probe would be measuring a different context"
    )
    assert len(targets) == context, (
        f"expected exactly {context} targets, got {len(targets)}"
    )
    id_ceiling = 1 << (8 * BYTE_GRAM_N)
    assert int(examples.min()) >= 0 and int(examples.max()) < id_ceiling, (
        f"example ids must lie in [0, {id_ceiling}) for {BYTE_GRAM_N}-grams; got "
        f"{int(examples.min())}..{int(examples.max())}"
    )
    assert int(targets.min()) >= 0 and int(targets.max()) < cfg.output_vocab, (
        f"targets must lie in [0, {cfg.output_vocab}) for output_vocab="
        f"{cfg.output_vocab}; got {int(targets.min())}..{int(targets.max())}"
    )
    emit(
        f"data: 1 synthetic document of {n_bytes} deterministic varied bytes "
        f"(seed {seed}), uniform over all 256 byte values. In memory only -- no "
        "corpus is read or downloaded, and these bytes are not text."
    )
    emit(
        f"examples {len(examples)} (ids < {id_ceiling}, targets in "
        f"[0, {cfg.output_vocab})) -- asserted"
    )
    emit("")

    free_before, total = torch.cuda.mem_get_info(device)
    emit("=== device memory before the update ===")
    emit(f"free   {free_before / (1024**2):,.1f} MiB")
    emit(f"total  {total / (1024**2):,.1f} MiB")

    torch.manual_seed(seed)
    mirror = BhanoxMirror(Bhanox(cfg)).to(device)
    optimizer = build_optimizer(mirror, lr=lr)
    n_params = sum(p.numel() for p in trainable_parameters(mirror))
    emit(f"model built and moved to {device} ({n_params:,} trainable values)")

    # The real pipeline, not a hand-rolled forward. One document of exactly
    # ``context`` examples yields exactly one chunk, hence exactly one update.
    chunk = next(
        iter_examples([document], max_examples=context, seed=seed, name="t4096-probe")
    )
    assert len(chunk) == context, (
        f"expected one chunk of {context} positions, got {len(chunk)}"
    )

    # Reset *here*: after the model and optimizer exist, immediately before the
    # update. Resetting earlier would fold setup allocations into the peak and
    # overstate what the update needs.
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    emit("")
    emit(f"=== running ONE update at context {context} (batch 1) ===")
    started = time.perf_counter()
    error: str | None = None
    oom: torch.cuda.OutOfMemoryError | None = None
    loss_value: float | None = None
    try:
        report, _ = train_chunk(mirror, optimizer, chunk)
        loss_value = report.loss
    except torch.cuda.OutOfMemoryError as exc:
        # Kept so it can be re-raised *after* the report is printed. The report is
        # the expensive part of the result: the peak reached before the failure is
        # the most informative number an OOM produces.
        oom = exc
        error = f"torch.cuda.OutOfMemoryError: {exc}"
    except RuntimeError as exc:
        error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    if error is not None:
        emit("")
        emit("=== FAILED ===")
        emit(f"the single update did not complete: {error}")
        emit(f"elapsed               {elapsed_s:.2f} s")
        emit(f"peak allocated        {peak_alloc:.1f} MiB")
        emit(f"peak reserved         {peak_reserved:.1f} MiB")
        emit("")
        emit("Reported as-is. This probe does NOT shrink the context, retry, or")
        emit("switch devices: an OOM at full context is the measurement.")
        emit("")
        emit("=== what this is not ===")
        emit("A failure here is evidence about capacity only. It says nothing")
        emit("about throughput, quality, energy, or whether a smaller context")
        emit("would train -- none of which was measured.")
        if oom is not None:
            raise oom
        raise RuntimeError(error)

    grads_finite = all(
        bool(torch.isfinite(p.grad).all())
        for p in trainable_parameters(mirror)
        if p.grad is not None
    )
    params_finite = all(
        bool(torch.isfinite(p).all()) for p in trainable_parameters(mirror)
    )

    emit("")
    emit("=== result ===")
    emit(f"update completed     {error is None}")
    emit(f"context               {context} positions, batch 1, 1 update")
    emit(
        f"loss                  {loss_value:.4f}   (MEANINGLESS DIAGNOSTIC)"
        if loss_value is not None
        else "loss                  n/a"
    )
    emit(
        f"loss finite           {loss_value is not None and math.isfinite(loss_value)}"
    )
    emit(f"gradients finite      {grads_finite}")
    emit(f"parameters finite     {params_finite}")
    emit(f"elapsed               {elapsed_s:.2f} s (synchronised either side)")
    emit(f"peak allocated        {peak_alloc:.1f} MiB")
    emit(f"peak reserved         {peak_reserved:.1f} MiB")
    emit(f"free before update    {free_before / (1024**2):,.1f} MiB")
    emit(f"device total          {total / (1024**2):,.1f} MiB")
    if error is not None:
        emit(f"error                 {error}")

    emit("")
    emit("=== what this is not ===")
    emit("One synthetic T=4096 update completed (or failed) on one device. That is")
    emit("all it establishes. It does NOT show: sustained training, throughput,")
    emit("learning on a real corpus, language quality, energy use, or that a long")
    emit("run's checkpoint/resume would work. The bytes are uniform random, so the")
    emit("loss above carries no learning signal and is a diagnostic only; a value")
    emit("near ln(256) is the expected result, not a failure.")
    emit("No GPU speed or energy claim is made or implied by any number above.")

    return {
        "device": str(device),
        "gpu_names": gpu_names,
        "config": cfg.name,
        "context": context,
        "batch": 1,
        "steps": 1,
        "completed": error is None,
        "loss": loss_value,
        "loss_finite": loss_value is not None and math.isfinite(loss_value),
        "grads_finite": grads_finite,
        "params_finite": params_finite,
        "elapsed_s": elapsed_s,
        "peak_alloc_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "free_before_mib": free_before / (1024**2),
        "total_mib": total / (1024**2),
        "trainable_params": n_params,
        "error": error,
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
    parser.add_argument(
        "--t4096-probe",
        action="store_true",
        help=(
            "Run the opt-in full-context capacity probe instead of the short "
            "smoke: exactly ONE real update at context 4096, on CUDA, with "
            "varied synthetic bytes. Expensive and off by default; see "
            "t4096_probe()."
        ),
    )
    parser.add_argument(
        "--t4096-context",
        type=int,
        default=T4096_CONTEXT,
        help=f"context for --t4096-probe (default {T4096_CONTEXT})",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {args.steps}")

    if args.t4096_probe:
        device = torch.device(args.device or "cuda:0")
        result = t4096_probe(
            device=device,
            config=args.config,
            context=args.t4096_context,
            lr=args.lr,
            seed=args.seed,
        )
        # Exit code reflects completion and finiteness only. A failure path has
        # already raised, so reaching here means the update ran.
        ok = bool(
            result["completed"] and result["loss_finite"] and result["grads_finite"]
        )
        return 0 if ok else 1

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
