"""Generation: a pure sequential loop with O(1) cost per token.

Purpose: turn a prompt into tokens without ever growing a cache. The state is
fixed-size, so the loop body allocates the same amount of memory at token 1 as
at token 100,000.

In simple words: no attention cache, no growing list, no "context length"
cliff. The model remembers what it can fit and forgets the rest on a schedule.

The O(1) claim is not asserted here, it is measured: ``tests/test_generate.py``
asserts the resident state is still 8,192 B after 64 generated tokens, and
``tests/test_model.py`` pins the same figure after 64 steps.

Bytes touched per token: the entire model is a fixed working set, so the figure
is constant and is what :func:`bhanox.audit.audit_bytes_per_token` reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    # Imported for the annotation only: model.py imports this module inside
    # generate(), so a runtime import here would be a cycle.
    from bhanox.model import Bhanox

__all__ = ["generate", "sample", "softmax"]


def softmax(z: NDArray[np.floating]) -> NDArray[np.float32]:
    """Numerically stable softmax over the last axis.

    Args:
        z: Real logits.

    Returns:
        Probabilities summing to 1 along the last axis.
    """
    z = np.asarray(z, dtype=np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-9)).astype(np.float32)


def sample(
    logits: NDArray[np.floating], temperature: float, rng: np.random.Generator
) -> int:
    """Pick one token id from logits.

    Args:
        logits: ``(vocab,)`` next-token scores.
        temperature: Softmax temperature. Values at or below 0 mean greedy
            argmax, which is what a deterministic test needs.
        rng: Random source, for reproducibility.

    Returns:
        The sampled token id.
    """
    if temperature <= 0.0:
        return int(np.argmax(logits))
    probs = softmax(np.asarray(logits, dtype=np.float32) / temperature)
    return int(rng.choice(probs.shape[-1], p=probs))


def generate(
    model: Bhanox,
    ids: NDArray[np.integer],
    *,
    max_new: int = 64,
    temperature: float = 0.8,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Generate ``max_new`` tokens after a prompt.

    Args:
        model: A :class:`bhanox.model.Bhanox`.
        ids: Prompt ids, any shape. Flattened to a 1-D sequence.
        max_new: Number of tokens to generate.
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        seed: RNG seed. Fixed seed means byte-identical output, which is what
            makes generation testable at all.

    Returns:
        ``int64`` array of the prompt followed by the generated ids.

    Raises:
        ValueError: If ``max_new`` is negative, or the prompt is empty.
    """
    if max_new < 0:
        raise ValueError(f"max_new must be >= 0, got {max_new}")
    prompt = np.asarray(ids, dtype=np.int64).reshape(-1)
    if prompt.size == 0:
        raise ValueError("generate needs a non-empty prompt")
    rng = np.random.default_rng(seed)
    out = [int(t) for t in prompt]
    # The prompt is consumed by the state, and the forward pass it returns
    # *already* predicts the next token. Re-stepping the last prompt token would
    # feed it to the model a second time, so the state would hold it twice.
    logits = model.forward(prompt[None, :])[0, -1]
    for _ in range(max_new):
        nxt = sample(logits, temperature, rng)
        out.append(nxt)
        # Feed the token just sampled, which advances the state by exactly one
        # and yields the distribution for the token after it. The final call's
        # result is discarded, which costs one step and saves the branch.
        logits = model.step(np.int64(nxt))
    return np.asarray(out, dtype=np.int64)
