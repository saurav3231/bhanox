"""Generation: a pure sequential loop with O(1) cost per token.

Purpose: turn a byte prompt into more bytes without ever growing a cache. The
state is fixed-size, so the loop body allocates the same amount of memory at
token 1 as at token 100,000.

In simple words: no attention cache, no growing list, no "context length"
cliff. The model remembers what it can fit and forgets the rest on a schedule.
Because the loop feeds one 4-gram at a time through :meth:`Bhanox.step` rather
than a window through :meth:`Bhanox.forward`, generation is not bounded by
``max_context`` -- that limit exists for training windows, not for the prompt
you happen to type.

The O(1) claim is not asserted here, it is measured: ``tests/test_generate.py``
asserts the resident state is still 8,192 B after 64 generated tokens, and
``tests/test_model.py`` pins the same figure after 64 steps.

Bytes touched per token: the entire model is a fixed working set, so the figure
is constant and is what :func:`bhanox.audit.audit_bytes_per_token` reports.

The unit contract, which the two halves of the model do not share: one model
*input* is a packed byte 4-gram and one model *output class* is a single byte.
:meth:`Bhanox.generate` is the byte API and hides the ids entirely;
:func:`generate_ids` is the low-level door for callers who already hold ids.
Neither ever returns prompt ids and byte values in one array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bhanox.frontend.hashbind import BYTE_GRAM_N, BYTE_GRAM_ORDER, encode_bytes

if TYPE_CHECKING:  # pragma: no cover
    # Imported for the annotation only: model.py imports this module inside
    # generate(), so a runtime import here would be a cycle.
    from bhanox.model import Bhanox

__all__ = ["BYTE_CLASSES", "generate", "generate_ids", "sample", "softmax"]

#: One output class per byte value. The head is ``(d_model, output_vocab)`` and
#: the frozen task is next-byte prediction, so 256 is the only width for which a
#: sampled class means anything. Checked at runtime rather than trusted from the
#: config, because a config that is merely valid is not necessarily this one.
BYTE_CLASSES = 256


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
    """Pick one byte value from logits.

    Args:
        logits: ``(vocab,)`` next-byte scores.
        temperature: Softmax temperature. Values at or below 0 mean greedy
            argmax, which is what a deterministic test needs.
        rng: Random source, for reproducibility.

    Returns:
        The sampled byte value, in ``0..255``.
    """
    if temperature <= 0.0:
        return int(np.argmax(logits))
    probs = softmax(np.asarray(logits, dtype=np.float32) / temperature)
    return int(rng.choice(probs.shape[-1], p=probs))


def _check_head(model: Bhanox) -> None:
    """Refuse a model whose output head is not one class per byte value.

    Args:
        model: The model about to generate.

    Raises:
        ValueError: If ``output_vocab`` is not 256.
    """
    width = int(model.config.output_vocab)
    if width != BYTE_CLASSES:
        raise ValueError(
            f"byte generation needs output_vocab={BYTE_CLASSES} -- one class "
            f"per byte, because the task is next-byte prediction -- but this "
            f"model's head is {width} wide. A sampled class of such a head is "
            "not a byte, so the result would not be bytes. Resize the head, or "
            "call generate_ids() if the classes are something you know."
        )


def _check_count(max_new: int) -> None:
    """Args:
        max_new: Requested number of new bytes.

    Raises:
        ValueError: If ``max_new`` is negative.
    """
    if max_new < 0:
        raise ValueError(f"max_new must be >= 0, got {max_new}")


def _roll(window: bytes, byte: int) -> bytes:
    """Advance the context window by exactly one byte.

    Args:
        window: The current ``BYTE_GRAM_N``-byte context, as bytes.
        byte: The byte value just predicted.

    Returns:
        The next context: ``window`` shifted left with ``byte`` appended.

    A model input is the packed window, so moving the text forward one byte is
    ``window[1:] + byte`` and nothing else. Feeding ``byte`` itself would be a
    different -- and meaningless -- 4-gram, which is the whole defect this
    module exists to not have.
    """
    return window[1:] + bytes((byte,))


def _run(
    model: Bhanox,
    grams: NDArray[np.integer],
    window: bytes,
    *,
    max_new: int,
    temperature: float,
    seed: int | None,
) -> list[int]:
    """The one loop both public entry points share.

    Args:
        model: The model to sample from.
        grams: Context ids to consume before the first prediction, one per
            position of the prompt.
        window: Those contexts' final window, as bytes, to roll forward from.
        max_new: How many bytes to generate.
        temperature: Sampling temperature; ``<= 0`` is greedy.
        seed: RNG seed.

    Returns:
        The generated byte values, and nothing else. The prompt is not echoed
        back, so the caller's ids and these bytes never share one array.
    """
    rng = np.random.default_rng(seed)
    out: list[int] = []
    logits: NDArray[np.float32] | None = None
    for gram in grams:
        # One input per context, in order. The last one's logits are the ones
        # that predict the byte after the prompt.
        logits = model.step(int(gram))
    for _ in range(max_new):
        assert logits is not None  # a non-empty prompt left logits behind
        nxt = sample(logits, temperature, rng)
        out.append(nxt)
        window = _roll(window, nxt)
        # Stepping the rolled window advances the state by exactly one input and
        # yields the distribution for the byte after it. The final step's logits
        # are discarded, which costs one step and saves the branch -- and it
        # leaves the state holding every generated byte.
        logits = model.step(int(encode_bytes(window, n=BYTE_GRAM_N)[0]))
    return out


def generate(
    model: Bhanox,
    prompt: bytes,
    *,
    max_new: int = 64,
    temperature: float = 0.8,
    seed: int | None = None,
    reset: bool = True,
) -> bytes:
    """Generate bytes after a byte prompt. The high-level public API.

    Args:
        model: A :class:`bhanox.model.Bhanox`.
        prompt: The bytes to continue. At least ``BYTE_GRAM_N`` of them, so that
            there is at least one context to condition on.
        max_new: How many bytes to generate.
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        seed: RNG seed. Fixed seed means byte-identical output, which is what
            makes generation testable at all.
        reset: Clear the recurrent state before starting, so a prompt cannot
            silently inherit state from an earlier call. Pass ``False`` to
            continue the sequence the model is already in the middle of.

    Returns:
        ``bytes``: the prompt followed by the generated bytes. This is not
        guaranteed to be valid UTF-8 -- the model emits byte values, and it is
        untrained if you have not trained it.

    Raises:
        TypeError: If ``prompt`` is not bytes-like. A ``str`` is refused on
            purpose rather than encoded here: the return value is bytes too, and
            an API that quietly converts text in would invite reading the output
            as text.
        ValueError: If ``max_new`` is negative, ``prompt`` is shorter than one
            context, or the model's output head is not one class per byte.
    """
    _check_count(max_new)
    if not isinstance(prompt, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"generate() takes a bytes prompt, got {type(prompt).__name__}. "
            "Encode it yourself -- generate(model, 'hi'.encode()). The output "
            "is bytes and is not guaranteed to be valid UTF-8, so there is no "
            "text wrapper here on purpose."
        )
    raw = bytes(prompt)
    if len(raw) < BYTE_GRAM_N:
        raise ValueError(
            f"a prompt needs at least {BYTE_GRAM_N} bytes to form one context, "
            f"got {len(raw)}"
        )
    _check_head(model)
    if max_new == 0:
        # A no-op query: the prompt unchanged, and the model not touched.
        return raw
    if reset:
        model.reset()
    new = _run(
        model,
        encode_bytes(raw, n=BYTE_GRAM_N),
        raw[-BYTE_GRAM_N:],
        max_new=max_new,
        temperature=temperature,
        seed=seed,
    )
    return raw + bytes(new)


def generate_ids(
    model: Bhanox,
    ids: NDArray[np.integer],
    *,
    max_new: int = 64,
    temperature: float = 0.8,
    seed: int | None = None,
    reset: bool = True,
) -> NDArray[np.uint8]:
    """Generate from context ids directly. The low-level door.

    Inputs are *input-space* ids -- packed byte 4-grams, as
    :func:`bhanox.frontend.encode` produces -- and the output is *output-space*
    byte values, sampled from the 256-way head. The two spaces are different and
    this function never puts them in one array: it returns only the generated
    bytes, and the caller keeps its own prompt ids. That is the difference from
    the API this replaced, which returned prompt ids followed by byte values in
    a single ``int64`` array with nothing in the type saying so.

    Each id is rolled forward the same way :func:`generate` rolls a byte window,
    so the loop stays coherent; the ids therefore have to be real 4-grams. If
    your ids are some other token space, they are not bytes, and rolling them
    produces a different-but-defined stream rather than a continuation of your
    tokens. Use the byte API for anything whose meaning you care about.

    Args:
        model: A :class:`bhanox.model.Bhanox`.
        ids: Prompt context ids, any shape. Flattened to a 1-D sequence. Each
            must fit in ``BYTE_GRAM_N`` bytes, because a context is a window and
            is unpacked as one.
        max_new: How many bytes to generate.
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        seed: RNG seed, for reproducible sampling.
        reset: Clear the recurrent state before starting. Pass ``False`` to
            continue the sequence the model is already in the middle of.

    Returns:
        ``uint8`` array of shape ``(max_new,)``: the generated byte values only.

    Raises:
        ValueError: If ``max_new`` is negative, ``ids`` is empty, an id does not
            fit in a context, or the output head is not one class per byte.
    """
    _check_count(max_new)
    prompt = np.asarray(ids, dtype=np.int64).reshape(-1)
    if prompt.size == 0:
        raise ValueError("generate_ids needs a non-empty prompt")
    limit = 1 << (8 * BYTE_GRAM_N)
    if int(prompt.min()) < 0 or int(prompt.max()) >= limit:
        raise ValueError(
            f"generate_ids takes 4-gram ids, so every id must be in "
            f"0..{limit - 1}; got {int(prompt.min())}..{int(prompt.max())}"
        )
    _check_head(model)
    if max_new == 0:
        return np.zeros(0, dtype=np.uint8)
    if reset:
        model.reset()
    new = _run(
        model,
        prompt,
        int(prompt[-1]).to_bytes(BYTE_GRAM_N, BYTE_GRAM_ORDER),
        max_new=max_new,
        temperature=temperature,
        seed=seed,
    )
    return np.asarray(new, dtype=np.uint8)
