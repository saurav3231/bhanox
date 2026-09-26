"""Reproducible weight-initialisation streams.

Every projection in the reference draws from its own stream, keyed by
``(seed, site, name, index)``. Two properties follow, and both are load-bearing:

- **Reproducible.** The same config gives the same weights in any process, on
  any machine, forever. A checkpoint is only meaningful if the model that reads
  it is the model that wrote it.
- **Distinct.** Two sites that would otherwise share a shape do not share a
  stream, so ``n_heads`` independent memories really are independent.

Why this module exists at all: the streams used to be keyed by
``abs(hash((site, cfg.name))) % 2**32``. Python randomises string hashing per
process (``PYTHONHASHSEED``), so every run drew fresh weights for the read-out,
the bypass, the unembed and the MoE -- the model was not reproducible, and no
test could have caught it, because every "expected" value was equally random.
The heads escaped only by accident: their key was a tuple of ``int``, which
hashes stably. That accident hid the second bug -- the key had no head index,
so all four heads of a layer drew the *same* stream and stayed byte-identical
forever. "Independent memories" was four copies of one.

So: never seed an RNG from ``hash()`` on a str here. Use :func:`init_rng`.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["init_rng"]


def init_rng(seed: int, site: str, name: str, index: int = 0) -> np.random.Generator:
    """Return the init stream for one projection site.

    Args:
        seed: The config seed. Changing it gives a different model of the same
            shape, which is what an ablation or a sweep wants.
        site: What is being initialised, e.g. ``"db"`` or ``"unembed"``. Keeps
            two projections of identical shape on different streams.
        name: Config name, so two presets with the same shape still differ.
        index: Which copy this is -- head index, layer index. This is the part
            that makes ``n_heads`` mean anything.

    Returns:
        A fresh :class:`numpy.random.Generator`. 64 bits of key material, taken
        from BLAKE2b rather than ``hash()`` for the reason in the module
        docstring.
    """
    key = f"{seed}\x00{site}\x00{name}\x00{index}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))
