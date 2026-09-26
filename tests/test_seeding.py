"""Reproducible init, and init streams that do not collide.

The bug this pins: every projection used to be seeded with
``abs(hash((site, cfg.name))) % 2**32``. ``hash()`` on a str is randomised per
process, so the read-out, bypass, unembed and MoE weights were different on
every run -- and no in-process test could catch it, because within one process
``hash("nano")`` is perfectly stable. That is what made it survive.

The same missing index made every head of a layer draw one stream, so
``n_heads`` independent memories were byte-identical copies that stayed
identical for the whole forward pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import bhanox

REPO = Path(__file__).resolve().parents[1]

# Runs in a child process. Prints one digest over the weights that were
# previously un-reproducible, plus one over the per-head weights that were
# previously identical.
CHILD = """
import hashlib
import numpy as np
import bhanox

m = bhanox.Bhanox(bhanox.load_config("nano"))
h = hashlib.sha256()
for a in (m.deltabanks[0].W_o, m.deltabanks[0].G, m.mixers[0].E, m.output):
    h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
print("bulk", h.hexdigest())
print("heads", " ".join(
    format(float(x.W_k.sum()), ".6f") for x in m.deltabanks[0].heads
))
"""


def _run(hashseed: str) -> tuple[str, str]:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    ).stdout.split()
    return out[1], " ".join(out[3:])


class TestReproducible:
    def test_same_weights_in_different_processes(self) -> None:
        """The whole point: two processes, two hash seeds, one model.

        If this fails, something is seeding an RNG from ``hash()`` again.
        """
        a_bulk, a_heads = _run("0")
        b_bulk, b_heads = _run("1")
        assert a_bulk == b_bulk, "weights differ across PYTHONHASHSEED"
        assert a_heads == b_heads, "per-head weights differ across PYTHONHASHSEED"

    def test_two_instances_agree(self) -> None:
        cfg = bhanox.load_config("nano")
        x, y = bhanox.Bhanox(cfg), bhanox.Bhanox(cfg)
        assert np.array_equal(x.deltabanks[0].W_o, y.deltabanks[0].W_o)
        assert np.array_equal(x.output, y.output)

    def test_seed_actually_changes_the_model(self) -> None:
        """A seed you cannot change is not a seed."""
        base = bhanox.load_config("nano")
        other = replace(base, seed=base.seed + 1)
        assert not np.array_equal(
            bhanox.Bhanox(base).deltabanks[0].W_o,
            bhanox.Bhanox(other).deltabanks[0].W_o,
        )

    def test_same_shape_different_name_differs(self) -> None:
        """Two presets of identical shape must not share weights."""
        base = bhanox.load_config("nano")
        a = bhanox.Bhanox(base)
        b = bhanox.Bhanox(replace(base, name="nano_copy"))
        assert not np.array_equal(a.deltabanks[0].W_o, b.deltabanks[0].W_o)


class TestStreamsDoNotCollide:
    def test_heads_are_not_copies(self) -> None:
        heads = bhanox.Bhanox(bhanox.load_config("nano")).deltabanks[0].heads
        for field in ("W_k", "W_q", "W_v", "W_r"):
            first = getattr(heads[0], field)
            for other in heads[1:]:
                assert not np.array_equal(
                    first, getattr(other, field)
                ), f"{field} is identical across heads"

    def test_heads_stay_distinct_through_a_forward_pass(self) -> None:
        """Distinct weights are not enough; distinct states are the claim.

        Identical weights plus identical decay schedules means identical states
        forever, which is what actually happened.
        """
        m = bhanox.Bhanox(bhanox.load_config("nano"))
        m.reset()
        for tok in (3, 9, 14, 22, 31, 40):
            m.step(tok)
        states = [h.state for h in m.deltabanks[0].heads]
        for other in states[1:]:
            assert not np.array_equal(states[0], other)

    def test_layers_are_not_copies(self) -> None:
        m = bhanox.Bhanox(bhanox.load_config("nano"))
        assert len(m.mixers) > 1
        first = m.mixers[0].E
        for other in m.mixers[1:]:
            assert not np.array_equal(first, other.E)


def test_digest_is_not_empty() -> None:
    """Cheap guard that the child script actually produced output."""
    bulk, heads = _run("0")
    assert len(bulk) == 64, "child did not print a sha256"
    assert len(heads.split()) == 4, f"expected 4 head sums, got {heads!r}"
