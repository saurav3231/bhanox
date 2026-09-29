"""Generate text from an untrained Bhanox and print the cost report.

This is the smoke test for the public API (law C11) and the quickest way to see
that the front door works end to end::

    python examples/generate_nano.py

The output is *not* language. Bhanox has no trained checkpoint until M2, so this
prints the shape of the interface and the measured cost of running it. Anything
that looked like English here would be a lie (law A3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bhanox


def main() -> int:
    """Continue a byte prompt, and report the cost. Returns an exit code."""
    cfg = bhanox.load_config("nano")
    model = bhanox.Bhanox(cfg)

    print("Bhanox quickstart")
    print("=" * 60)
    print(f"version       {bhanox.__version__}")
    print(f"preset        {cfg.name}")
    print(f"presets       {', '.join(bhanox.available_presets())}")
    print()

    prompt = b"the model remembers what it can fit"
    ids = bhanox.frontend.encode(prompt)
    print(f"prompt        {prompt!r}")
    print(f"encoded       {len(ids)} ids (4-gram byte windows)")
    print()

    out = bhanox.generate(model, prompt, max_new=16, temperature=0.8, seed=0)
    new = np.frombuffer(out[len(prompt) :], dtype=np.uint8)
    print(f"generated     {len(new)} bytes, {len(np.unique(new))} distinct")
    print(f"byte values   {new.tolist()}")
    print()

    print("Cost, all figures measured at call time:")
    print("-" * 60)
    print(bhanox.report(model))
    print()

    before = model.state_nbytes()
    bhanox.generate(model, prompt, max_new=256, temperature=0.8, seed=1)
    after = model.state_nbytes()
    print(
        f"state after 256 more bytes: {after:,} B (was {before:,} B) "
        "-- constant in context"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
