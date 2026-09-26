"""Benchmark harness for the P1 speed gate (law C10).

Usage::

    python scripts/benchmark.py --config nano --tokens 32 --json

What it measures, and what it does not:

* It times the **NumPy reference**. That number is a correctness harness and a
  regression guard. It is **not** the P1 claim.
* P1 is ``>= 20,000 tokens/s`` on the **native** runtime, which lands in M4. The
  native path does int8 work directly out of a fixed buffer; the reference pays
  for float32 temporaries and a Python-level loop per token per layer. Expect
  one to two orders of magnitude between them.
* So this script reports the reference figure labelled ``reference``, and reports
  P1 as ``NOT MEASURED``. Anything else would be an estimate dressed as a
  measurement, which law A3 forbids.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhanox.audit import audit_bytes_per_token
from bhanox.config import available_presets, load_config
from bhanox.model import Bhanox

#: The claim P1 will be judged against, in the M4 native runtime.
P1_TARGET_TOKENS_PER_S = 20_000
P1_MILESTONE = "M4"


def benchmark(config: str, tokens: int, seed: int) -> dict[str, object]:
    """Time autoregressive stepping for one preset.

    Args:
        config: Preset name.
        tokens: How many tokens to generate.
        seed: RNG seed for the token stream, so runs are comparable.

    Returns:
        A dict of measured figures. Every value here was timed or counted by
        this function.
    """
    cfg = load_config(config)
    model = Bhanox(cfg)
    rng = np.random.default_rng(seed)

    prompt = rng.integers(0, cfg.output_vocab, 4).astype(np.int64)
    model.reset()
    from bhanox.generate import generate

    start = time.perf_counter()
    out = generate(model, prompt, max_new=tokens, temperature=0.0)
    elapsed = time.perf_counter() - start

    report = audit_bytes_per_token(model)
    rate = tokens / elapsed if elapsed > 0 else float("inf")
    return {
        "config": config,
        "backend": "numpy-reference",
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": rate,
        "parameters": model.param_count(),
        "state_bytes": model.state_nbytes(),
        "bytes_per_token_per_layer": report.layer_total,
        "l2_utilization": report.utilization,
        "i2_passed": report.passed,
        "generated_tokens": int(out.size),
        "p1_target_tokens_per_second": P1_TARGET_TOKENS_PER_S,
        "p1_status": f"NOT MEASURED until {P1_MILESTONE} (native runtime)",
    }


def main() -> int:
    """Parse arguments, run the benchmark, print JSON. Returns a shell exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="nano", choices=available_presets())
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    result = benchmark(args.config, args.tokens, args.seed)
    if not args.json:
        print(
            f"# {platform.python_implementation()} "
            f"{platform.python_version()} on {platform.system()}"
        )
    print(json.dumps(result, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
