"""Audit every preset against invariants I2 and I3.

Run it with::

    python examples/audit_demo.py

Expect ``small`` to FAIL. That is the intended output: the audit reports what it
measures and does not tune the number away (ADR-003). If this script ever prints
all-PASS, the budgets changed and ADR-003 needs revisiting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bhanox
from bhanox.audit import audit_bytes_per_token


def main() -> int:
    """Audit every preset and print the residency reports. Returns an exit code."""
    print("Bhanox residency audit (I2) and op whitelist (I3)")
    print("=" * 72)
    for name in bhanox.available_presets():
        cfg = bhanox.load_config(name)
        model = bhanox.Bhanox(cfg)
        print()
        print(audit_bytes_per_token(model).summary())
        print()
        print(
            f"  {'parameters':<22}{model.param_count():>10,}  "
            f"packed {model.packed_nbytes() / 1e6:.2f} MB int8"
        )
        print(
            f"  {'recurrent state':<22}{model.state_nbytes():>10,} B  "
            f"constant in context"
        )

    print()
    print("=" * 72)
    print("P1 (>= 20,000 tokens/s) is NOT MEASURED until the M4 native runtime.")
    print("See docs/benchmarks.md for what is measured and what is modelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
