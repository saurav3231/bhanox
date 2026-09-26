# Contributing

## Before you push

```bash
pip install -e ".[dev]"
black .
ruff check .
mypy
pytest
```

All four are CI gates. If any of them needs a suppression to make your change
land, that is a signal worth writing down rather than a detail to skip past.

## The laws

These are the project's rules. They are enforced in CI, mostly by
`tests/test_codebase_limits.py` and `tests/test_invariants.py`, so breaking one
fails the build rather than starting an argument.

| law | rule |
|---|---|
| C1 | Black formatting, 88 columns. |
| C2 | Ruff and mypy clean. `disallow_untyped_defs` is on. |
| C5 | One concern per module. 500 lines is the ceiling, excluding docstrings. |
| C7 | Tests accompany the change. Coverage ratchets up, never down. |
| C8 | **NumPy is the only runtime dependency.** No exceptions. |
| C9 | Torch lives in `bhanox/train/` and nowhere else. It is an optional extra. |
| C10 | The P1 speed gate hooks into `scripts/benchmark.py`. |
| C11 | Examples are smoke tests. If it is in the README, it runs. |
| A3 | Every number is `measured` or `modeled`. Never an unmarked estimate. |
| D2 | I1–I4 are release blockers. |

## Style that is not lintable

**Explain why, not what.** The code says what it does. A comment earns its place
by saying what a reader would otherwise get wrong — the trade-off, the sharp
edge, the thing that will break if someone "cleans it up". The repo has a lot of
these because the architecture has a lot of non-obvious decisions.

**Document the trap, not the mechanism.** When something is load-bearing in a way
that is easy to break, say so at the point of breakage. Two real examples:

- `HashBind.embed` needs `(arr >= 0) & (arr < vocab_table)`. Drop the lower bound
  and numpy indexes the table from the end, so `-1` silently reads the last row.
- `MicroExpertLayer` must store dequantized weights. Raw int8 codes give router
  logits with a spread of ~500, the softmax saturates, and the router picks the
  same two experts forever.

**Do not tune a number to make a test pass.** If a measurement is inconvenient,
report it. `small` fails invariant I2 at 337.5% of budget and that is the correct
output of `examples/audit_demo.py`.

**No new runtime dependency without an ADR.** NumPy only, and that is not a
starting position for negotiation.

## Tests

Test the mechanism, not the implementation. A test that asserts `x == 3` because
that is what the code does is worthless; a test that asserts a 3-gram and a
4-gram do not collide is worth keeping.

Some specific expectations:

- A gate that cannot fail is not a gate. Every invariant test has a companion
  that constructs a violating case and asserts it raises.
- When you pin a number, say in the test that it is a regression guard, not a
  target. If the number moves, the *claim* moved.
- Prefer a property to an example where both are available.

## Pull requests

One concern per commit. Describe what you changed and, more importantly, what you
measured. If you found a bug in something else, say so rather than fixing it
quietly in the same diff.
