"""Byte budgeting for append-only slot tables.

Spec D7 wants both of the model's memories to be explicit, bounded, settable
ceilings. The temporary state has a fixed shape, so all it needs is a check. The
vault is the one a caller resizes while the model runs, and resizing an
append-only slot table has two ways to go wrong, both of which are handled here
rather than in :mod:`bhanox.memory.vectorvault`.

The first is admitting a budget the arrays cannot honour. The slot tables are
allocated up front, so a budget that merely stopped writes would still hold
1.5 MB resident while reporting 64 KB. :meth:`ByteBudgetMixin.set_budget`
resizes the tables, which makes the reported number the number the process pays.

The second is treating a ceiling as a purchase order. Sizing the table to the
budget means ``set_budget("4GB")`` allocates 4 GB, which is the out-of-memory
failure the spec is trying to prevent. The table is therefore
``min(entry_cap, budget limit)``: the budget caps, and ``entry_cap`` decides how
much is really reserved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bhanox.budgets import parse_bytes, perm_entry_bytes


class ByteBudgetMixin:
    """Byte-ceiling budgeting for a vault with preallocated slot tables.

    The host class supplies the slot tables and the counters; see the
    annotations below for the contract. They are declared, not assigned, so the
    mixin documents the attributes it needs without owning them.
    """

    bits: int
    d_value: int
    n_slots: int
    filled: int
    clock: int
    perm_budget_bytes: int | None
    entry_cap: int
    keys: NDArray[np.uint8]
    values: NDArray[np.floating]
    salience: NDArray[np.float32]
    age: NDArray[np.int64]

    def _drop_where(self, drop: NDArray[np.bool_]) -> None:
        """Remove entries where ``drop`` is True. Supplied by the host."""
        raise NotImplementedError

    @property
    def entry_bytes(self) -> int:
        """Bytes one occupied entry costs: key, value, and bookkeeping."""
        return perm_entry_bytes(self.bits, self.d_value)

    @property
    def used_bytes(self) -> int:
        """Permanent bytes in use, counting only occupied slots.

        This is the number a budget is checked against, as opposed to
        :attr:`nbytes`, which is what the preallocation cost whether or not the
        slots are used. A vault that has admitted three entries is not using
        1024 entries' worth of memory.
        """
        return self.filled * self.entry_bytes

    @property
    def nbytes(self) -> int:
        """Bytes actually resident: the slot tables as currently sized.

        This is what the process pays, and after a budget resize it tracks the
        budget. :attr:`used_bytes` is the smaller "data admitted" figure.
        """
        return int(
            self.keys.nbytes
            + self.values.nbytes
            + self.salience.nbytes
            + self.age.nbytes
        )

    def budget_entries(self) -> int | None:
        """Entries the byte budget allows, or ``None`` if uncapped.

        This is the budget's own limit, not the current table size. A budget
        looser than the cap is not binding, and the vault should then behave
        exactly as it did before budgets existed.
        """
        if self.perm_budget_bytes is None:
            return None
        return self.perm_budget_bytes // self.entry_bytes

    def max_admissible_entries(self) -> int:
        """Entries that may be held: the entry cap, tightened by any budget."""
        allowed = self.budget_entries()
        return self.entry_cap if allowed is None else min(self.entry_cap, allowed)

    def set_budget(self, budget: str | int | None) -> None:
        """Cap the permanent memory, resizing the slot tables to fit.

        Live and idempotent, per spec D7: this is the one memory a caller can
        resize while the model runs. Lowering the budget evicts the least
        important entries and shrinks the tables rather than raising, because
        running out of room must cost recall, not the process. A budget too
        small to hold even one entry is not an error -- it truncates to nothing,
        since an exception here would hand the caller a crash where the whole
        point of a memory is to be the thing that gives way.

        Args:
            budget: ``"2GB"``, an int, or ``None`` to lift the cap.

        Raises:
            ValueError: If the budget cannot be parsed.
        """
        self.perm_budget_bytes = parse_bytes(budget)
        self._truncate_to_budget()
        self._resize_to_budget()

    def set_entry_cap(self, n_slots: int) -> None:
        """Change how many entries the vault may hold, live.

        This is the ``E`` knob from spec D7, distinct from the byte budget: E
        is a count, the budget is a ceiling, and E wins when it is tighter.
        Growing E allocates immediately; the new slots start empty.

        Args:
            n_slots: The new entry cap. Must be at least one.

        Raises:
            ValueError: If ``n_slots`` is not positive, or is below the number
                of entries already stored. Shrinking below that would silently
                discard data; :meth:`set_budget` truncates by importance instead.
        """
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        if n_slots < self.filled:
            raise ValueError(
                f"cannot cap at {n_slots} entries while holding {self.filled}. "
                "Use set_budget to truncate by importance score instead."
            )
        self.entry_cap = int(n_slots)
        self._resize_to_budget()

    def _truncate_to_budget(self) -> int:
        """Evict least-important entries until the budget is met.

        Returns:
            How many entries were evicted.

        Why importance order: eviction uses the same salience x recency score as
        the vault's own slot selection, so shrinking the budget drops exactly
        the entries it would have overwritten anyway. Truncating in slot order
        instead would discard recent, important memories just because they sit
        at a high index.
        """
        allowed = self.budget_entries()
        if allowed is None or self.filled <= allowed:
            return 0
        recency = (self.age.astype(np.float64) + 1.0) / (self.clock + 1.0)
        score = self.salience.astype(np.float64) * recency
        # argpartition then sort the winners, so ties break by slot order. With
        # `allowed == 0` the selection is empty and everything is dropped.
        top = np.argpartition(-score, allowed - 1)[:allowed]
        keep = np.zeros(self.n_slots, dtype=bool)
        keep[np.sort(top)] = True
        evicted = int(self.filled - allowed)
        self._drop_where(~keep)
        return evicted

    def _resize_to_budget(self) -> None:
        """Resize the slot tables to ``min(entry_cap, budget limit)``.

        Lifting the budget restores the tables to the cap, since the cap is what
        the budget had shrunk them away from. Entries past the new size are
        dropped, but :meth:`_truncate_to_budget` has already removed the
        unimportant ones, so what is lost here was never worth keeping.
        """
        allowed = self.budget_entries()
        target = self.entry_cap if allowed is None else min(self.entry_cap, allowed)
        if target == self.n_slots:
            return
        shape_for: dict[str, tuple[int, ...]] = {
            "keys": (target, self.bits // 8),
            "values": (target, self.d_value),
            "salience": (target,),
            "age": (target,),
        }
        for name, dtype in (
            ("keys", np.uint8),
            ("values", np.float32),
            ("salience", np.float32),
            ("age", np.int64),
        ):
            resized = np.zeros(shape_for[name], dtype=dtype)
            keep = min(self.filled, target)
            resized[:keep] = getattr(self, name)[:keep]
            setattr(self, name, resized)
        self.n_slots = target
