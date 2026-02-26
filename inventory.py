"""
Inventory: a named-variable store that holds intermediate DSL results.

Each slot records the value, its inferred type label, and how it was produced
(for replay / explanation).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict


@dataclass
class Slot:
    """One inventory entry."""
    name: str
    value: Any
    type_label: str          # human-readable, e.g. "Grid", "Objects(5)"
    provenance: str = ""     # e.g. "objects(I, True, False, True)"
    step_number: int = 0


class Inventory:
    """Ordered named-variable store with undo support."""

    def __init__(self):
        self._slots: OrderedDict[str, Slot] = OrderedDict()
        self._history: List[str] = []   # stack of slot names for undo
        self._counter: int = 0          # for auto-naming x1, x2, ...

    # ── core operations ─────────────────────────────────────────────

    def set(self, name: str, value: Any, provenance: str = "",
            step_number: int = 0) -> Slot:
        type_label = infer_type_label(value)
        slot = Slot(
            name=name,
            value=value,
            type_label=type_label,
            provenance=provenance,
            step_number=step_number,
        )
        self._slots[name] = slot
        self._history.append(name)
        return slot

    def get(self, name: str) -> Any:
        if name not in self._slots:
            raise KeyError(f"Variable '{name}' not in inventory")
        return self._slots[name].value

    def get_slot(self, name: str) -> Slot:
        if name not in self._slots:
            raise KeyError(f"Variable '{name}' not in inventory")
        return self._slots[name]

    def has(self, name: str) -> bool:
        return name in self._slots

    def next_name(self) -> str:
        """Auto-generate the next variable name (x1, x2, ...)."""
        self._counter += 1
        name = f"x{self._counter}"
        while name in self._slots:
            self._counter += 1
            name = f"x{self._counter}"
        return name

    def undo(self) -> Optional[str]:
        """Remove the most recently added slot (except 'I'). Returns removed name."""
        while self._history:
            name = self._history.pop()
            if name == "I":
                continue
            if name in self._slots:
                del self._slots[name]
                return name
        return None

    def reset(self, keep: Optional[set] = None):
        """Clear everything except the names in `keep`."""
        keep = keep or {"I"}
        to_remove = [k for k in self._slots if k not in keep]
        for k in to_remove:
            del self._slots[k]
        self._history = [n for n in self._history if n in self._slots]
        self._counter = 0

    # ── queries ─────────────────────────────────────────────────────

    def names(self) -> List[str]:
        return list(self._slots.keys())

    def summary(self) -> Dict[str, dict]:
        """Return a lightweight summary of every slot (for observations)."""
        out = {}
        for name, slot in self._slots.items():
            out[name] = {
                "type": slot.type_label,
                "provenance": slot.provenance,
                "preview": _preview(slot.value, slot.type_label),
            }
        return out

    def __len__(self):
        return len(self._slots)

    def __contains__(self, name):
        return name in self._slots

    def __repr__(self):
        lines = []
        for n, s in self._slots.items():
            lines.append(f"  {n}: {s.type_label}  ({s.provenance})")
        return "Inventory(\n" + "\n".join(lines) + "\n)"


# ── type inference helpers ──────────────────────────────────────────────

def infer_type_label(value: Any) -> str:
    """Return a human-readable type string for a DSL value."""
    if callable(value) and not isinstance(value, (frozenset, tuple)):
        return "Callable"
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Integer"
    if isinstance(value, tuple):
        if len(value) == 0:
            return "Tuple(0)"
        first = value[0]
        # Grid: tuple of tuples of ints
        if isinstance(first, tuple) and len(first) > 0 and isinstance(first[0], int):
            return f"Grid({len(value)}x{len(first)})"
        # IntegerTuple
        if isinstance(first, int) and len(value) == 2:
            return "IntegerTuple"
        # Tuple of other things
        return f"Tuple({len(value)})"
    if isinstance(value, frozenset):
        if len(value) == 0:
            return "FrozenSet(0)"
        sample = next(iter(value))
        # Object: frozenset of (int, (int, int))
        if isinstance(sample, tuple) and len(sample) == 2:
            val, idx = sample
            if isinstance(val, int) and isinstance(idx, tuple):
                return f"Object({len(value)})"
            # Indices: frozenset of (int, int)
            if isinstance(val, int) and isinstance(idx, int):
                return f"Indices({len(value)})"
            # Objects: frozenset of frozensets
            if isinstance(val, frozenset) or isinstance(idx, frozenset):
                pass  # fall through
        if isinstance(sample, frozenset):
            return f"Objects({len(value)})"
        if isinstance(sample, int):
            return f"IntegerSet({len(value)})"
        return f"FrozenSet({len(value)})"
    return type(value).__name__


def _preview(value: Any, type_label: str, max_len: int = 120) -> str:
    """Short text preview of a value."""
    if type_label == "Callable":
        name = getattr(value, '__name__', None) or "lambda"
        return f"<fn:{name}>"
    if type_label.startswith("Grid"):
        rows = value
        lines = []
        for r in rows[:3]:
            lines.append(" ".join(str(v) for v in r[:12]))
        if len(rows) > 3:
            lines.append("...")
        return " | ".join(lines)
    if type_label.startswith("Object("):
        cells = sorted(value, key=lambda c: (c[1][0], c[1][1]))[:6]
        parts = [f"{v}@({i},{j})" for v, (i, j) in cells]
        if len(value) > 6:
            parts.append("...")
        return ", ".join(parts)
    if type_label.startswith("Objects("):
        return f"{len(value)} objects"
    if type_label.startswith("Indices("):
        pts = sorted(value)[:6]
        parts = [f"({i},{j})" for i, j in pts]
        if len(value) > 6:
            parts.append("...")
        return ", ".join(parts)
    s = repr(value)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s
