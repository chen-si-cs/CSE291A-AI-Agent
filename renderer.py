"""
Renderer: converts DSL values into human-readable text representations.

Supports plain text and ANSI-colored output for terminal play.
"""

from __future__ import annotations
from typing import Any, List, Tuple, Optional


# ARC color palette → ANSI 256-color codes
_COLOR_ANSI = {
    0: "\033[48;5;0m\033[38;5;244m",    # black bg, gray text
    1: "\033[48;5;21m\033[38;5;15m",     # blue
    2: "\033[48;5;196m\033[38;5;15m",    # red
    3: "\033[48;5;28m\033[38;5;15m",     # green
    4: "\033[48;5;226m\033[38;5;0m",     # yellow
    5: "\033[48;5;244m\033[38;5;15m",    # gray
    6: "\033[48;5;199m\033[38;5;15m",    # magenta
    7: "\033[48;5;208m\033[38;5;0m",     # orange
    8: "\033[48;5;39m\033[38;5;0m",      # light blue
    9: "\033[48;5;88m\033[38;5;15m",     # maroon
}
_RESET = "\033[0m"


def render_grid(grid: tuple, colored: bool = False, indent: int = 2) -> str:
    """Render a Grid (tuple of tuples of ints) as a text block."""
    if not grid or not isinstance(grid, tuple):
        return "<empty grid>"
    h, w = len(grid), len(grid[0])
    prefix = " " * indent
    lines = [f"{prefix}Grid({h}x{w}):"]
    for row in grid:
        if colored:
            cells = []
            for v in row:
                ansi = _COLOR_ANSI.get(v, "")
                cells.append(f"{ansi} {v} {_RESET}")
            lines.append(prefix + "".join(cells))
        else:
            lines.append(prefix + " ".join(f"{v}" for v in row))
    return "\n".join(lines)


def render_diff(input_grid: tuple, output_grid: tuple, indent: int = 2) -> str:
    """Show which cells changed between input and output."""
    if not input_grid or not output_grid:
        return "<cannot diff>"
    h_in, w_in = len(input_grid), len(input_grid[0])
    h_out, w_out = len(output_grid), len(output_grid[0])
    prefix = " " * indent
    changes = []
    if (h_in, w_in) != (h_out, w_out):
        changes.append(f"Shape changed: ({h_in}x{w_in}) → ({h_out}x{w_out})")
    h, w = min(h_in, h_out), min(w_in, w_out)
    for i in range(h):
        for j in range(w):
            if input_grid[i][j] != output_grid[i][j]:
                changes.append(
                    f"({i},{j}): {input_grid[i][j]} → {output_grid[i][j]}"
                )
    if not changes:
        return f"{prefix}No changes."
    if len(changes) > 30:
        shown = changes[:25]
        shown.append(f"... and {len(changes) - 25} more changes")
        changes = shown
    return "\n".join(f"{prefix}{c}" for c in changes)


def render_value(value: Any, type_label: str = "",
                 colored: bool = False, max_detail: int = 40) -> str:
    """Render any DSL value for the inspect action."""
    if type_label.startswith("Grid"):
        return render_grid(value, colored=colored)

    if type_label == "Callable":
        name = getattr(value, '__name__', None) or "lambda"
        return f"  Callable: <{name}>"

    if type_label == "Boolean":
        return f"  Boolean: {value}"

    if type_label == "Integer":
        return f"  Integer: {value}"

    if type_label == "IntegerTuple":
        return f"  IntegerTuple: {value}"

    if type_label.startswith("Object("):
        cells = sorted(value, key=lambda c: (c[1][0], c[1][1]))
        lines = [f"  Object ({len(value)} cells):"]
        for v, (i, j) in cells[:max_detail]:
            lines.append(f"    color={v} @ ({i},{j})")
        if len(cells) > max_detail:
            lines.append(f"    ... ({len(cells) - max_detail} more)")
        return "\n".join(lines)

    if type_label.startswith("Objects("):
        lines = [f"  Objects ({len(value)} objects):"]
        for idx, obj in enumerate(sorted(value, key=len, reverse=True)):
            if idx >= 8:
                lines.append(f"    ... ({len(value) - 8} more)")
                break
            colors = {v for v, _ in obj}
            lines.append(f"    [{idx}] {len(obj)} cells, colors={colors}")
        return "\n".join(lines)

    if type_label.startswith("Indices("):
        pts = sorted(value)
        lines = [f"  Indices ({len(value)} points):"]
        for p in pts[:max_detail]:
            lines.append(f"    ({p[0]},{p[1]})")
        if len(pts) > max_detail:
            lines.append(f"    ... ({len(pts) - max_detail} more)")
        return "\n".join(lines)

    if type_label.startswith("IntegerSet"):
        return f"  IntegerSet: {sorted(value)}"

    if type_label.startswith("Tuple("):
        lines = [f"  Tuple ({len(value)} items):"]
        for idx, item in enumerate(value[:max_detail]):
            lines.append(f"    [{idx}] {repr(item)[:100]}")
        if len(value) > max_detail:
            lines.append(f"    ... ({len(value) - max_detail} more)")
        return "\n".join(lines)

    if type_label.startswith("FrozenSet"):
        return f"  FrozenSet({len(value)}): {repr(value)[:200]}"

    return f"  {type_label}: {repr(value)[:200]}"


def render_train_example(pair: dict, index: int,
                         colored: bool = False) -> str:
    """Render one train input/output pair."""
    inp = _to_grid_tuple(pair["input"])
    out = _to_grid_tuple(pair["output"])
    lines = [f"═══ Train Example {index} ═══"]
    lines.append("Input:")
    lines.append(render_grid(inp, colored=colored))
    lines.append("Output:")
    lines.append(render_grid(out, colored=colored))
    lines.append("Changes:")
    lines.append(render_diff(inp, out))
    return "\n".join(lines)


def render_inventory_summary(summary: dict) -> str:
    """Render the inventory summary dict as text."""
    if not summary:
        return "  (empty)"
    lines = []
    for name, info in summary.items():
        prov = f"  = {info['provenance']}" if info['provenance'] else ""
        lines.append(f"  {name:6s} : {info['type']:20s} │ {info['preview']}{prov}")
    return "\n".join(lines)


def render_available_functions(funcs: dict, filter_str: str = "") -> str:
    """Render DSL function list, optionally filtered."""
    lines = []
    for name, info in sorted(funcs.items()):
        if filter_str and filter_str.lower() not in name.lower():
            continue
        sig = info.get("signature", "")
        doc = info.get("doc", "")
        lines.append(f"  {name}{sig}  — {doc}")
    if not lines:
        return "  No matching functions."
    return "\n".join(lines)


def _to_grid_tuple(grid_list) -> tuple:
    """Convert list-of-lists (from JSON) to tuple-of-tuples."""
    if isinstance(grid_list, tuple):
        return grid_list
    return tuple(tuple(row) for row in grid_list)
