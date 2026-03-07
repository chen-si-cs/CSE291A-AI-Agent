"""
Format observation dict as text for LM prompt (BC training and inference).
Works for both live env observation and serialized trajectory observation.

Key design: includes inventory state (types + provenance) at every step so the
model knows what variables exist and how they were created.
"""
from __future__ import annotations
import json
from typing import Any, Dict


def format_observation_for_prompt(obs: Dict[str, Any]) -> str:
    """Format observation dict as text for the model (same format for train and act)."""
    lines = [
        f"Puzzle: {obs.get('puzzle_id', '?')}",
        f"Step: {obs.get('turn', 0)} / Budget: {obs.get('budget_remaining', '?')} remaining",
        "",
    ]

    # ── Train examples + test input (grid context) ──────────────
    train_examples = obs.get("train_examples") or []
    if train_examples:
        for ex in train_examples:
            lines.append(f"--- Train Example {ex.get('index', 0)} ---")
            lines.append(f"Input ({ex.get('input_shape', '')}):")
            lines.append(ex.get("input_grid", ""))
            lines.append(f"Output ({ex.get('output_shape', '')}):")
            lines.append(ex.get("output_grid", ""))
            lines.append(ex.get("diff", ""))
            lines.append("")
        test = obs.get("test_input") or {}
        lines.append(f"--- Test Input ({test.get('shape', '?')}) ---")
        lines.append(test.get("grid", ""))
        lines.append("")

    # ── Inventory state (provenance chain) ──────────────────────
    # Critical: tells the model what variables exist,
    # their types, and how they were created.
    inventory = obs.get("inventory") or {}
    if inventory:
        lines.append("Inventory:")
        for name, info in inventory.items():
            if isinstance(info, dict):
                type_label = info.get("type", "?")
                prov = info.get("provenance", "")
                prov_str = f" = {prov}" if prov else ""
                lines.append(f"  {name}: {type_label}{prov_str}")
            else:
                lines.append(f"  {name}: {info}")
        lines.append("")

    # ── Last action result ──────────────────────────────────────
    last = obs.get("last_action_result") or {}
    if last:
        msg = last.get("message")
        if not msg:
            msg = json.dumps(last, default=str)
        lines.append(f"Result: {msg}")
        lines.append("")

    lines.append("Respond with ONE command (e.g. execute func(args) -> var, or submit O).")
    lines.append("")
    return "\n".join(lines)