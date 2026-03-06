"""
Format observation dict as text for LM prompt (BC training and inference).
Works for both live env observation and serialized trajectory observation.
"""
from __future__ import annotations
import json
from typing import Any, Dict


def format_observation_for_prompt(obs: Dict[str, Any]) -> str:
    """Format observation dict as text for the model (same format for train and act)."""
    lines = [
        f"Puzzle: {obs.get('puzzle_id', '?')}",
        f"Budget: {obs.get('budget_remaining', '?')} steps",
        "",
    ]
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
        lines.append("Respond with ONE command per turn (e.g. execute objects(I, True, False, True) -> x1, submit O).")
        lines.append("")
    last = obs.get("last_action_result") or {}
    if last:
        msg = last.get("message")
        if not msg:
            msg = json.dumps(last, default=str)
        lines.append(f"Result: {msg}")
    lines.append(f"Budget remaining: {obs.get('budget_remaining', '?')} steps.")
    lines.append("")
    return "\n".join(lines)
