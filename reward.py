"""
Reward computation for the ARC DSL simulator.

Supports exact match, partial (cell-level) scoring, and step penalties.
"""

from __future__ import annotations
from typing import Tuple


def compute_reward(
    submitted: tuple,
    expected: tuple,
    steps_taken: int,
    max_steps: int,
    step_cost: float = 0.01,
    wrong_shape_penalty: float = 0.1,
) -> Tuple[float, bool, dict]:
    """
    Compute the reward for a submission.

    Returns:
        (reward, is_correct, info_dict)
    """
    info = {}

    # Validate that submitted is a grid
    if not _is_grid(submitted):
        return (
            -wrong_shape_penalty,
            False,
            {"error": "Submitted value is not a valid Grid (tuple of tuples of ints)"}
        )

    sub_h, sub_w = len(submitted), len(submitted[0]) if submitted else 0
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0

    info["submitted_shape"] = (sub_h, sub_w)
    info["expected_shape"] = (exp_h, exp_w)

    # Shape mismatch
    if (sub_h, sub_w) != (exp_h, exp_w):
        info["shape_match"] = False
        return (-wrong_shape_penalty, False, info)

    info["shape_match"] = True

    # Cell-by-cell comparison
    total_cells = sub_h * sub_w
    correct_cells = 0
    for i in range(sub_h):
        for j in range(sub_w):
            if submitted[i][j] == expected[i][j]:
                correct_cells += 1

    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    info["accuracy"] = accuracy
    info["correct_cells"] = correct_cells
    info["total_cells"] = total_cells

    is_correct = (accuracy == 1.0)
    info["exact_match"] = is_correct

    if is_correct:
        # Full reward, with small bonus for efficiency
        efficiency_bonus = max(0, (max_steps - steps_taken) / max_steps) * 0.1
        reward = 1.0 + efficiency_bonus
    else:
        # Partial credit based on accuracy
        reward = accuracy * 0.5  # partial credit capped at 0.5

    info["reward"] = reward
    return (reward, is_correct, info)


def step_penalty(step_cost: float = 0.01) -> float:
    """Per-step penalty to encourage efficiency."""
    return -step_cost


def invalid_action_penalty() -> float:
    """Penalty for an invalid action (type error, bad function, etc.)."""
    return -0.05


def budget_exceeded_penalty() -> float:
    """Penalty when the agent exceeds the step budget."""
    return -0.5


def _is_grid(value) -> bool:
    """Check if a value looks like a valid Grid."""
    if not isinstance(value, tuple) or len(value) == 0:
        return False
    first_row = value[0]
    if not isinstance(first_row, tuple) or len(first_row) == 0:
        return False
    # Check all rows are tuples of the same length with int values
    w = len(first_row)
    for row in value:
        if not isinstance(row, tuple) or len(row) != w:
            return False
        for v in row:
            if not isinstance(v, int):
                return False
    return True
