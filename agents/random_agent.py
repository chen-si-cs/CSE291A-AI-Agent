"""
RandomAgent: picks random DSL functions with random arguments.

This is a baseline to show the environment works. It will almost never
solve a puzzle, but demonstrates the action space.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List

from agents.base_agent import BaseAgent


# Common constants the random agent can use as literal arguments
_LITERALS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    -1, -2,
    True, False,
    (0, 0), (1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1),
    (2, 2), (3, 3),
]

# Functions that take 1 arg (common subset)
_UNARY = [
    "hmirror", "vmirror", "dmirror", "rot90", "rot180", "rot270",
    "height", "width", "shape", "palette", "numcolors", "mostcolor",
    "leastcolor", "tophalf", "bottomhalf", "lefthalf", "righthalf",
    "trim", "compress", "normalize", "toindices", "asobject",
    "flip", "invert", "double", "halve", "increment", "decrement",
    "size", "first", "last", "totuple", "portrait", "square",
    "ulcorner", "lrcorner", "centerofmass", "center",
    "backdrop", "delta", "inbox", "outbox", "box", "corners",
    "frontiers", "partition", "fgpartition", "asindices",
    "identity", "color", "merge", "dedupe",
]

# Functions that take 2 args
_BINARY = [
    "fill", "ofcolor", "colorfilter", "sizefilter", "replace",
    "hconcat", "vconcat", "recolor", "shift", "crop",
    "upscale", "downscale", "hsplit", "vsplit",
    "compose", "lbind", "rbind", "matcher",
    "add", "subtract", "multiply",
    "combine", "intersection", "difference",
    "equality", "greater", "contained",
    "hmatching", "vmatching", "adjacent", "manhattan",
    "connect", "insert", "remove",
]


class RandomAgent(BaseAgent):
    """Agent that takes random actions — useful as a baseline."""

    def __init__(self, max_steps_before_submit: int = 8):
        self.max_steps = max_steps_before_submit
        self.step_count = 0
        self.inv_vars: List[str] = []

    def setup(self, observation: dict) -> None:
        self.step_count = 0
        self.inv_vars = list(observation.get("inventory", {}).keys())

    def act(self, observation: dict) -> dict:
        self.inv_vars = list(observation.get("inventory", {}).keys())
        self.step_count += 1

        # After enough steps, submit whatever we have
        if self.step_count >= self.max_steps:
            # Submit last variable or I
            answer = self.inv_vars[-1] if self.inv_vars else "I"
            return {"type": "submit", "answer": answer}

        # Random: pick a unary or binary function
        if random.random() < 0.6 and self.inv_vars:
            # Unary
            func = random.choice(_UNARY)
            arg = random.choice(self.inv_vars)
            return {
                "type": "execute",
                "function": func,
                "args": [arg],
            }
        elif self.inv_vars:
            # Binary
            func = random.choice(_BINARY)
            arg1 = random.choice(self.inv_vars)
            arg2 = random.choice(self.inv_vars + _LITERALS)
            return {
                "type": "execute",
                "function": func,
                "args": [arg1, arg2],
            }
        else:
            return {"type": "submit", "answer": "I"}

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        pass
