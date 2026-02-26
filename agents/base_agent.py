"""
BaseAgent: abstract interface for agents that interact with ArcEnv.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for ARC DSL simulator agents."""

    @abstractmethod
    def setup(self, observation: dict) -> None:
        """
        Called once at the start of each episode (after env.reset()).

        The observation contains train examples, the test input grid,
        and the initial inventory.
        """
        ...

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """
        Given the current observation, return an action dict.

        Action types: execute, inspect, submit, undo, reset_inventory,
                      list_functions, train_inspect, help, inventory.

        For execute:
            {
                "type": "execute",
                "function": "objects",
                "args": ["I", True, False, True],
                "store_as": "x1"
            }

        For submit:
            {"type": "submit", "answer": "O"}
        """
        ...

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        """
        Called when the episode ends (correct answer, wrong answer, or budget exceeded).
        Override for learning agents.
        """
        pass

    def on_step_result(self, observation: dict, reward: float,
                       done: bool, info: dict) -> None:
        """
        Called after each step with the result.
        Override for agents that need to react to feedback.
        """
        pass
