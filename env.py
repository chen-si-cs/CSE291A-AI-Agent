"""
ArcEnv: the main simulator environment for AI agents to solve ARC-AGI puzzles
by composing DSL operations.

Gym-like API:
    obs = env.reset(puzzle_id)
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from puzzle_db import PuzzleDB, Puzzle
from dsl_engine import DSLEngine, DSLExecutionError, DSLTimeout
from inventory import Inventory, infer_type_label
from renderer import (
    render_grid, render_diff, render_value,
    render_train_example, render_inventory_summary,
    render_available_functions,
)
from reward import (
    compute_reward, step_penalty, invalid_action_penalty,
    budget_exceeded_penalty,
)
from text_parser import parse_command, HELP_TEXT


class ArcEnv:
    """
    Text-adventure-style environment for solving ARC-AGI puzzles with DSL operations.

    The agent has an inventory of named variables. It calls DSL functions,
    inspects results, and submits a final Grid answer.
    """

    def __init__(
        self,
        data_dirs: Optional[List[str]] = None,
        max_steps: int = 20,
        step_cost: float = 0.01,
        render_mode: str = "text",       # "text" | "ansi" | "json"
        allow_train_exec: bool = True,   # allow executing DSL on train inputs
    ):
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.render_mode = render_mode
        self.allow_train_exec = allow_train_exec

        # Load puzzles
        self.puzzle_db = PuzzleDB(data_dirs)

        # DSL engine
        self.engine = DSLEngine()

        # Episode state (initialized on reset)
        self.puzzle: Optional[Puzzle] = None
        self.test_index: int = 0
        self.inventory = Inventory()
        self.steps_taken: int = 0
        self.total_reward: float = 0.0
        self.done: bool = True
        self._episode_log: List[dict] = []

    # ════════════════════════════════════════════════════════════════
    #  Core API
    # ════════════════════════════════════════════════════════════════

    def reset(
        self,
        puzzle_id: Optional[str] = None,
        test_index: int = 0
    ) -> dict:
        """
        Start a new episode.

        Args:
            puzzle_id: specific puzzle, or None for random.
            test_index: which test pair (usually 0).

        Returns:
            Initial observation dict.
        """
        if puzzle_id:
            self.puzzle = self.puzzle_db.get(puzzle_id)
        else:
            self.puzzle = self.puzzle_db.random()

        self.test_index = test_index
        self.steps_taken = 0
        self.total_reward = 0.0
        self.done = False
        self._episode_log = []

        # Initialize inventory with test input
        self.inventory = Inventory()
        test_input = self.puzzle.test_input(test_index)
        self.inventory.set("I", test_input, provenance="test_input", step_number=0)

        # Optionally add train inputs
        if self.allow_train_exec:
            for i in range(self.puzzle.num_train):
                name = f"train_in_{i}"
                self.inventory.set(
                    name,
                    self.puzzle.train_input(i),
                    provenance=f"train_input[{i}]",
                    step_number=0,
                )
                name_out = f"train_out_{i}"
                self.inventory.set(
                    name_out,
                    self.puzzle.train_output(i),
                    provenance=f"train_output[{i}]",
                    step_number=0,
                )

        return self._build_observation(initial=True)

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """
        Execute one action.

        Args:
            action: an action dict (see README for formats).
                    Can also be a raw string (will be parsed).

        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return self._build_observation(), 0.0, True, {"error": "Episode is done"}

        # Parse string commands
        if isinstance(action, str):
            action = parse_command(action)

        action_type = action.get("type", "error")
        reward = 0.0
        info = {"action": action}

        try:
            if action_type == "execute":
                reward, info = self._handle_execute(action)
            elif action_type == "inspect":
                reward, info = self._handle_inspect(action)
            elif action_type == "submit":
                reward, info = self._handle_submit(action)
            elif action_type == "train_inspect":
                reward, info = self._handle_train_inspect(action)
            elif action_type == "undo":
                reward, info = self._handle_undo()
            elif action_type == "reset_inventory":
                reward, info = self._handle_reset_inventory()
            elif action_type == "list_functions":
                reward, info = self._handle_list_functions(action)
            elif action_type == "help":
                reward, info = 0.0, {"message": HELP_TEXT}
            elif action_type == "inventory":
                reward, info = self._handle_inventory_list()
            elif action_type == "error":
                reward = invalid_action_penalty()
                info = {"error": action.get("message", "Parse error")}
            else:
                reward = invalid_action_penalty()
                info = {"error": f"Unknown action type: '{action_type}'"}
        except Exception as e:
            reward = invalid_action_penalty()
            info = {"error": f"Unexpected error: {type(e).__name__}: {e}"}

        # Step cost (only for execute and submit)
        if action_type in ("execute", "submit"):
            self.steps_taken += 1
            reward += step_penalty(self.step_cost)

        # Budget check
        if not self.done and self.steps_taken >= self.max_steps:
            self.done = True
            reward += budget_exceeded_penalty()
            info["budget_exceeded"] = True

        self.total_reward += reward
        info["total_reward"] = self.total_reward
        info["steps_taken"] = self.steps_taken

        self._episode_log.append({
            "step": self.steps_taken,
            "action": action,
            "reward": reward,
        })

        return self._build_observation(action_result=info), reward, self.done, info

    def step_text(self, command: str) -> Tuple[dict, float, bool, dict]:
        """Convenience: parse a text command and step."""
        return self.step(parse_command(command))

    # ════════════════════════════════════════════════════════════════
    #  Action handlers
    # ════════════════════════════════════════════════════════════════

    def _handle_execute(self, action: dict) -> Tuple[float, dict]:
        func_name = action.get("function", "")
        raw_args = action.get("args", [])
        store_as = action.get("store_as", None)

        if not self.engine.has_function(func_name):
            # Fallback: check if func_name is an inventory variable holding a callable.
            # This handles solver patterns like `x3 = x2(I)` where x2 stores a
            # composed/partial function from chain(), compose(), lbind(), etc.
            if self.inventory.has(func_name):
                stored_val = self.inventory.get(func_name)
                if callable(stored_val):
                    try:
                        resolved_args = self.engine.resolve_args(raw_args, self.inventory)
                        result = stored_val(*resolved_args)
                    except Exception as e:
                        return invalid_action_penalty(), {
                            "error": f"Error calling inventory callable '{func_name}': {e}"
                        }
                    # Store result (same logic as normal execute path below)
                    if store_as is None:
                        store_as = self.inventory.next_name()
                    arg_strs = []
                    for raw in raw_args:
                        if isinstance(raw, bool):
                            arg_strs.append(str(raw))
                        elif isinstance(raw, str):
                            arg_strs.append(raw)
                        else:
                            arg_strs.append(repr(raw))
                    provenance = f"{func_name}({', '.join(arg_strs)})"
                    slot = self.inventory.set(
                        store_as, result,
                        provenance=provenance,
                        step_number=self.steps_taken + 1,
                    )
                    return 0.0, {
                        "action_ok": True,
                        "stored": store_as,
                        "type": slot.type_label,
                        "preview": slot.type_label,
                        "provenance": provenance,
                        "message": f"✓ {store_as} = {provenance}  →  {slot.type_label}",
                    }
            return invalid_action_penalty(), {
                "error": f"Unknown function: '{func_name}'",
                "suggestion": self._suggest_function(func_name),
            }

        # Resolve arguments
        try:
            resolved_args = self.engine.resolve_args(raw_args, self.inventory)
        except (KeyError, ValueError) as e:
            return invalid_action_penalty(), {"error": str(e)}

        # Execute
        try:
            result = self.engine.execute(func_name, resolved_args)
        except DSLTimeout:
            return invalid_action_penalty(), {"error": f"Timeout executing {func_name}"}
        except DSLExecutionError as e:
            return invalid_action_penalty(), {"error": str(e)}

        # Store result
        if store_as is None:
            store_as = self.inventory.next_name()

        # Build provenance string
        arg_strs = []
        for raw in raw_args:
            if isinstance(raw, bool):
                arg_strs.append(str(raw))
            elif isinstance(raw, str):
                arg_strs.append(raw)
            else:
                arg_strs.append(repr(raw))
        provenance = f"{func_name}({', '.join(arg_strs)})"

        slot = self.inventory.set(
            store_as, result,
            provenance=provenance,
            step_number=self.steps_taken + 1,
        )

        return 0.0, {
            "action_ok": True,
            "stored": store_as,
            "type": slot.type_label,
            "preview": slot.type_label,
            "provenance": provenance,
            "message": f"✓ {store_as} = {provenance}  →  {slot.type_label}",
        }

    def _handle_inspect(self, action: dict) -> Tuple[float, dict]:
        target = action.get("target", "")
        if not self.inventory.has(target):
            return 0.0, {"error": f"Variable '{target}' not in inventory"}

        slot = self.inventory.get_slot(target)
        detail = render_value(
            slot.value, slot.type_label,
            colored=(self.render_mode == "ansi"),
        )

        return 0.0, {
            "target": target,
            "type": slot.type_label,
            "provenance": slot.provenance,
            "detail": detail,
            "message": f"── {target} ({slot.type_label}) ──\n{detail}",
        }

    def _handle_submit(self, action: dict) -> Tuple[float, dict]:
        answer_var = action.get("answer", "")
        if not self.inventory.has(answer_var):
            return invalid_action_penalty(), {
                "error": f"Variable '{answer_var}' not in inventory"
            }

        submitted = self.inventory.get(answer_var)
        expected = self.puzzle.test_output(self.test_index)

        reward, is_correct, score_info = compute_reward(
            submitted, expected,
            self.steps_taken, self.max_steps,
            step_cost=self.step_cost,
        )

        self.done = True

        if is_correct:
            msg = (f"🎉 CORRECT! Exact match in {self.steps_taken} steps. "
                   f"Reward: {reward:.3f}")
        else:
            acc = score_info.get("accuracy", 0)
            msg = (f"✗ Incorrect. Accuracy: {acc:.1%} "
                   f"({score_info.get('correct_cells', 0)}/{score_info.get('total_cells', 0)} cells). "
                   f"Reward: {reward:.3f}")

        return reward, {
            "success": is_correct,
            "message": msg,
            **score_info,
        }

    def _handle_train_inspect(self, action: dict) -> Tuple[float, dict]:
        index = action.get("index", 0)
        which = action.get("which", "both")

        if index < 0 or index >= self.puzzle.num_train:
            return 0.0, {
                "error": f"Train index {index} out of range (0..{self.puzzle.num_train - 1})"
            }

        pair = self.puzzle.train[index]
        colored = (self.render_mode == "ansi")

        if which == "both":
            detail = render_train_example(pair, index, colored=colored)
        elif which == "input":
            grid = tuple(tuple(r) for r in pair["input"])
            detail = f"Train {index} Input:\n{render_grid(grid, colored=colored)}"
        elif which == "output":
            grid = tuple(tuple(r) for r in pair["output"])
            detail = f"Train {index} Output:\n{render_grid(grid, colored=colored)}"
        elif which == "diff":
            inp = tuple(tuple(r) for r in pair["input"])
            out = tuple(tuple(r) for r in pair["output"])
            detail = f"Train {index} Diff:\n{render_diff(inp, out)}"
        else:
            detail = render_train_example(pair, index, colored=colored)

        return 0.0, {"message": detail}

    def _handle_undo(self) -> Tuple[float, dict]:
        removed = self.inventory.undo()
        if removed:
            return 0.0, {"message": f"✓ Removed variable '{removed}'"}
        return 0.0, {"message": "Nothing to undo."}

    def _handle_reset_inventory(self) -> Tuple[float, dict]:
        keep = {"I"}
        if self.allow_train_exec:
            for i in range(self.puzzle.num_train):
                keep.add(f"train_in_{i}")
                keep.add(f"train_out_{i}")
        self.inventory.reset(keep=keep)
        return 0.0, {"message": "✓ Inventory reset (kept I and train data)."}

    def _handle_list_functions(self, action: dict) -> Tuple[float, dict]:
        filter_str = action.get("filter", "")
        catalog = self.engine.function_catalog(filter_str)
        detail = render_available_functions(catalog, filter_str)
        count = len(catalog)
        return 0.0, {
            "message": f"DSL Functions ({count} {'matching' if filter_str else 'total'}):\n{detail}",
            "count": count,
        }

    def _handle_inventory_list(self) -> Tuple[float, dict]:
        summary = self.inventory.summary()
        detail = render_inventory_summary(summary)
        return 0.0, {
            "message": f"Inventory ({len(self.inventory)} variables):\n{detail}",
        }

    # ════════════════════════════════════════════════════════════════
    #  Observation builder
    # ════════════════════════════════════════════════════════════════

    def _build_observation(self, initial: bool = False,
                           action_result: Optional[dict] = None) -> dict:
        """Build the observation dict returned to the agent."""
        obs = {
            "puzzle_id": self.puzzle.puzzle_id if self.puzzle else None,
            "turn": self.steps_taken,
            "budget_remaining": self.max_steps - self.steps_taken,
            "done": self.done,
            "inventory": self.inventory.summary(),
        }

        if initial:
            # Include train examples overview
            train_summaries = []
            for i in range(self.puzzle.num_train):
                pair = self.puzzle.train[i]
                inp = tuple(tuple(r) for r in pair["input"])
                out = tuple(tuple(r) for r in pair["output"])
                train_summaries.append({
                    "index": i,
                    "input_shape": f"{len(inp)}x{len(inp[0])}",
                    "output_shape": f"{len(out)}x{len(out[0])}",
                    "input_grid": render_grid(inp, colored=(self.render_mode == "ansi")),
                    "output_grid": render_grid(out, colored=(self.render_mode == "ansi")),
                    "diff": render_diff(inp, out),
                })
            obs["train_examples"] = train_summaries

            test_inp = self.puzzle.test_input(self.test_index)
            obs["test_input"] = {
                "shape": f"{len(test_inp)}x{len(test_inp[0])}",
                "grid": render_grid(test_inp, colored=(self.render_mode == "ansi")),
            }

        if action_result:
            obs["last_action_result"] = action_result

        return obs

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════

    def _suggest_function(self, bad_name: str) -> str:
        """Suggest similar function names."""
        names = self.engine.function_names()
        close = [n for n in names if bad_name.lower() in n.lower()
                 or n.lower() in bad_name.lower()]
        if close:
            return f"Did you mean: {', '.join(close[:5])}?"
        return ""

    def get_episode_log(self) -> List[dict]:
        return list(self._episode_log)

    def get_expected_output(self) -> tuple:
        """Peek at expected output (for debugging/evaluation, not for agent)."""
        return self.puzzle.test_output(self.test_index)

    def render_expected(self) -> str:
        """Render expected output (for debugging only)."""
        return render_grid(self.get_expected_output(),
                           colored=(self.render_mode == "ansi"))
