"""
LearningAgent: behavioral cloning with a trained small LM (Qwen3-1.7B) and optional lookup fallback.

Train the model with scripts/train_bc_model.py on offline_trajectories.json.
Then load the checkpoint via model_path; at act() the model is called to generate the next command.
If no model_path is given, or model output fails to parse, falls back to lookup table (from
data_path) or random agent.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from text_parser import parse_command
from observation_format import format_observation_for_prompt
import random


# Prompt prefix/suffix used in train_bc_model.py so generation matches training
PROMPT_PREFIX = "Observation:\n"
ACTION_PREFIX = "\nAction: "


def _fix_json_action(action: dict) -> dict:
    """Fix JSON round-trip damage: convert lists back to tuples in action args.

    JSON has no tuple type, so (1, -1) serializes as [1, -1].  DSL functions
    expect tuples, so we must convert back when loading from JSON.
    """
    if "args" not in action:
        return action
    action = dict(action)  # shallow copy
    action["args"] = [_list_to_tuple(a) for a in action["args"]]
    return action


def _list_to_tuple(val):
    """Recursively convert lists to tuples (for JSON-deserialized action args)."""
    if isinstance(val, list):
        return tuple(_list_to_tuple(v) for v in val)
    return val


class LearningAgent(BaseAgent):
    """
    BC agent: uses a trained Qwen3-1.7B to generate next action from observation.
    Falls back to (puzzle_id, step) lookup or random when model is not loaded or fails.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        fallback_max_steps_before_submit: int = 15,
    ):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._lookup: Dict[tuple, dict] = {}
        self._fallback = RandomAgent(max_steps_before_submit=fallback_max_steps_before_submit)
        self._current_trajectory: List[tuple] = []

        if data_path and os.path.isfile(data_path):
            self.load_offline_data(data_path)
        if model_path and os.path.isdir(model_path):
            self._load_model()

    def _load_model(self) -> None:
        """Lazy-load tokenizer and model from model_path (requires transformers, torch)."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(
                "LearningAgent with model_path requires: pip install torch transformers"
            ) from e
        path = self.model_path or ""
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=getattr(torch, "bfloat16", torch.float32) if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def load_offline_data(self, path: str) -> int:
        """Load trajectories from JSON into lookup table (fallback). Returns count loaded."""
        if not path or not os.path.isfile(path):
            return 0
        with open(path) as f:
            trajectories = json.load(f)
        return self.train_offline(trajectories)

    def train_offline(self, trajectories: List[dict]) -> int:
        """Build lookup (puzzle_id, step) -> action from trajectory list. Returns count."""
        n = 0
        for traj in trajectories:
            pid = traj.get("puzzle_id")
            for i, s in enumerate(traj.get("steps", [])):
                step_idx = s.get("step", i)
                action = s.get("action")
                if pid and action is not None:
                    self._lookup[(pid, step_idx)] = _fix_json_action(action)
                    n += 1
        return n

    def setup(self, observation: dict) -> None:
        self._fallback.setup(observation)
        self._current_trajectory = []

    def act(self, observation: dict) -> dict:
        puzzle_id = observation.get("puzzle_id")
        turn = observation.get("turn", 0)
        key = (puzzle_id, turn)

        # 1. Try trained model if loaded
        if self._model is not None and self._tokenizer is not None:
            action = self._act_with_model(observation)
            if action is not None and action.get("type") != "error":
                self._current_trajectory.append((puzzle_id, turn, action))
                return action

        # 2. Fallback: lookup from offline trajectories
        if key in self._lookup:
            action = self._lookup[key]
            self._current_trajectory.append((puzzle_id, turn, action))
            return action

        # 3. Random fallback
        action = self._fallback.act(observation)
        self._current_trajectory.append((puzzle_id, turn, action))
        return action

    def _act_with_model(self, observation: dict) -> Optional[dict]:
        """Format obs, generate with model, parse response. Returns action dict or None on failure."""
        if self._model is None or self._tokenizer is None:
            return None
        import torch
        prompt_text = format_observation_for_prompt(observation)
        full_prompt = PROMPT_PREFIX + prompt_text + ACTION_PREFIX
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.strip().split("\n")[0].strip()
        action = parse_command(gen)
        if action.get("type") == "error":
            return None
        return action

    def on_step_result(self, observation: dict, reward: float, done: bool, info: dict) -> None:
        self._fallback.on_step_result(observation, reward, done, info)

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        self._fallback.on_episode_end(total_reward, steps, info)

    def train_online(self, env, n_episodes: int = 50, puzzle_ids: Optional[List[str]] = None, verbose: bool = False) -> dict:
        """Run episodes; on success, cache trajectory in lookup (self-imitation). Returns stats."""
        ids = puzzle_ids or getattr(env, "puzzle_db", None)
        if ids is not None and hasattr(ids, "ids"):
            ids = ids.ids()
        ids = ids or []
        if not ids:
            return {"episodes": 0, "solved": 0, "cached_steps": 0}
        solved = 0
        cached_before = len(self._lookup)
        for _ in range(n_episodes):
            pid = random.choice(ids)
            obs = env.reset(puzzle_id=pid)
            self.setup(obs)
            done = False
            while not done:
                action = self.act(obs)
                obs, reward, done, info = env.step(action)
                self.on_step_result(obs, reward, done, info)
                if done:
                    self.on_episode_end(info.get("total_reward", 0), info.get("steps_taken", 0), info)
                    if info.get("exact_match"):
                        solved += 1
                    break
        if verbose:
            print(f"[LearningAgent] train_online: {n_episodes} episodes, solved={solved}, cached +{len(self._lookup) - cached_before} steps")
        return {"episodes": n_episodes, "solved": solved, "cached_steps": len(self._lookup) - cached_before}

    def save_lookup(self, path: str) -> None:
        """Save lookup table to JSON (for fallback only)."""
        out = {f"{pid}|{step}": action for (pid, step), action in self._lookup.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)

    def load_lookup(self, path: str) -> int:
        """Load lookup table from JSON. Returns count."""
        if not path or not os.path.isfile(path):
            return 0
        with open(path) as f:
            out = json.load(f)
        n = 0
        for k, action in out.items():
            if "|" in k:
                pid, step = k.split("|", 1)
                try:
                    self._lookup[(pid, int(step))] = action
                    n += 1
                except ValueError:
                    pass
        return n
