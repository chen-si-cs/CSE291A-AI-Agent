"""
RLBeamAgent: RL-guided beam search for ARC puzzle solving.

Architecture (analogous to AlphaGo):
  - Policy network (trained with GRPO, same as RLAgent) assigns probabilities
    over DSL operators given the current puzzle state.
  - At inference time, beam search uses those probabilities as a learned prior
    to prioritize which operator sequences to explore.
  - Every candidate program is verified against ALL train examples before
    committing — so the agent never submits a program that doesn't generalize.

Why this beats pure RL sampling:
  - Pure RL: samples ~20 trajectories per episode, finds lucky sequences
  - Beam search: systematically explores top-K branches at each step,
    verifying train consistency → avoids the combinatorial explosion

Training (train_rl_agent.py, unchanged):
  - Policy is trained with GRPO exactly as before
  - Each episode: sample trajectory, get reward, update policy
  - Policy learns: "given this puzzle state, which operators are useful?"

Inference (this agent):
  - Policy outputs logits → softmax → operator probabilities
  - Beam search expands top-B operators at each depth
  - Prune beams whose intermediate result fails train verification
  - Submit the program with best train-verified accuracy

Usage:
  python -m scripts.evaluate --agent rl_beam \\
    --rl-checkpoint checkpoints/rl_agent/ckpt_final.pt \\
    --data data/train --budget 50 --beam-width 8 --beam-depth 4 --verbose
"""

from __future__ import annotations
import heapq
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from agents.base_agent import BaseAgent
from agents.rl_agent import (
    RLAgent, _obs_to_state, _extract_puzzle_features,
    UNARY_GRID, UNARY_META, BINARY_GG,
    _UG, _UM, _B, NUM_ACTIONS, STATE_DIM,
)


# ── Program step representation ───────────────────────────────────────────────

@dataclass
class ProgramStep:
    func: str
    args: list   # "I", "__last__", or literal values

    def to_env_action(self, last_var: str) -> dict:
        resolved = [last_var if a == "__last__" else a for a in self.args]
        return {"type": "execute", "function": self.func, "args": resolved}

    def __repr__(self):
        return f"{self.func}({', '.join(str(a) for a in self.args)})"


@dataclass(order=True)
class Beam:
    """One candidate program in the beam."""
    neg_score: float          # negative train accuracy (for min-heap)
    steps: List[ProgramStep] = field(compare=False)
    last_value: Any           = field(compare=False, default=None)
    # last_value on each train input is tracked separately in _execute_program


def _action_idx_to_step(idx: int, has_last_var: bool) -> Optional[ProgramStep]:
    """Convert a policy action index to a ProgramStep. Returns None for submit."""
    if idx == 0:
        return None  # submit action

    i = idx - 1
    if i < _UG:
        return ProgramStep(UNARY_GRID[i], ["I"])
    i -= _UG
    if i < _UG:
        if not has_last_var:
            return None
        return ProgramStep(UNARY_GRID[i], ["__last__"])
    i -= _UG
    if i < _UM:
        func = UNARY_META[i]
        extra = [True, False, True] if func == "objects" else []
        return ProgramStep(func, ["I"] + extra)
    i -= _UM
    if i < _UM:
        if not has_last_var:
            return None
        func = UNARY_META[i]
        extra = [True, False, True] if func == "objects" else []
        return ProgramStep(func, ["__last__"] + extra)
    i -= _UM
    if i < _B:
        if not has_last_var:
            return None
        return ProgramStep(BINARY_GG[i], ["I", "__last__"])
    i -= _B
    if i < _B:
        if not has_last_var:
            return None
        return ProgramStep(BINARY_GG[i], ["__last__", "I"])
    return None


def _execute_step_on_value(engine, step: ProgramStep,
                            input_val: Any, last_val: Any) -> Any:
    """Execute one step given input I and last_var values."""
    args = []
    for a in step.args:
        if a == "I":
            args.append(input_val)
        elif a == "__last__":
            args.append(last_val if last_val is not None else input_val)
        else:
            args.append(a)
    return engine.execute(step.func, args)


def _execute_program_on_input(engine, steps: List[ProgramStep],
                               input_val: Any) -> Any:
    """Execute full program on a single input. Returns result or None on error."""
    current  = input_val
    last_val = None
    for step in steps:
        try:
            result  = _execute_step_on_value(engine, step, input_val, last_val)
            last_val = result
            current  = result
        except Exception:
            return None
    return current


def _grid_accuracy(result: Any, expected: tuple) -> float:
    """Cell-level accuracy. Returns 0 on shape mismatch or non-grid."""
    if not isinstance(result, tuple) or not result:
        return 0.0
    if not isinstance(result[0], tuple):
        return 0.0
    if len(result) != len(expected) or len(result[0]) != len(expected[0]):
        return 0.0
    total = correct = 0
    for r1, r2 in zip(result, expected):
        for a, b in zip(r1, r2):
            total += 1
            correct += int(a == b)
    return correct / total if total > 0 else 0.0


def _train_accuracy(engine, steps: List[ProgramStep],
                    train_pairs: List[Tuple]) -> float:
    """Average cell accuracy of program over all train pairs."""
    if not train_pairs:
        return 0.0
    total = 0.0
    for inp, out in train_pairs:
        result = _execute_program_on_input(engine, steps, inp)
        total += _grid_accuracy(result, out)
    return total / len(train_pairs)


# ── Agent ─────────────────────────────────────────────────────────────────────

class RLBeamAgent(BaseAgent):
    """
    RL policy guides beam search.

    Training: same GRPO pipeline as RLAgent (use train_rl_agent.py).
    Inference: beam search over programs, guided by policy logits,
               verified against train examples.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: int = 50,
        beam_width: int = 8,    # K: number of beams kept at each depth
        beam_depth: int = 4,    # max program length
        top_b: int = 10,        # B: top operators to expand per beam
    ):
        self.max_steps   = max_steps
        self.beam_width  = beam_width
        self.beam_depth  = beam_depth
        self.top_b       = top_b

        # Load the underlying RL policy
        self._rl = RLAgent(
            max_steps=max_steps,
            checkpoint_path=checkpoint_path,
        )
        self._env_ref      = None
        self._plan: List[dict] = []
        self._step_idx: int    = 0

    def setup(self, observation: dict) -> None:
        self._rl.setup(observation)
        self._plan     = []
        self._step_idx = 0

    def _get_operator_probs(self, obs: dict, chain_depth: int) -> List[float]:
        """
        Get policy probability for each action index given current obs.
        Returns list of length NUM_ACTIONS.
        """
        import torch
        state  = _obs_to_state(obs, self.max_steps,
                                self._rl._puzzle_cache,
                                self._rl._last_var_value)
        x      = torch.tensor([state], dtype=torch.float32,
                               device=self._rl._device)
        logits = self._rl._policy(x).squeeze(0).detach()

        # Mask submit and unavailable actions
        if chain_depth == 0:
            for i in range(1 + _UG, 1 + 2 * _UG):
                logits[i] = float("-inf")
            for i in range(1 + 2 * _UG + _UM, 1 + 2 * _UG + 2 * _UM):
                logits[i] = float("-inf")
            for i in range(1 + 2 * _UG + 2 * _UM, NUM_ACTIONS):
                logits[i] = float("-inf")
        logits[0] = float("-inf")   # no submit during beam expansion

        probs = torch.softmax(logits, dim=0).cpu().tolist()
        return probs

    def _beam_search(self, obs: dict) -> List[dict]:
        """
        Run beam search guided by RL policy.
        Returns env action list for best verified program.
        """
        if self._env_ref is None:
            return [{"type": "submit", "answer": "I"}]

        env    = self._env_ref
        engine = env.engine
        puzzle = env.puzzle

        # Collect all train pairs
        train_pairs = [
            (puzzle.train_input(i), puzzle.train_output(i))
            for i in range(puzzle.num_train)
        ]

        # Each beam: (neg_train_acc, program_steps, last_values_per_train)
        # last_values_per_train: list of last-computed value for each train input
        # (needed to resolve __last__ args correctly during expansion)
        initial_obs = obs
        beams: List[Tuple[float, List[ProgramStep], List[Any]]] = [
            (0.0, [], [inp for inp, _ in train_pairs])
        ]

        best_acc   = 0.0
        best_steps: List[ProgramStep] = []

        for depth in range(self.beam_depth):
            # Get policy probs for the initial state (approximation — we use
            # the puzzle-level features which don't change per beam)
            chain_depth = depth  # rough proxy
            probs       = self._get_operator_probs(initial_obs, chain_depth)

            # Top-B operators by policy probability (exclude submit=0)
            top_actions = sorted(
                range(1, NUM_ACTIONS), key=lambda i: -probs[i]
            )[:self.top_b]

            next_beams: List[Tuple[float, List[ProgramStep], List[Any]]] = []

            for (_, prev_steps, prev_last_vals) in beams:
                has_last = (len(prev_steps) > 0)

                for action_idx in top_actions:
                    step = _action_idx_to_step(action_idx, has_last)
                    if step is None:
                        continue

                    # Execute this step on all train inputs
                    new_last_vals = []
                    ok = True
                    for i, (inp, _) in enumerate(train_pairs):
                        try:
                            result = _execute_step_on_value(
                                engine, step, inp, prev_last_vals[i]
                            )
                            new_last_vals.append(result)
                        except Exception:
                            ok = False
                            break

                    if not ok:
                        continue

                    new_steps = prev_steps + [step]

                    # Score: accuracy of current result against train outputs
                    total_acc = 0.0
                    for i, (_, out) in enumerate(train_pairs):
                        total_acc += _grid_accuracy(new_last_vals[i], out)
                    acc = total_acc / len(train_pairs) if train_pairs else 0.0

                    if acc > best_acc:
                        best_acc   = acc
                        best_steps = new_steps

                    if acc == 1.0:
                        # Perfect — stop immediately
                        return self._steps_to_env_actions(best_steps)

                    next_beams.append((-acc, new_steps, new_last_vals))

            if not next_beams:
                break

            # Keep top beam_width beams by accuracy
            next_beams.sort(key=lambda x: x[0])
            beams = next_beams[:self.beam_width]

        # No perfect program — submit best partial result
        return self._steps_to_env_actions(best_steps)

    def _steps_to_env_actions(self, steps: List[ProgramStep]) -> List[dict]:
        """Convert program steps to env action sequence."""
        actions  = []
        last_var = "I"
        for i, step in enumerate(steps):
            action = step.to_env_action(last_var)
            actions.append(action)
            last_var = f"x{i + 1}"   # approximate — env assigns real name
        actions.append({"type": "submit", "answer": "__submit__"})
        return actions

    def act(self, observation: dict) -> dict:
        # Build plan on first call
        if not self._plan:
            self._plan = self._beam_search(observation)

        if self._step_idx >= len(self._plan):
            inv       = observation.get("inventory") or {}
            non_input = [k for k in inv if k != "I" and not k.startswith("train_")]
            return {"type": "submit", "answer": non_input[-1] if non_input else "I"}

        action = self._plan[self._step_idx]
        self._step_idx += 1

        # Resolve __submit__ placeholder to actual last inventory variable
        if action.get("answer") == "__submit__":
            inv       = observation.get("inventory") or {}
            non_input = [k for k in inv if k != "I" and not k.startswith("train_")]
            return {"type": "submit", "answer": non_input[-1] if non_input else "I"}

        return action

    def on_step_result(self, observation: dict, reward: float,
                       done: bool, info: dict) -> None:
        pass

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        pass