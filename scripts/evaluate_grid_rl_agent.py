#!/usr/bin/env python3
"""
Evaluate GridRLAgent and compare against baselines.

Baselines:
  random   -- picks a macro-action uniformly at random each step
  rl_agent -- runs the RLAgent (agents/rl_agent.py) for direct comparison

Usage:
    python -m scripts.evaluate_grid_rl_agent \\
        --checkpoint checkpoints/grid_rl/ckpt_final.pt \\
        --data data/train --n 50 --compare-all
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.grid_rl_agent import GridRLAgent, NUM_MACROS, get_macro_actions
from scripts.train_grid_rl_agent import run_episode


# ── Random baseline ───────────────────────────────────────────────────────────

class RandomMacroAgent:
    """Picks a macro-action uniformly at random every step."""
    def setup(self, obs): pass
    def act(self, obs):
        idx = random.randint(0, NUM_MACROS - 1)
        self._last_idx = idx
        self._last_obs = obs
        steps = get_macro_actions(obs)[idx]
        return steps[0] if steps else {"type": "inspect", "target": "I"}
    def get_macro_steps(self, obs):
        return get_macro_actions(self._last_obs)[self._last_idx]
    def on_step_result(self, *a, **k): pass
    def on_episode_end(self, *a, **k): pass
    def reinforce_update(self): return 0.0


# ── RLAgent baseline wrapper ─────────────────────────────────────────────────

class RLAgentWrapper:
    """
    Wraps the RLAgent so it runs in the same evaluation loop.
    Uses its checkpoint if available, else uses random weights.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, max_steps: int = 20):
        from agents.rl_agent import RLAgent
        self._agent = RLAgent(max_steps=max_steps,
                              checkpoint_path=checkpoint_path)

    def setup(self, obs):
        self._agent.setup(obs)

    def act(self, obs):
        return self._agent.act(obs)

    def get_macro_steps(self, obs):
        # takes single DSL actions, not macros — return a 1-step list
        return [self._agent.act(obs)]

    def on_step_result(self, obs, r, done, info):
        self._agent.on_step_result(obs, r, done, info)

    def on_episode_end(self, *a):
        pass

    def reinforce_update(self):
        return 0.0


def run_RLagent_episode(env, agent: RLAgentWrapper,
                         puzzle_id: str) -> tuple:
    """Run one episode with the RLagent (single-action steps)."""
    from agents.rl_agent import RLAgent
    obs  = env.reset(puzzle_id)
    agent.setup(obs)
    total_reward = 0.0
    n_steps      = 0
    done         = False
    info         = {}

    while not done and obs.get("budget_remaining", 0) > 0:
        action = agent._agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        n_steps      += 1
        agent._agent.on_step_result(obs, reward, done, info)

    puzzle_solved = info.get("success", False) and not info.get("budget_exceeded", False)
    return total_reward, puzzle_solved, n_steps


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(env, agent, puzzle_ids, label: str,
             is_rlagent=False, verbose=False) -> List[Dict]:
    results = []
    t0 = time.time()
    for i, pid in enumerate(puzzle_ids):
        try:
            if is_rlagent:
                reward, success, n = run_RLagent_episode(env, agent, pid)
            else:
                reward, success, n = run_episode(env, agent, pid)
            r = {"puzzle_id": pid, "success": success,
                 "reward": reward, "steps": n}
            results.append(r)
            if verbose:
                tick = "✓" if success else "✗"
                print(f"  [{i+1:3d}/{len(puzzle_ids)}] [{tick}] {pid:<20} "
                      f"r={reward:.3f}")
        except Exception as e:
            print(f"  ERROR {pid}: {e}")
            results.append({"puzzle_id": pid, "success": False,
                             "reward": 0.0, "steps": 0})

    n        = len(results)
    n_solved = sum(r["success"] for r in results)
    avg_r    = sum(r["reward"]  for r in results) / max(n, 1)
    print(f"\n  {label}")
    print(f"  Solved:     {n_solved}/{n} ({n_solved/max(n,1):.1%})")
    print(f"  Avg reward: {avg_r:.4f}")
    print(f"  Time:       {time.time()-t0:.1f}s")
    return results


def print_comparison(all_results: Dict[str, List[Dict]]):
    agents = list(all_results.keys())
    by     = {a: {r["puzzle_id"]: r for r in all_results[a]} for a in agents}
    common = set.intersection(*[set(d) for d in by.values()])
    n      = len(common)
    if not common:
        return

    print(f"\n{'='*62}")
    print(f"  COMPARISON  ({n} puzzles)")
    print(f"{'='*62}")
    print(f"  {'Metric':<18}" + "".join(f" {a:>14}" for a in agents))
    print(f"  {'-'*58}")

    def pct(a): return sum(1 for p in common if by[a][p]["success"]) / n
    def avr(a): return sum(by[a][p]["reward"] for p in common) / n

    print(f"  {'Solve rate':<18}" + "".join(f" {pct(a):>13.1%}" for a in agents))
    print(f"  {'Avg reward':<18}" + "".join(f" {avr(a):>13.4f}" for a in agents))
    print(f"{'='*62}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--data",  nargs="+", default=["data/train"])
    parser.add_argument("--n",     type=int,  default=None)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--compare-all",     action="store_true")
    parser.add_argument("--rl-agent-ckpt",   default=None,
                        help="Path to RLAgent checkpoint .pt")
    parser.add_argument("--output",          default=None)
    parser.add_argument("--verbose", "-v",   action="store_true")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    env = ArcEnv(data_dirs=args.data, max_steps=args.budget)
    puzzle_ids = env.puzzle_db.ids()
    if args.n:
        puzzle_ids = puzzle_ids[:args.n]
    print(f"Evaluating on {len(puzzle_ids)} puzzles\n")

    all_results = {}

    # GridRLAgent
    agent = GridRLAgent(deterministic=True)
    agent.load(args.checkpoint)
    print("="*50)
    all_results["grid_rl"] = evaluate(
        env, agent, puzzle_ids, "GridRLAgent (trained)", verbose=args.verbose
    )

    if args.compare_all:
        print("\n" + "="*50)
        rand_agent = RandomMacroAgent()
        all_results["random"] = evaluate(
            env, rand_agent, puzzle_ids, "Random macro baseline"
        )

        print("\n" + "="*50)
        rl_agent = RLAgentWrapper(
            checkpoint_path=args.rl_agent_ckpt, max_steps=args.budget
        )
        all_results["rl_agent"] = evaluate(
            env, rl_agent, puzzle_ids, "RLAgent",
            is_rlagent=True
        )

        print_comparison(all_results)

    # Show what the agent would do on a few puzzles
    print("\n=== Sample macro selections (qualitative) ===")
    from agents.grid_rl_agent import _describe_macro
    for pid in puzzle_ids[:3]:
        obs = env.reset(pid)
        top = agent.top_macros(obs, k=3)
        print(f"\n  Puzzle: {pid}")
        for idx, prob, desc in top:
            print(f"    #{idx} p={prob:.3f}  {desc}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()