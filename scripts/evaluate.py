#!/usr/bin/env python3
"""
Evaluate an agent across multiple ARC puzzles.

Usage:
    python -m scripts.evaluate --agent random --data data/train --n 50
    python -m scripts.evaluate --agent random --puzzle 00d62c1b
"""

import argparse
import json
import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.random_agent import RandomAgent


def run_episode(env: ArcEnv, agent, puzzle_id=None, verbose=False):
    """Run one episode. Returns dict of metrics."""
    obs = env.reset(puzzle_id)
    agent.setup(obs)

    pid = obs["puzzle_id"]
    if verbose:
        print(f"  Puzzle: {pid}", end=" ", flush=True)

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        agent.on_step_result(obs, reward, done, info)

        if done:
            agent.on_episode_end(total_reward, steps, info)

    success = info.get("success", info.get("exact_match", False))
    accuracy = info.get("accuracy", 0.0)

    if verbose:
        status = "✓" if success else "✗"
        print(f"{status}  steps={steps}  reward={total_reward:.3f}  "
              f"accuracy={accuracy:.1%}")

    return {
        "puzzle_id": pid,
        "success": success,
        "accuracy": accuracy,
        "steps": steps,
        "total_reward": total_reward,
        "budget_exceeded": info.get("budget_exceeded", False),
    }


def make_agent(agent_name: str):
    """Factory for agents."""
    if agent_name == "random":
        return RandomAgent(max_steps_before_submit=8)
    else:
        raise ValueError(f"Unknown agent: {agent_name}. Available: random")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent on ARC puzzles")
    parser.add_argument("--agent", "-a", type=str, default="random",
                        help="Agent name: random")
    parser.add_argument("--data", "-d", type=str, nargs="+",
                        default=["data/train", "data/evaluation"])
    parser.add_argument("--puzzle", "-p", type=str, default=None,
                        help="Run on a specific puzzle only")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of puzzles to evaluate (default: all)")
    parser.add_argument("--budget", "-b", type=int, default=20)
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="Episodes per puzzle (for stochastic agents)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    env = ArcEnv(data_dirs=args.data, max_steps=args.budget)
    agent = make_agent(args.agent)

    if len(env.puzzle_db) == 0:
        print("No puzzles found!")
        sys.exit(1)

    # Select puzzles
    if args.puzzle:
        puzzle_ids = [args.puzzle]
    else:
        puzzle_ids = env.puzzle_db.ids()
        if args.n:
            puzzle_ids = puzzle_ids[:args.n]

    print(f"Agent: {args.agent}")
    print(f"Puzzles: {len(puzzle_ids)}")
    print(f"Episodes per puzzle: {args.episodes}")
    print(f"Budget: {args.budget} steps")
    print()

    # Run evaluation
    all_results = []
    t0 = time.time()

    for pid in puzzle_ids:
        for ep in range(args.episodes):
            try:
                result = run_episode(env, agent, pid, verbose=args.verbose)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR on {pid}: {e}")
                all_results.append({
                    "puzzle_id": pid,
                    "success": False,
                    "accuracy": 0.0,
                    "steps": 0,
                    "total_reward": 0.0,
                    "error": str(e),
                })

    elapsed = time.time() - t0

    # Aggregate
    n_total = len(all_results)
    n_solved = sum(1 for r in all_results if r["success"])
    avg_accuracy = sum(r["accuracy"] for r in all_results) / max(n_total, 1)
    avg_steps = sum(r["steps"] for r in all_results) / max(n_total, 1)
    avg_reward = sum(r["total_reward"] for r in all_results) / max(n_total, 1)
    n_budget = sum(1 for r in all_results if r.get("budget_exceeded"))

    print(f"\n{'═'*50}")
    print(f"  RESULTS")
    print(f"{'═'*50}")
    print(f"  Total episodes:     {n_total}")
    print(f"  Solved:             {n_solved}/{n_total} ({n_solved/max(n_total,1):.1%})")
    print(f"  Avg accuracy:       {avg_accuracy:.1%}")
    print(f"  Avg steps:          {avg_steps:.1f}")
    print(f"  Avg reward:         {avg_reward:.3f}")
    print(f"  Budget exceeded:    {n_budget}")
    print(f"  Time:               {elapsed:.1f}s")
    print(f"{'═'*50}")

    if args.output:
        summary = {
            "agent": args.agent,
            "n_puzzles": len(puzzle_ids),
            "n_episodes": n_total,
            "solved": n_solved,
            "solve_rate": n_solved / max(n_total, 1),
            "avg_accuracy": avg_accuracy,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "budget_exceeded": n_budget,
            "elapsed_sec": elapsed,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
