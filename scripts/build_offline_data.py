#!/usr/bin/env python3
"""
Build offline trajectory dataset from solvers.py.

For each puzzle that has a solve_<puzzle_id> function:
  1. Extract action sequence via solver_trajectory
  2. Run env.reset(puzzle_id) and replay actions
  3. Collect (observation, action) at each step and save to JSON

Usage:
  python -m scripts.build_offline_data --data data/train data/evaluation --out data/offline_trajectories.json
"""

from __future__ import annotations
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from solver_trajectory import solver_function_to_actions, get_solver_puzzle_id


def load_all_solvers():
    """Import solvers module and return {puzzle_id: solve_fn}."""
    import solvers
    out = {}
    for name in dir(solvers):
        if name.startswith("solve_") and callable(getattr(solvers, name)):
            fn = getattr(solvers, name)
            pid = get_solver_puzzle_id(fn)
            if pid:
                out[pid] = fn
    return out


def build_trajectories(env: ArcEnv, solver_map: dict, max_per_puzzle: int = 1) -> list:
    """
    For each puzzle_id in solver_map that exists in env, replay solver and collect (obs, action).
    Returns list of trajectories; each trajectory is list of {"observation": ..., "action": ...}.
    """
    trajectories = []
    for puzzle_id, solve_fn in solver_map.items():
        if puzzle_id not in env.puzzle_db:
            continue
        actions = solver_function_to_actions(solve_fn)
        if not actions:
            continue
        obs = env.reset(puzzle_id=puzzle_id)
        traj = []
        for i, action in enumerate(actions):
            traj.append({
                "observation": _serialize_obs(obs),
                "action": action,
                "step": i,
            })
            obs, reward, done, info = env.step(action)
            if info.get("error"):
                break
            if done:
                break
        if traj:
            trajectories.append({
                "puzzle_id": puzzle_id,
                "steps": traj,
                "success": info.get("success", False),
            })
    return trajectories


def _serialize_obs(obs: dict) -> dict:
    """Make observation JSON-serializable (no fancy types)."""
    out = {}
    for k, v in obs.items():
        if k == "inventory":
            out[k] = dict(v) if isinstance(v, dict) else v
        elif isinstance(v, (dict, list, str, int, float, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build offline trajectories from solvers")
    parser.add_argument("--data", "-d", type=str, nargs="+", default=["data/train", "data/evaluation"])
    parser.add_argument("--out", "-o", type=str, default="data/offline_trajectories.json")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max DSL steps per puzzle (default 200; longest solver needs ~44)")
    args = parser.parse_args()

    env = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    solver_map = load_all_solvers()

    puzzle_ids_in_env = set(env.puzzle_db.ids())
    matched = [p for p in solver_map if p in puzzle_ids_in_env]
    print(f"Solvers: {len(solver_map)}, In env: {len(puzzle_ids_in_env)}, Matched: {len(matched)}")

    trajectories = build_trajectories(env, solver_map)
    print(f"Built {len(trajectories)} trajectories")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(trajectories, f, indent=2, default=str)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
