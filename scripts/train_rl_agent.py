#!/usr/bin/env python3
"""
Train RLAgent with REINFORCE on ArcEnv.

Runs many episodes, collects (s, a, log_prob, r), then updates the policy with REINFORCE.
Saves checkpoint to --save_dir.

Usage:
  python -m scripts.train_rl_agent --data data/train --save_dir checkpoints/rl_agent --episodes 500
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.rl_agent import RLAgent


def main():
    parser = argparse.ArgumentParser(description="Train RL agent (REINFORCE) on ARC DSL env")
    parser.add_argument("--data", "-d", type=str, nargs="+", default=["data/train"])
    parser.add_argument("--save_dir", "-o", type=str, default="checkpoints/rl_agent")
    parser.add_argument("--episodes", "-e", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    env = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    agent = RLAgent(max_steps=args.max_steps)

    puzzle_ids = env.puzzle_db.ids()
    if not puzzle_ids:
        print("No puzzles found in data dirs.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)
    import random
    solved_count = 0
    for ep in range(args.episodes):
        pid = random.choice(puzzle_ids)
        obs = env.reset(puzzle_id=pid)
        agent.setup(obs)
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            agent.on_step_result(obs, reward, done, info)
            if done:
                agent.on_episode_end(
                    info.get("total_reward", 0),
                    info.get("steps_taken", 0),
                    info,
                )
                if info.get("exact_match", False):
                    solved_count += 1
                break

        agent.reinforce_update(gamma=args.gamma)

        if args.verbose and (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{args.episodes}, solved so far: {solved_count}")

        if (ep + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ckpt_ep{ep + 1}.pt")
            agent.save(path)
            if args.verbose:
                print(f"Saved {path}")

    path = os.path.join(args.save_dir, "ckpt_final.pt")
    agent.save(path)
    print(f"Training done. Solved {solved_count}/{args.episodes} episodes. Checkpoint: {path}")


if __name__ == "__main__":
    main()
