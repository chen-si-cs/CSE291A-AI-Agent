#!/usr/bin/env python3
"""
Train RLAgent (REINFORCE or GRPO) on ArcEnv.

Recommended command:
  python -m scripts.train_rl_agent --data data/train --episodes 8000 --grpo_k 4 --verbose

Why grpo_k=4 not 8:
  With 104 puzzles, k=8 means the same puzzle is run 8 times per update.
  When the policy collapses onto 2 puzzles, those 2 puzzles generate massive
  positive gradients at rate k=8 per episode. k=4 halves this amplification
  while still giving group-relative baselines.
"""

from __future__ import annotations
import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.rl_agent import RLAgent
from reward import compute_reward

SHAPE_SCALE = 0.3


def _train_accuracy(env: ArcEnv, stored_var: str) -> float:
    """Accuracy of stored_var against train output (not test — no oracle leakage)."""
    try:
        val = env.inventory.get(stored_var)
        if not isinstance(val, tuple):
            return 0.0
        train_out = env.puzzle.train_output(0)
        _, _, si = compute_reward(val, train_out, 0, 1, step_cost=0.0)
        return si.get("accuracy", 0.0)
    except Exception:
        return 0.0


def _identity_train_accuracy(env: ArcEnv) -> float:
    try:
        val       = env.puzzle.train_input(0)
        train_out = env.puzzle.train_output(0)
        _, _, si  = compute_reward(val, train_out, 0, 1, step_cost=0.0)
        return si.get("accuracy", 0.0)
    except Exception:
        return 0.0


def run_episode(env: ArcEnv, agent: RLAgent, puzzle_id: str,
                shape: bool = True) -> tuple[bool, float, list]:
    obs      = env.reset(puzzle_id=puzzle_id)
    agent.setup(obs)
    agent._env_ref = env
    done     = False
    prev_acc = _identity_train_accuracy(env) if shape else 0.0

    while not done:
        action                  = agent.act(obs)
        obs, reward, done, info = env.step(action)

        # Penalty for submitting raw input unchanged
        if action.get("_raw_input_submit"):
            reward -= 0.3

        # Shaped reward: improvement vs train output
        if shape and info.get("action_ok") and not done:
            stored_var = info.get("stored")
            if stored_var:
                new_acc  = _train_accuracy(env, stored_var)
                shaped   = (new_acc - prev_acc) * SHAPE_SCALE
                reward  += shaped
                prev_acc = new_acc

        agent.on_step_result(obs, reward, done, info)
        if done:
            agent.on_episode_end(
                info.get("total_reward", 0),
                info.get("steps_taken", 0),
                info,
            )

    solved = info.get("exact_match", False)
    return solved, info.get("total_reward", 0.0), agent.get_trajectory()


def _curriculum_sample(puzzle_ids: list, success_counts: dict,
                        attempt_counts: dict, solved_replay: list) -> str:
    """
    3-tier curriculum:
      15% replay solved puzzles (keep policy sharp)
      60% learnable zone (5-60% success rate, ≥5 attempts)
      25% uniform random (explore unseen puzzles)
    """
    r = random.random()

    if solved_replay and r < 0.15:
        return random.choice(solved_replay)

    learnable = [
        pid for pid in puzzle_ids
        if attempt_counts.get(pid, 0) >= 5
        and 0.05 <= success_counts.get(pid, 0) / attempt_counts[pid] <= 0.60
    ]

    if learnable and r < 0.75:
        return random.choice(learnable)

    return random.choice(puzzle_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       "-d", nargs="+", default=["data/train"])
    parser.add_argument("--save_dir",   "-o", default="checkpoints/rl_agent")
    parser.add_argument("--episodes",   "-e", type=int, default=8000)
    parser.add_argument("--max_steps",  type=int, default=20)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--grpo_k",     type=int, default=4,
                        help="GRPO group size. Recommend 4 (not 8) to reduce collapse risk.")
    parser.add_argument("--no_shaping", action="store_true")
    parser.add_argument("--verbose",    "-v", action="store_true")
    args = parser.parse_args()

    env        = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    agent      = RLAgent(max_steps=args.max_steps)
    puzzle_ids = env.puzzle_db.ids()

    if not puzzle_ids:
        print("No puzzles found.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)
    shape    = not args.no_shaping
    use_grpo = args.grpo_k > 1

    print(f"Puzzles: {len(puzzle_ids)}  |  Actions: {agent.num_actions}  |  "
          f"State: {agent.state_dim}  |  Shaping: {shape}  |  "
          f"{'GRPO k=' + str(args.grpo_k) if use_grpo else 'REINFORCE'}  |  "
          f"Episodes: {args.episodes}")

    solved_count   = 0
    solved_replay: list[str]      = []
    success_counts: dict[str,int] = defaultdict(int)
    attempt_counts: dict[str,int] = defaultdict(int)
    ema_baseline   = 0.0

    # Track steps histogram to detect collapse
    step_counts: list[int] = []

    for ep in range(args.episodes):
        pid = _curriculum_sample(puzzle_ids, success_counts, attempt_counts, solved_replay)

        if use_grpo:
            trajectories = []
            ep_steps = []
            for _ in range(args.grpo_k):
                solved, _, traj = run_episode(env, agent, pid, shape=shape)
                trajectories.append(traj)
                ep_steps.append(len(traj))
                attempt_counts[pid] += 1
                if solved:
                    solved_count += 1
                    success_counts[pid] += 1
                    if pid not in solved_replay:
                        solved_replay.append(pid)
            step_counts.append(sum(ep_steps) / len(ep_steps))
            agent.grpo_update(trajectories, gamma=args.gamma)

        else:
            solved, total_reward, traj = run_episode(env, agent, pid, shape=shape)
            attempt_counts[pid] += 1
            step_counts.append(len(traj))
            ema_baseline = 0.9 * ema_baseline + 0.1 * total_reward
            agent._trajectory = traj
            agent.reinforce_update(gamma=args.gamma, baseline=ema_baseline)

            if solved:
                solved_count += 1
                success_counts[pid] += 1
                if pid not in solved_replay:
                    solved_replay.append(pid)

        if args.verbose and (ep + 1) % 100 == 0:
            # Average steps over last 100 episodes — low value signals collapse
            recent_steps = step_counts[-100:] if len(step_counts) >= 100 else step_counts
            avg_steps    = sum(recent_steps) / len(recent_steps)
            n_learnable  = sum(
                1 for p in puzzle_ids
                if attempt_counts.get(p, 0) >= 5
                and 0.05 <= success_counts.get(p, 0) / attempt_counts[p] <= 0.60
            )
            entropy_beta = agent._anneal_entropy()
            print(f"Episode {ep+1:5d}/{args.episodes}  "
                  f"solved: {solved_count:5d}  "
                  f"unique: {len(solved_replay):3d}/{len(puzzle_ids)}  "
                  f"learnable: {n_learnable:3d}  "
                  f"avg_steps: {avg_steps:.1f}  "
                  f"entropy_beta: {entropy_beta:.3f}  "
                  f"updates: {agent._total_updates}")

        if (ep + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ckpt_ep{ep+1}.pt")
            agent.save(path)
            if args.verbose:
                print(f"  → Saved {path}")

    path = os.path.join(args.save_dir, "ckpt_final.pt")
    agent.save(path)
    print(f"\nDone. Solved {solved_count} episodes. "
          f"Unique: {len(solved_replay)}/{len(puzzle_ids)}. "
          f"Checkpoint: {path}")


if __name__ == "__main__":
    main()