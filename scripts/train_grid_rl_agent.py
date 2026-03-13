#!/usr/bin/env python3
"""
Train GridRLAgent with REINFORCE on ARC-AGI.

Structure mirrors train_rl_agent.py as closely as possible.
The only real difference: each policy decision picks a *macro*, which may
execute 1-3 DSL steps automatically.  We sum the raw env rewards across
those steps to get one reward number per macro decision — exactly like
rl_agent, where each policy decision produces one env reward.

Usage:
    python -m scripts.train_grid_rl_agent --smoke-test
    python -m scripts.train_grid_rl_agent --episodes 5000 --save-dir checkpoints/grid_rl
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.grid_rl_agent import GridRLAgent, get_macro_actions, _describe_macro, NUM_MACROS


def run_episode(env, agent, puzzle_id: str, verbose=False):
    """
    Run one episode.

    The agent picks a macro at each step.  Each macro is 1-3 DSL calls
    executed in sequence.  We sum the raw env rewards across those calls
    and treat the total as the reward for that one macro decision —
    mirroring how rl_agent handles a single-step action.

    Returns (total_env_reward, solved, n_macro_steps).
    """
    obs  = env.reset(puzzle_id)
    agent.setup(obs)

    total_reward = 0.0
    n_macros     = 0
    done         = False
    info         = {}

    while not done and obs.get("budget_remaining", 0) > 0:
        agent.act(obs)
        macro_steps = agent.get_macro_steps(obs)
        if not macro_steps:
            break

        # Execute every DSL call in the macro; sum raw env rewards.
        macro_reward = 0.0
        for action in macro_steps:
            if done:
                break
            obs, env_reward, done, info = env.step(action)
            macro_reward += env_reward
            if done:
                break

        total_reward += macro_reward
        n_macros     += 1
        agent.on_step_result(obs, macro_reward, done, info)

        if verbose:
            desc = _describe_macro(macro_steps)
            tick = "✓" if info.get("exact_match") else " "
            print(f"  [{tick}] macro {n_macros}: {desc:<45}  r={macro_reward:+.3f}")

    solved = info.get("exact_match", False)
    agent.on_episode_end(total_reward, n_macros, info)
    return total_reward, solved, n_macros


def main():
    parser = argparse.ArgumentParser(description="Train GridRLAgent on ARC-AGI")
    parser.add_argument("--data",        nargs="+", default=["data/train"])
    parser.add_argument("--episodes",    type=int,   default=5000)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--hidden-dim",  type=int,   default=128)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--max-steps",   type=int,   default=20)
    parser.add_argument("--save-dir",    default="checkpoints/grid_rl")
    parser.add_argument("--save-every",  type=int,   default=500)
    parser.add_argument("--log-every",   type=int,   default=100)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--smoke-test",  action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.episodes   = 100
        args.log_every  = 20
        args.save_every = 999

    random.seed(args.seed)

    env = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    puzzle_ids = env.puzzle_db.ids()
    print(f"Env: {len(puzzle_ids)} puzzles  |  max_steps={args.max_steps}")

    agent = GridRLAgent(hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma)
    n_p = sum(p.numel() for p in agent._policy.parameters())
    print(f"Policy: {n_p:,} params  (state=57 -> {args.hidden_dim} -> "
          f"{args.hidden_dim//2} -> {NUM_MACROS} macros, ceiling ~4.75%)")
    print(f"Training for {args.episodes} episodes\n")

    os.makedirs(args.save_dir, exist_ok=True)
    stats      = []
    rew_buf    = []
    win_buf    = []
    success_buf = []   # puzzles solved at least once — replayed to reinforce the policy
    t0         = time.time()

    print(f"{'Ep':>6}  {'AvgRew':>8}  {'Win%':>6}  {'Loss':>9}")
    print("-" * 38)

    for ep in range(1, args.episodes + 1):
        # Occasionally replay a previously solved puzzle (if any) to amplify the
        # rare positive gradient signal.  25% of episodes use success replay.
        if success_buf and random.random() < 0.25:
            pid = random.choice(success_buf)
        else:
            pid = random.choice(puzzle_ids)

        reward, solved, n_macros = run_episode(env, agent, pid, verbose=args.verbose)
        loss = agent.reinforce_update()

        if solved and pid not in success_buf:
            success_buf.append(pid)

        rew_buf.append(reward)
        win_buf.append(int(solved))
        if len(rew_buf) > 200:
            rew_buf.pop(0); win_buf.pop(0)

        stats.append({
            "episode": ep, "puzzle_id": pid,
            "reward": reward, "success": solved,
            "n_macros": n_macros, "loss": loss,
        })

        if ep % args.log_every == 0:
            avg_r = sum(rew_buf) / len(rew_buf)
            win   = sum(win_buf) / len(win_buf) * 100
            print(f"{ep:>6}  {avg_r:>8.4f}  {win:>5.1f}%  {loss:>9.5f}")

        if ep % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"ckpt_ep{ep}.pt")
            agent.save(ckpt)
            with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
                json.dump(stats, f)
            print(f"  → Saved {ckpt}")

    elapsed = time.time() - t0
    solved_total = sum(s["success"] for s in stats)
    print(f"\nDone in {elapsed:.1f}s  |  solved {solved_total}/{args.episodes} "
          f"({solved_total/args.episodes:.1%})")

    final = os.path.join(args.save_dir, "ckpt_final.pt")
    agent.save(final)
    with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"Checkpoint: {final}")
    print(f"\nEvaluate:")
    print(f"  python -m scripts.evaluate_grid_rl_agent --checkpoint {final}")


if __name__ == "__main__":
    main()
