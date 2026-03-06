#!/usr/bin/env python3
"""
Train GridRLAgent with REINFORCE on ARC-AGI.

The agent takes macro-actions (1-3 DSL step templates) and observes a
57-dim grid feature vector. This is sequential decision-making: multiple
macro-actions per puzzle episode, each updating the inventory toward a solution.

Key differences from RLAgent (agents/rl_agent.py):
  - State: 57 grid features vs 4 numbers (agent can see the puzzle)
  - Actions: 40 macro-actions covering real DSL functions vs 29 unary-only ops
  - Training: REINFORCE with moving-average baseline vs bare REINFORCE
  - Episode: multiple macro-actions per step (each macro = 1-3 DSL calls)

Usage:
    python -m scripts.train_grid_rl_agent --smoke-test
    python -m scripts.train_grid_rl_agent --episodes 500 --save-dir checkpoints/grid_rl
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
from agents.grid_rl_agent import GridRLAgent, get_macro_actions, _describe_macro


def shape_reward(env_reward: float, info: dict, done: bool) -> float:
    """
    Convert env reward to a training signal that cannot be gamed by
    submitting the input unchanged.

    The env gives partial credit (up to 0.5) based on cell-level accuracy,
    which means submit(I) averages ~0.25 reward on many puzzles since input
    and output often share cells.  This creates a strong incentive to submit
    immediately without trying to solve anything.

    Shaped reward:
      +1.0  exact match (solved)
       0.0  anything else (wrong submission, step, invalid action)

    This is sparse but unambiguous: the only way to get reward is to solve
    the puzzle.  The agent must explore; there is no shortcut.
    """
    if done and info.get("success"):
        return 1.0
    return 0.0


def run_episode(env, agent, puzzle_id: str, verbose=False):
    """
    Run one episode: agent takes macro-actions until done or budget exhausted.
    Each macro-action executes 1-3 DSL steps in the env.
    Returns (shaped_total_reward, success, n_macro_steps).
    """
    obs = env.reset(puzzle_id)
    agent.setup(obs)

    total_reward = 0.0
    n_macros     = 0
    done         = False

    while not done and obs.get("budget_remaining", 0) > 0:
        # Agent picks a macro
        agent.act(obs)
        macro_steps = agent.get_macro_steps(obs)

        if not macro_steps:
            break

        # Execute all steps of the macro
        macro_shaped = 0.0
        for action in macro_steps:
            if done:
                break
            obs, env_reward, done, info = env.step(action)
            macro_shaped += shape_reward(env_reward, info, done)
            if done:
                break

        total_reward += macro_shaped
        n_macros     += 1
        agent.on_step_result(obs, macro_shaped, done, info)

        if verbose:
            desc = _describe_macro(macro_steps)
            tick = "✓" if done and info.get("success") else " "
            print(f"  [{tick}] macro {n_macros}: {desc:<45}  r={macro_shaped:+.3f}")

    success = info.get("success", False)
    agent.on_episode_end(total_reward, n_macros, info)
    return total_reward, success, n_macros


def main():
    parser = argparse.ArgumentParser(
        description="Train GridRLAgent on ARC-AGI"
    )
    parser.add_argument("--data",        nargs="+", default=["data/train"])
    parser.add_argument("--episodes",    type=int,   default=5000)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--hidden-dim",  type=int,   default=128)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--max-steps",   type=int,   default=20,
                        help="Max DSL budget per episode")
    parser.add_argument("--save-dir",    default="checkpoints/grid_rl")
    parser.add_argument("--save-every",  type=int,   default=500)
    parser.add_argument("--log-every",   type=int,   default=100)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--smoke-test",  action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.episodes  = 100
        args.log_every = 20
        args.save_every = 999

    random.seed(args.seed)

    env = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    puzzle_ids = env.puzzle_db.ids()
    print(f"Env: {len(puzzle_ids)} puzzles  |  max_steps={args.max_steps}")

    agent = GridRLAgent(hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma)
    n_p = sum(p.numel() for p in agent._policy.parameters())
    print(f"Policy: {n_p:,} params  (state=57 → {args.hidden_dim} → "
          f"{args.hidden_dim//2} → 40 macros)")
    print(f"Training for {args.episodes} episodes\n")

    os.makedirs(args.save_dir, exist_ok=True)
    stats    = []
    rew_buf  = []
    win_buf  = []
    t0       = time.time()

    print(f"{'Ep':>6}  {'AvgRew':>8}  {'Win%':>6}  {'Loss':>9}  {'Baseline':>9}")
    print("-" * 48)

    for ep in range(1, args.episodes + 1):
        pid = random.choice(puzzle_ids)
        reward, success, n_macros = run_episode(
            env, agent, pid, verbose=args.verbose
        )
        loss = agent.reinforce_update()

        rew_buf.append(reward)
        win_buf.append(int(success))
        if len(rew_buf) > 200:
            rew_buf.pop(0); win_buf.pop(0)

        stats.append({
            "episode": ep, "puzzle_id": pid,
            "reward": reward, "success": success,
            "n_macros": n_macros, "loss": loss,
        })

        if ep % args.log_every == 0:
            avg_r = sum(rew_buf) / len(rew_buf)
            win   = sum(win_buf) / len(win_buf) * 100
            print(f"{ep:>6}  {avg_r:>8.4f}  {win:>5.1f}%  "
                  f"{loss:>9.5f}  {agent._baseline:>9.4f}")

        if ep % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"ckpt_ep{ep}.pt")
            agent.save(ckpt)
            with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
                json.dump(stats, f)
            print(f"  → Saved {ckpt}")

    elapsed = time.time() - t0
    solved  = sum(s["success"] for s in stats)
    print(f"\nDone in {elapsed:.1f}s  |  solved {solved}/{args.episodes} "
          f"({solved/args.episodes:.1%})")

    final = os.path.join(args.save_dir, "ckpt_final.pt")
    agent.save(final)
    with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"Checkpoint: {final}")
    print(f"\nEvaluate:")
    print(f"  python -m scripts.evaluate_grid_rl_agent --checkpoint {final}")


if __name__ == "__main__":
    main()