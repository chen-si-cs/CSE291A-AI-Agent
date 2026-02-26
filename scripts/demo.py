#!/usr/bin/env python3
"""
Demo: solve puzzle 00d62c1b step-by-step, showing the environment in action.

Usage:
    python -m scripts.demo
    python -m scripts.demo --puzzle 00d62c1b --data data/train
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv


def demo_00d62c1b(env: ArcEnv):
    """
    Demonstrate solving puzzle 00d62c1b:
    Fill enclosed (non-border-touching) background regions with color 4.

    Solution from solvers.py:
        x1 = objects(I, T, F, F)
        x2 = colorfilter(x1, ZERO)
        x3 = rbind(bordering, I)
        x4 = compose(flip, x3)
        x5 = mfilter(x2, x4)
        O  = fill(I, FOUR, x5)
    """
    print("═" * 60)
    print("  DEMO: Solving puzzle 00d62c1b")
    print("  Pattern: fill enclosed background regions with yellow (4)")
    print("═" * 60)

    obs = env.reset("00d62c1b")
    print(f"\nPuzzle: {obs['puzzle_id']}")
    print(f"Train examples: {len(obs['train_examples'])}")

    # Show first train example
    ex = obs["train_examples"][0]
    print(f"\nTrain Example 0:")
    print(ex["input_grid"])
    print(ex["output_grid"])
    print(f"Changes: {ex['diff']}")

    # Show test input
    print(f"\nTest Input:")
    print(obs["test_input"]["grid"])

    # Step-by-step solution
    steps = [
        ("execute objects(I, True, False, False) -> x1",
         "Find all connected single-color regions (including background)"),
        ("execute colorfilter(x1, 0) -> x2",
         "Keep only the black (color 0) regions"),
        ("execute rbind(bordering, I) -> x3",
         "Create a function that checks if a patch touches the grid border"),
        ("execute compose(flip, x3) -> x4",
         "Negate it: now checks if a patch does NOT touch the border"),
        ("execute mfilter(x2, x4) -> x5",
         "Filter & merge: get all cells in non-bordering black regions"),
        ("execute fill(I, 4, x5) -> O",
         "Fill those cells with color 4 (yellow)"),
        ("submit O",
         "Submit the answer!"),
    ]

    for command, explanation in steps:
        print(f"\n{'─'*50}")
        print(f"  💡 {explanation}")
        print(f"  > {command}")

        obs, reward, done, info = env.step_text(command)
        result = obs.get("last_action_result", info)

        msg = result.get("message", "")
        err = result.get("error", "")
        if err:
            print(f"  ✗ {err}")
        elif msg:
            print(f"  {msg}")

        if done:
            print(f"\n{'═'*60}")
            print(f"  Episode complete!")
            print(f"  Correct: {info.get('success', info.get('exact_match', False))}")
            print(f"  Steps: {info.get('steps_taken', env.steps_taken)}")
            print(f"  Total reward: {info.get('total_reward', env.total_reward):.3f}")
            print(f"{'═'*60}")
            break


def demo_generic(env: ArcEnv, puzzle_id: str):
    """Show puzzle info and let the user see what's there."""
    obs = env.reset(puzzle_id)
    print(f"\nPuzzle: {obs['puzzle_id']}")
    print(f"Train examples: {len(obs['train_examples'])}")
    print(f"Budget: {obs['budget_remaining']} steps\n")

    for ex in obs["train_examples"]:
        print(f"--- Train Example {ex['index']} ---")
        print(ex["input_grid"])
        print(ex["output_grid"])
        print(f"Diff: {ex['diff']}\n")

    print(f"--- Test Input ---")
    print(obs["test_input"]["grid"])
    print(f"\n--- Expected Output (for reference) ---")
    print(env.render_expected())


def main():
    parser = argparse.ArgumentParser(description="Demo ARC DSL Simulator")
    parser.add_argument("--puzzle", "-p", type=str, default=None)
    parser.add_argument("--data", "-d", type=str, nargs="+",
                        default=["data/train", "data/evaluation"])
    args = parser.parse_args()

    env = ArcEnv(data_dirs=args.data, max_steps=30)

    if len(env.puzzle_db) == 0:
        print("No puzzles found in data dirs. Loading the example puzzle...")
        # Try loading from uploads
        import glob
        for p in glob.glob("*.json") + glob.glob("data/**/*.json", recursive=True):
            try:
                env.puzzle_db.load_single(p)
            except Exception:
                pass

    if args.puzzle:
        if args.puzzle == "00d62c1b" and args.puzzle in env.puzzle_db:
            demo_00d62c1b(env)
        else:
            demo_generic(env, args.puzzle)
    elif "00d62c1b" in env.puzzle_db:
        demo_00d62c1b(env)
    elif len(env.puzzle_db) > 0:
        demo_generic(env, env.puzzle_db.ids()[0])
    else:
        print("No puzzles available to demo.")


if __name__ == "__main__":
    main()
