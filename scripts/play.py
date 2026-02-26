#!/usr/bin/env python3
"""
Interactive human play mode for the ARC DSL Simulator.

Usage:
    python -m scripts.play                      # random puzzle
    python -m scripts.play --puzzle 00d62c1b    # specific puzzle
    python -m scripts.play --data data/train    # specific data dir
"""

import argparse
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from text_parser import HELP_TEXT


def main():
    parser = argparse.ArgumentParser(description="Play ARC puzzles interactively")
    parser.add_argument("--puzzle", "-p", type=str, default=None,
                        help="Puzzle ID (e.g. 00d62c1b)")
    parser.add_argument("--data", "-d", type=str, nargs="+",
                        default=["data/train", "data/evaluation"],
                        help="Data directories")
    parser.add_argument("--budget", "-b", type=int, default=30,
                        help="Max steps per episode")
    parser.add_argument("--color", "-c", action="store_true",
                        help="Use ANSI colors")
    parser.add_argument("--cheat", action="store_true",
                        help="Show expected output at start")
    args = parser.parse_args()

    render_mode = "ansi" if args.color else "text"

    env = ArcEnv(
        data_dirs=args.data,
        max_steps=args.budget,
        render_mode=render_mode,
    )

    if len(env.puzzle_db) == 0:
        print("No puzzles found! Check your --data paths.")
        print(f"Looked in: {args.data}")
        sys.exit(1)

    print("╔══════════════════════════════════════╗")
    print("║   ARC DSL Simulator — Interactive    ║")
    print("╚══════════════════════════════════════╝")
    print(f"Loaded {len(env.puzzle_db)} puzzles.")
    print("Type 'help' for commands.\n")

    # Reset
    obs = env.reset(args.puzzle)
    puzzle_id = obs["puzzle_id"]
    print(f"Puzzle: {puzzle_id}")
    print(f"Budget: {obs['budget_remaining']} steps\n")

    # Show train examples
    for ex in obs.get("train_examples", []):
        print(f"--- Train Example {ex['index']} ---")
        print(f"Input ({ex['input_shape']}):")
        print(ex["input_grid"])
        print(f"Output ({ex['output_shape']}):")
        print(ex["output_grid"])
        print()

    # Show test input
    test = obs.get("test_input", {})
    print(f"--- Test Input ({test.get('shape', '?')}) ---")
    print(test.get("grid", ""))
    print()

    if args.cheat:
        print("--- Expected Output (CHEAT) ---")
        print(env.render_expected())
        print()

    # Main loop
    while True:
        try:
            prompt = f"[step {env.steps_taken}/{env.max_steps}] > "
            command = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if command.lower() == "new":
            obs = env.reset(args.puzzle)
            print(f"\nNew puzzle: {obs['puzzle_id']}")
            print(f"Budget: {obs['budget_remaining']} steps")
            continue
        if command.lower() == "cheat":
            print(env.render_expected())
            continue

        obs, reward, done, info = env.step_text(command)

        # Print result
        result = obs.get("last_action_result", info)
        msg = result.get("message", "")
        err = result.get("error", "")

        if err:
            print(f"  ✗ Error: {err}")
        if msg:
            print(msg)
        if reward != 0 and not done:
            print(f"  (reward: {reward:+.3f})")

        if done:
            print(f"\n{'='*40}")
            print(f"Episode ended. Total reward: {info.get('total_reward', env.total_reward):.3f}")
            if info.get("budget_exceeded"):
                print("Budget exceeded!")
            print(f"{'='*40}")

            try:
                again = input("Play again? (y/n/same): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if again in ("y", "yes", ""):
                obs = env.reset()
                print(f"\nNew puzzle: {obs['puzzle_id']}")
                for ex in obs.get("train_examples", []):
                    print(f"--- Train Example {ex['index']} ---")
                    print(ex["input_grid"])
                    print(ex["output_grid"])
                    print()
                test = obs.get("test_input", {})
                print(f"--- Test Input ---")
                print(test.get("grid", ""))
            elif again == "same":
                obs = env.reset(puzzle_id)
                print(f"\nRestarting puzzle: {puzzle_id}")
            else:
                print("Goodbye!")
                break


if __name__ == "__main__":
    main()
