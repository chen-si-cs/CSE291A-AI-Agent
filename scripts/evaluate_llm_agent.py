#!/usr/bin/env python3
"""
Evaluate LLMAgent on ARC puzzles using TritonAI API (OpenAI-compatible).

Usage:
    python -m scripts.evaluate_llm_agent --n 5
    python -m scripts.evaluate_llm_agent --model api-llama-4-scout --n 10
    python -m scripts.evaluate_llm_agent --model us.anthropic.claude-opus-4-6-v1 --n 5
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import ArcEnv
from agents.llm_agent import LLMAgent


API_URL = "https://tritonai-api.ucsd.edu/v1/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    print("Please set it before running this script.", file=sys.stderr)
    sys.exit(1)

AVAILABLE_MODELS = [
    "api-llama-4-scout",
    "us.anthropic.claude-opus-4-6-v1",
    "us.deepseek.r1-v1:0",
    "mistral.mistral-large-3-675b-instruct",
    "us.amazon.nova-premier-v1:0",
    "moonshotai.kimi-k2.5",
    "minimax.minimax-m2",
    "api-gpt-oss-120b",
]


def make_llm_call(model: str, system_prompt: str | None = None):
    """Create an llm_call function for the given model."""

    def llm_call(messages: list[dict]) -> str:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)

        payload = {
            "model": model,
            "messages": api_messages,
            "max_tokens": 16384,
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return llm_call


def run_episode(env: ArcEnv, agent: LLMAgent, puzzle_id: str, verbose: bool = False):
    """Run one episode, return (solved, reward, steps, info)."""
    obs = env.reset(puzzle_id)
    agent.setup(obs)

    done = False
    steps = 0
    consecutive_errors = 0
    while not done:
        try:
            action = agent.act(obs)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] agent.act failed: {e}")
            consecutive_errors += 1
            if consecutive_errors >= 5:
                break
            continue

        if action.get("type") == "error":
            consecutive_errors += 1
            if verbose:
                print(f"  [PARSE ERROR] {action.get('message', '')}")
            if consecutive_errors >= 5:
                break
            # Feed error back to the LLM
            obs["last_action_result"] = {
                "error": f"Parse Error: Could not parse command. Please ensure your LAST LINE is a valid command like: execute objects(I, True, False, False) -> x1"
            }
            continue

        consecutive_errors = 0
        obs, reward, done, info = env.step(action)
        steps += 1

        if verbose:
            result = obs.get("last_action_result", {})
            msg = result.get("message", result.get("error", ""))
            print(f"  Step {steps}: {msg}")

        if steps >= env.max_steps:
            break

    solved = info.get("exact_match", False) if done else False
    total_reward = info.get("total_reward", env.total_reward) if done else env.total_reward
    return solved, total_reward, steps, info


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Agent on ARC puzzles")
    parser.add_argument("--model", type=str, default="api-llama-4-scout",
                        help=f"Model name. Available: {AVAILABLE_MODELS}")
    parser.add_argument("--data", type=str, nargs="+",
                        default=["data/train", "data/evaluation"])
    parser.add_argument("--n", type=int, default=5,
                        help="Number of puzzles to evaluate")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--puzzle", "-p", type=str, default=None,
                        help="Run on a specific puzzle ID")
    parser.add_argument("--id-file", type=str, default=None,
                        help="JSON file with list of puzzle IDs to evaluate")
    args = parser.parse_args()

    env = ArcEnv(data_dirs=args.data, max_steps=args.max_steps)
    print(f"Model: {args.model}")
    print(f"Puzzles available: {len(env.puzzle_db)}")

    # Select puzzles
    if args.puzzle:
        puzzle_ids = [args.puzzle]
    elif args.id_file:
        import json as _json
        with open(args.id_file) as f:
            puzzle_ids = [pid for pid in _json.load(f) if pid in set(env.puzzle_db.ids())]
        if args.n and args.n < len(puzzle_ids):
            import random
            random.seed(42)
            puzzle_ids = random.sample(puzzle_ids, args.n)
    else:
        import random
        all_ids = env.puzzle_db.ids()
        random.seed(42)
        puzzle_ids = random.sample(all_ids, min(args.n, len(all_ids)))

    print(f"Evaluating on {len(puzzle_ids)} puzzles\n")

    llm_call = make_llm_call(args.model)
    agent = LLMAgent(llm_call=llm_call, verbose=args.verbose, use_system_message=False)

    results = []
    solved_count = 0
    total_time = 0

    for i, pid in enumerate(puzzle_ids):
        print(f"[{i+1}/{len(puzzle_ids)}] Puzzle {pid} ... ", end="", flush=True)
        t0 = time.time()

        try:
            solved, reward, steps, info = run_episode(env, agent, pid, verbose=args.verbose)
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append({"puzzle_id": pid, "solved": False, "reward": -1, "steps": 0, "error": str(e)})
            continue

        elapsed = time.time() - t0
        total_time += elapsed

        status = "SOLVED" if solved else "FAILED"
        print(f"{status}  reward={reward:.3f}  steps={steps}  time={elapsed:.1f}s")

        if solved:
            solved_count += 1
        acc = info.get("accuracy", 0.0) if info else 0.0
        results.append({"puzzle_id": pid, "solved": solved, "reward": reward, "steps": steps, "accuracy": acc})

    # Summary
    n = len(results)
    avg_reward = sum(r["reward"] for r in results) / n if n else 0
    avg_steps = sum(r["steps"] for r in results) / n if n else 0
    avg_acc = sum(r.get("accuracy", 0) for r in results) / n if n else 0

    print(f"\n{'='*55}")
    print(f"  MODEL: {args.model}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Total episodes:     {n}")
    print(f"  Solved:             {solved_count}/{n} ({100*solved_count/n:.1f}%)")
    print(f"  Avg accuracy:       {100*avg_acc:.1f}%")
    print(f"  Avg reward:         {avg_reward:.3f}")
    print(f"  Avg steps:          {avg_steps:.1f}")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Avg time/puzzle:    {total_time/n:.1f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
