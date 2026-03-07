#!/usr/bin/env python3
"""
Train BC model with grid search over model / inference parameters.

Instead of searching over learning-rate and batch-size, we fix those and
search over parameters that affect the model's behaviour at inference time:
  - max_length  (context window used during tokenization / training)
  - temperature (passed to the evaluation agent for generation)

For every (max_length, temperature) combo the script:
  1. Trains for N epochs (or reuses a checkpoint if max_length was already trained).
  2. Evaluates with the given temperature.
  3. Reports a comparison grid at the end.

Usage:
  python -m scripts.train_bc_model \
    --data data/offline_trajectories.json \
    --save_dir checkpoints/bc_grid \
    --model_name Qwen/Qwen3-1.7B \
    --epochs 10 \
    --max_lengths 512 768 1024 \
    --temperatures 0.1 0.5 1.0 \
    --eval_data data/train \
    --eval_n 50
"""

from __future__ import annotations
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_parser import action_to_command
from observation_format import format_observation_for_prompt
from puzzle_ids import TRAIN_IDS, TEST_IDS


# ─── Data loading ─────────────────────────────────────────────────

def build_prompt_target_pairs(trajectories_path: str) -> list[tuple[str, str]]:
    with open(trajectories_path) as f:
        trajectories = json.load(f)
    pairs = []
    for traj in trajectories:
        for s in traj.get("steps", []):
            obs = s.get("observation", {})
            action = s.get("action", {})
            if not action or action.get("type") == "error":
                continue
            cmd = action_to_command(action)
            if not cmd:
                continue
            prompt = format_observation_for_prompt(obs)
            pairs.append((prompt, cmd))
    return pairs


# ─── Per-epoch evaluation callback ────────────────────────────────

def run_eval_subprocess(model_dir, offline_data, eval_data, eval_n, eval_budget,
                        temperature=None, split="train"):
    """Shell out to evaluate.py and return parsed metrics."""
    eval_output = os.path.join(model_dir, f"eval_results_t{temperature}.json")
    # eval_data can be a string or list of dirs
    if isinstance(eval_data, str):
        eval_data = [eval_data]
    cmd = [
        sys.executable, "-m", "scripts.evaluate",
        "--agent", "learning",
        "--model-path", model_dir,
        "--offline-data", offline_data,
        "--data", *eval_data,
        "--n", str(eval_n),
        "--budget", str(eval_budget),
        "--split", split,
        "--output", eval_output,
    ]
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ⚠ Eval failed (exit {result.returncode})")
        print(f"    stderr: {result.stderr[-400:]}")
        return {"eval_ok": False, "eval_time": elapsed}

    try:
        with open(eval_output) as f:
            data = json.load(f)
        return {
            "eval_ok":      True,
            "eval_time":    elapsed,
            "solve_rate":   data.get("solve_rate", 0),
            "avg_accuracy": data.get("avg_accuracy", 0),
            "avg_reward":   data.get("avg_reward", 0),
            "avg_steps":    data.get("avg_steps", 0),
            "solved":       data.get("solved", 0),
            "n_episodes":   data.get("n_episodes", 0),
        }
    except Exception as e:
        print(f"    ⚠ Could not parse eval results: {e}")
        return {"eval_ok": False, "eval_time": elapsed}


def make_eval_callback_class(
    trainer_ref,
    tokenizer,
    save_dir,
    offline_data,
    eval_data,
    eval_n,
    eval_budget,
    epoch_metrics,
    temperature=None,
):
    """Build a TrainerCallback class that evals after each epoch."""
    from transformers import TrainerCallback

    class EvalAfterEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            print(f"\n{'─'*50}")
            print(f"  Epoch {epoch} complete — saving & evaluating")
            print(f"{'─'*50}")
            trainer_ref[0].save_model(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

            train_loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    train_loss = entry["loss"]
                    break

            metrics = run_eval_subprocess(
                epoch_dir, offline_data, eval_data, eval_n, eval_budget,
                temperature=temperature,
            )
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss

            epoch_metrics.append(metrics)

            if metrics.get("eval_ok"):
                print(f"  → solve_rate={metrics['solve_rate']:.1%}  "
                      f"acc={metrics['avg_accuracy']:.1%}  "
                      f"reward={metrics['avg_reward']:.3f}  "
                      f"loss={train_loss or 0:.4f}")
            print()

            with open(os.path.join(save_dir, "epoch_metrics.json"), "w") as f:
                json.dump(epoch_metrics, f, indent=2, default=str)

    return EvalAfterEpochCallback


# ─── Plotting ─────────────────────────────────────────────────────

def plot_learning_curves(epoch_metrics, save_dir):
    """Generate presentation-ready learning curve plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    valid = [m for m in epoch_metrics if m.get("eval_ok")]
    if not valid:
        print("No valid eval results to plot.")
        return

    epochs   = [m["epoch"] for m in valid]
    solve    = [m["solve_rate"] for m in valid]
    accuracy = [m["avg_accuracy"] for m in valid]
    reward   = [m["avg_reward"] for m in valid]
    loss     = [m.get("train_loss") for m in valid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("BC Training — Per-Epoch Learning Curves", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, solve, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Solve Rate", title="Solve Rate")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, accuracy, "s-", color="#16a34a", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Avg Accuracy", title="Average Accuracy")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, reward, "^-", color="#dc2626", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Avg Reward", title="Average Reward")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    loss_clean = [(e, l) for e, l in zip(epochs, loss) if l is not None]
    if loss_clean:
        ax.plot([e for e, _ in loss_clean], [l for _, l in loss_clean],
                "D-", color="#9333ea", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Train Loss", title="Training Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "learning_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")
    plt.close(fig)


def plot_grid_results(grid_results, save_dir):
    """Heatmap of grid-search results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")
    except ImportError:
        return

    if not grid_results:
        return

    max_lengths = sorted(set(r["max_length"] for r in grid_results))
    temperatures = sorted(set(r["temperature"] for r in grid_results))
    lookup = {(r["max_length"], r["temperature"]): r for r in grid_results}

    for metric_key, title in [("solve_rate", "Solve Rate"), ("avg_accuracy", "Avg Accuracy")]:
        grid = np.zeros((len(temperatures), len(max_lengths)))
        for i, t in enumerate(temperatures):
            for j, ml in enumerate(max_lengths):
                r = lookup.get((ml, t), {})
                grid[i, j] = r.get(metric_key, 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(max_lengths)))
        ax.set_xticklabels(max_lengths)
        ax.set_yticks(range(len(temperatures)))
        ax.set_yticklabels(temperatures)
        ax.set_xlabel("Context Window (max_length)")
        ax.set_ylabel("Temperature")
        ax.set_title(f"Grid Search — {title}")
        for i in range(len(temperatures)):
            for j in range(len(max_lengths)):
                ax.text(j, i, f"{grid[i,j]:.1%}", ha="center", va="center",
                        color="white" if grid[i,j] > grid.max()*0.6 else "black",
                        fontsize=11)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"grid_{metric_key}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Grid plots saved to {save_dir}/grid_*.png")


# ─── Single training run for a given max_length ──────────────────

def train_single(args, max_length: int, run_dir: str):
    """Train one model with the given max_length. Returns (tokenizer, epoch_metrics, train_time)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset
    import torch

    pairs = build_prompt_target_pairs(args.data)
    print(f"  Loaded {len(pairs)} (prompt, action) pairs")
    if not pairs:
        print("  No pairs found — skipping.")
        return None, [], 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    prompt_prefix = "Observation:\n"
    action_prefix = "\nAction: "

    def tokenize_fn(examples):
        input_ids_list, labels_list, real_lens = [], [], []
        for p, a in zip(examples["prompt"], examples["target"]):
            prompt_part = prompt_prefix + p + action_prefix
            target_part = a + tokenizer.eos_token
            prompt_enc = tokenizer(prompt_part, add_special_tokens=True,
                                   truncation=True, max_length=max_length - 64)
            target_enc = tokenizer(target_part, add_special_tokens=False,
                                   truncation=True, max_length=64)
            p_ids = prompt_enc["input_ids"]
            t_ids = target_enc["input_ids"]
            ids = p_ids + t_ids
            real_len = len(ids)
            pad_len = max_length - real_len
            if pad_len > 0:
                ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad_len
            else:
                ids = ids[:max_length]
                real_len = max_length
            input_ids_list.append(ids)
            real_lens.append(real_len)
            lab = [-100] * len(p_ids) + t_ids + [-100] * max(0, pad_len)
            lab = (lab + [-100] * max_length)[:max_length]
            labels_list.append(lab)
        return {
            "input_ids": input_ids_list,
            "attention_mask": [[1]*rl + [0]*(max_length - rl) for rl in real_lens],
            "labels": labels_list,
        }

    dataset = Dataset.from_dict({
        "prompt": [p for p, _ in pairs],
        "target": [a for _, a in pairs],
    })
    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=dataset.column_names, desc="Tokenizing")

    epoch_metrics = []
    trainer_ref = [None]

    callbacks = []
    if not args.skip_eval:
        EvalCB = make_eval_callback_class(
            trainer_ref=trainer_ref,
            tokenizer=tokenizer,
            save_dir=run_dir,
            offline_data=args.data,
            eval_data=args.eval_data,
            eval_n=args.eval_n,
            eval_budget=args.eval_budget,
            epoch_metrics=epoch_metrics,
            temperature=None,  # temperature is evaluated separately in the grid
        )
        callbacks.append(EvalCB())

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="no",
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        callbacks=callbacks,
    )
    trainer_ref[0] = trainer

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    return tokenizer, epoch_metrics, train_time


# ─── Main (grid search) ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train BC model with grid search over temperature & context window"
    )
    # Training args (fixed across grid)
    parser.add_argument("--data", "-d", type=str, default="data/offline_trajectories.json")
    parser.add_argument("--save_dir", "-o", type=str, default="checkpoints/bc_grid")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Grid search params
    parser.add_argument("--max_lengths", type=int, nargs="+", default=[512, 768, 1024],
                        help="Context window sizes to search over")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                        help="Generation temperatures to search over")

    # Eval args
    parser.add_argument("--eval_data", type=str, nargs="+",
                        default=["data/train", "data/evaluation"],
                        help="Data dir(s) for evaluation puzzles (must include dirs for both splits)")
    parser.add_argument("--eval_n", type=int, default=50,
                        help="Number of puzzles to evaluate per grid point")
    parser.add_argument("--eval_budget", type=int, default=30,
                        help="Max steps per puzzle during eval")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip eval (just train checkpoints for each max_length)")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print("Install: pip install torch transformers datasets")
        raise SystemExit(1) from e

    os.makedirs(args.save_dir, exist_ok=True)

    combos = list(itertools.product(args.max_lengths, args.temperatures))
    print(f"\n{'═'*60}")
    print(f"  GRID SEARCH: {len(args.max_lengths)} max_lengths × {len(args.temperatures)} temperatures")
    print(f"  = {len(combos)} evaluation points")
    print(f"  (Only {len(args.max_lengths)} training runs needed — temperature is eval-time)")
    print(f"  max_lengths:  {args.max_lengths}")
    print(f"  temperatures: {args.temperatures}")
    print(f"  epochs: {args.epochs}  lr: {args.lr}  batch_size: {args.batch_size}")
    print(f"{'═'*60}\n")

    # Phase 1: Train one model per unique max_length
    trained = {}  # max_length -> run_dir
    for ml in args.max_lengths:
        run_dir = os.path.join(args.save_dir, f"ml_{ml}")
        print(f"\n{'━'*60}")
        print(f"  TRAINING: max_length={ml}  →  {run_dir}")
        print(f"{'━'*60}")
        if os.path.exists(os.path.join(run_dir, "config.json")):
            print(f"  Checkpoint already exists — skipping training.")
            trained[ml] = run_dir
            continue
        os.makedirs(run_dir, exist_ok=True)
        _, epoch_metrics, train_time = train_single(args, ml, run_dir)
        trained[ml] = run_dir
        with open(os.path.join(run_dir, "train_summary.json"), "w") as f:
            json.dump({
                "max_length": ml, "epochs": args.epochs,
                "lr": args.lr, "batch_size": args.batch_size,
                "train_time": train_time,
                "epoch_metrics": epoch_metrics,
            }, f, indent=2, default=str)
        if epoch_metrics:
            plot_learning_curves(epoch_metrics, run_dir)

    # Phase 2: Evaluate every (max_length, temperature) combination
    print(f"\n{'═'*60}")
    print(f"  GRID EVALUATION")
    print(f"{'═'*60}")

    grid_results = []
    for ml, temp in combos:
        run_dir = trained[ml]
        print(f"\n  Evaluating: max_length={ml}  temperature={temp}")
        metrics = run_eval_subprocess(
            run_dir, args.data, args.eval_data, args.eval_n, args.eval_budget,
            temperature=temp,
        )
        metrics["max_length"] = ml
        metrics["temperature"] = temp
        grid_results.append(metrics)
        if metrics.get("eval_ok"):
            print(f"    → solve_rate={metrics['solve_rate']:.1%}  "
                  f"acc={metrics['avg_accuracy']:.1%}  "
                  f"reward={metrics['avg_reward']:.3f}")

    # Save & display
    grid_path = os.path.join(args.save_dir, "grid_results.json")
    with open(grid_path, "w") as f:
        json.dump(grid_results, f, indent=2, default=str)

    print(f"\n{'═'*60}")
    print(f"  GRID SEARCH RESULTS")
    print(f"{'═'*60}")
    print(f"  {'MaxLen':<9} {'Temp':<8} {'Solve%':<9} {'Acc%':<9} {'Reward':<9}")
    print(f"  {'─'*44}")
    for r in grid_results:
        if r.get("eval_ok"):
            print(f"  {r['max_length']:<9} {r['temperature']:<8.2f} "
                  f"{r['solve_rate']:<9.1%} {r['avg_accuracy']:<9.1%} "
                  f"{r['avg_reward']:<9.3f}")
    print(f"  {'─'*44}")

    valid = [r for r in grid_results if r.get("eval_ok")]
    if valid:
        best = max(valid, key=lambda r: (r["solve_rate"], r["avg_accuracy"]))
        print(f"\n  ★ Best config: max_length={best['max_length']}  "
              f"temperature={best['temperature']}")
        print(f"    solve_rate={best['solve_rate']:.1%}  "
              f"avg_accuracy={best['avg_accuracy']:.1%}")
        print(f"    Checkpoint: {trained[best['max_length']]}/")

    plot_grid_results(grid_results, args.save_dir)
    print(f"\nAll results saved to {grid_path}")


if __name__ == "__main__":
    main()