#!/usr/bin/env python3
"""
Train BC model with per-epoch evaluation.

Trains Qwen3-1.7B for N epochs, runs the full evaluation pipeline after each
epoch, and saves a learning curve (JSON + plots). One training run gives you
the whole picture — no grid search needed.

Usage (Colab):
  python -m scripts.train_bc_model_tracked \
    --data data/offline_trajectories.json \
    --save_dir checkpoints/bc_tracked \
    --model_name Qwen/Qwen3-1.7B \
    --epochs 10 \
    --lr 2e-5 \
    --eval_data data/train \
    --eval_n 50

The script produces:
  <save_dir>/epoch_metrics.json   — per-epoch metrics
  <save_dir>/learning_curves.png  — plots for your presentation
  <save_dir>/epoch_<N>/           — checkpoint per epoch
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_parser import action_to_command
from observation_format import format_observation_for_prompt


# ─── Data loading (unchanged) ──────────────────────────────────────

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


# ─── Per-epoch evaluation callback ─────────────────────────────────

def run_eval_subprocess(model_dir, offline_data, eval_data, eval_n, eval_budget):
    """Shell out to evaluate.py and return parsed metrics."""
    eval_output = os.path.join(model_dir, "eval_results.json")
    cmd = [
        sys.executable, "-m", "scripts.evaluate",
        "--agent", "learning",
        "--model-path", model_dir,
        "--offline-data", offline_data,
        "--data", eval_data,
        "--n", str(eval_n),
        "--budget", str(eval_budget),
        "--output", eval_output,
    ]
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
    trainer_ref,       # will hold the Trainer once created
    tokenizer,
    save_dir,
    offline_data,
    eval_data,
    eval_n,
    eval_budget,
    epoch_metrics,     # list we append results to
):
    """Build a TrainerCallback class that evals after each epoch."""
    from transformers import TrainerCallback

    class EvalAfterEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save checkpoint
            print(f"\n{'─'*50}")
            print(f"  Epoch {epoch} complete — saving & evaluating")
            print(f"{'─'*50}")
            trainer_ref[0].save_model(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

            # Get training loss from log history
            train_loss = None
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    train_loss = entry["loss"]
                    break

            # Run eval
            metrics = run_eval_subprocess(
                epoch_dir, offline_data, eval_data, eval_n, eval_budget
            )
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss

            epoch_metrics.append(metrics)

            # Live progress
            if metrics.get("eval_ok"):
                print(f"  → solve_rate={metrics['solve_rate']:.1%}  "
                      f"acc={metrics['avg_accuracy']:.1%}  "
                      f"reward={metrics['avg_reward']:.3f}  "
                      f"loss={train_loss or 0:.4f}")
            print()

            # Save running results so you can check mid-training
            with open(os.path.join(save_dir, "epoch_metrics.json"), "w") as f:
                json.dump(epoch_metrics, f, indent=2, default=str)

    return EvalAfterEpochCallback


# ─── Plotting ──────────────────────────────────────────────────────

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

    # Solve rate
    ax = axes[0, 0]
    ax.plot(epochs, solve, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Solve Rate", title="Solve Rate")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, accuracy, "s-", color="#16a34a", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Avg Accuracy", title="Average Accuracy")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1, 0]
    ax.plot(epochs, reward, "^-", color="#dc2626", linewidth=2, markersize=7)
    ax.set(xlabel="Epoch", ylabel="Avg Reward", title="Average Reward")
    ax.grid(True, alpha=0.3)

    # Training loss
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


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train BC model with per-epoch evaluation"
    )
    # Training args
    parser.add_argument("--data", "-d", type=str, default="data/offline_trajectories.json")
    parser.add_argument("--save_dir", "-o", type=str, default="checkpoints/bc_tracked")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    # Eval args
    parser.add_argument("--eval_data", type=str, default="data/train",
                        help="Data dir for evaluation puzzles")
    parser.add_argument("--eval_n", type=int, default=50,
                        help="Number of puzzles to evaluate per epoch")
    parser.add_argument("--eval_budget", type=int, default=20,
                        help="Max steps per puzzle during eval")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip eval (just train and save checkpoints)")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print("Install: pip install torch transformers datasets")
        raise SystemExit(1) from e

    # Load data
    pairs = build_prompt_target_pairs(args.data)
    print(f"Loaded {len(pairs)} (prompt, action) pairs")
    if not pairs:
        print("No pairs found. Check trajectory JSON.")
        raise SystemExit(1)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Tokenize
    prompt_prefix = "Observation:\n"
    action_prefix = "\nAction: "

    def tokenize_fn(examples):
        input_ids_list, labels_list, real_lens = [], [], []
        for p, a in zip(examples["prompt"], examples["target"]):
            prompt_part = prompt_prefix + p + action_prefix
            target_part = a + tokenizer.eos_token
            prompt_enc = tokenizer(prompt_part, add_special_tokens=True,
                                   truncation=True, max_length=args.max_length - 64)
            target_enc = tokenizer(target_part, add_special_tokens=False,
                                   truncation=True, max_length=64)
            p_ids = prompt_enc["input_ids"]
            t_ids = target_enc["input_ids"]
            ids = p_ids + t_ids
            real_len = len(ids)
            pad_len = args.max_length - real_len
            if pad_len > 0:
                ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad_len
            else:
                ids = ids[:args.max_length]
                real_len = args.max_length
            input_ids_list.append(ids)
            real_lens.append(real_len)
            lab = [-100] * len(p_ids) + t_ids + [-100] * max(0, pad_len)
            lab = (lab + [-100] * args.max_length)[:args.max_length]
            labels_list.append(lab)
        return {
            "input_ids": input_ids_list,
            "attention_mask": [[1]*rl + [0]*(args.max_length - rl) for rl in real_lens],
            "labels": labels_list,
        }

    dataset = Dataset.from_dict({
        "prompt": [p for p, _ in pairs],
        "target": [a for _, a in pairs],
    })
    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=dataset.column_names, desc="Tokenizing")

    # Setup callback
    epoch_metrics = []
    trainer_ref = [None]  # mutable ref so callback can access trainer

    callbacks = []
    if not args.skip_eval:
        EvalCB = make_eval_callback_class(
            trainer_ref=trainer_ref,
            tokenizer=tokenizer,
            save_dir=args.save_dir,
            offline_data=args.data,
            eval_data=args.eval_data,
            eval_n=args.eval_n,
            eval_budget=args.eval_budget,
            epoch_metrics=epoch_metrics,
        )
        callbacks.append(EvalCB())

    # Train
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="no",  # we save manually in the callback
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
    trainer_ref[0] = trainer  # give callback access

    print(f"\n{'═'*55}")
    print(f"  Training for {args.epochs} epochs, eval after each")
    print(f"  LR={args.lr}  BS={args.batch_size}  MaxLen={args.max_length}")
    print(f"  Eval: {args.eval_n} puzzles, budget={args.eval_budget}")
    print(f"{'═'*55}\n")

    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0

    # Save final model
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # Save final metrics
    summary = {
        "config": {
            "model": args.model_name,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "eval_n": args.eval_n,
        },
        "total_train_time": total_time,
        "epoch_metrics": epoch_metrics,
    }
    metrics_path = os.path.join(args.save_dir, "epoch_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'═'*55}")
    print(f"  EPOCH-BY-EPOCH RESULTS")
    print(f"{'═'*55}")
    print(f"  {'Epoch':<7} {'Solve%':<9} {'Acc%':<9} {'Reward':<9} {'Loss':<9}")
    print(f"  {'─'*43}")
    for m in epoch_metrics:
        if m.get("eval_ok"):
            print(f"  {m['epoch']:<7} {m['solve_rate']:<9.1%} {m['avg_accuracy']:<9.1%} "
                  f"{m['avg_reward']:<9.3f} {m.get('train_loss', 0):<9.4f}")
    print(f"  {'─'*43}")
    print(f"  Total time: {total_time:.0f}s")

    if epoch_metrics:
        best = max([m for m in epoch_metrics if m.get("eval_ok")],
                   key=lambda m: (m["solve_rate"], m["avg_accuracy"]),
                   default=None)
        if best:
            print(f"\n  ★ Best epoch: {best['epoch']}  "
                  f"(solve={best['solve_rate']:.1%}, acc={best['avg_accuracy']:.1%})")
            print(f"    Checkpoint: {args.save_dir}/epoch_{best['epoch']}/")

    # Plot
    plot_learning_curves(epoch_metrics, args.save_dir)
    print(f"\nAll results saved to {metrics_path}")


if __name__ == "__main__":
    main()