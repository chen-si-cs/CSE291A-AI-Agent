#!/usr/bin/env python3
"""
Train BC model with LoRA and grid search over inference parameters.

Uses LoRA (Low-Rank Adaptation) instead of full fine-tuning:
  - Only ~0.5-2% of parameters are trained → strong regularization
  - Much less GPU memory needed
  - Better generalization on small datasets (we have ~582 pairs)

At save time, LoRA weights are merged into the base model so the saved
checkpoint is a standard HuggingFace model — learning_agent.py needs no
changes to load it.

Grid search axes:
  - max_length  (context window for tokenization / training)
  - temperature (generation temperature at eval time)

Usage:
  python -m scripts.train_bc_model \
    --data data/offline_trajectories.json \
    --save_dir checkpoints/bc_grid \
    --model_name Qwen/Qwen3-1.7B \
    --epochs 10 \
    --max_lengths 1024 1536 2048 \
    --temperatures 0.1 0.5 1.0 \
    --lora_r 16 --lora_alpha 32
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
    peft_model_ref,
    tokenizer,
    save_dir,
    offline_data,
    eval_data,
    eval_n,
    eval_budget,
    epoch_metrics,
    base_model_name,
    temperature=None,
):
    """Build a TrainerCallback class that evals after each epoch.

    Strategy: at each epoch save a MERGED checkpoint using peft's reversible
    merge_adapter()/unmerge_adapter(), so training continues uninterrupted.
    """
    from transformers import TrainerCallback

    class EvalAfterEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            print(f"\n{'─'*50}")
            print(f"  Epoch {epoch} complete — saving & evaluating")
            print(f"{'─'*50}")

            peft_model = peft_model_ref[0]

            # Save merged checkpoint using reversible merge
            peft_model.eval()
            try:
                # merge_adapter() is reversible (unlike merge_and_unload)
                peft_model.merge_adapter()
                # Save the underlying base model with merged weights
                peft_model.base_model.model.save_pretrained(epoch_dir)
                tokenizer.save_pretrained(epoch_dir)
            finally:
                # Always unmerge so training can continue
                peft_model.unmerge_adapter()
                peft_model.train()

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
    fig.suptitle("BC Training (LoRA) — Per-Epoch Learning Curves", fontsize=15, fontweight="bold")

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
        ax.set_title(f"Grid Search (LoRA) — {title}")
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
    """Train one model with LoRA at the given max_length.

    Returns (tokenizer, epoch_metrics, train_time).
    Saved checkpoint is a MERGED model (standard HF format).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    pairs = build_prompt_target_pairs(args.data)
    print(f"  Loaded {len(pairs)} (prompt, action) pairs")
    if not pairs:
        print("  No pairs found — skipping.")
        return None, [], 0

    # ── Load base model & tokenizer ─────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # ── Apply LoRA ──────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_get_target_modules(model, args.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")

    # Enable gradient checkpointing for memory efficiency
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ── Tokenize ────────────────────────────────────────────────
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

    # ── Callbacks ───────────────────────────────────────────────
    epoch_metrics = []
    peft_model_ref = [model]  # mutable ref for callback

    callbacks = []
    if not args.skip_eval:
        EvalCB = make_eval_callback_class(
            peft_model_ref=peft_model_ref,
            tokenizer=tokenizer,
            save_dir=run_dir,
            offline_data=args.data,
            eval_data=args.eval_data,
            eval_n=args.eval_n,
            eval_budget=args.eval_budget,
            epoch_metrics=epoch_metrics,
            base_model_name=args.model_name,
            temperature=None,  # temperature searched separately in grid
        )
        callbacks.append(EvalCB())

    # ── Train ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="no",
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        # LoRA-specific: warmup helps with small datasets
        warmup_ratio=0.06,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        callbacks=callbacks,
    )

    print(f"\n  Training with LoRA for {args.epochs} epochs")
    print(f"  LR={args.lr}  BS={args.batch_size}  MaxLen={max_length}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # ── Save final merged model ─────────────────────────────────
    # Merge LoRA weights into base model so the saved checkpoint
    # is a standard HF model — learning_agent.py loads it normally.
    print(f"  Merging LoRA into base model and saving to {run_dir}")
    peft_model = peft_model_ref[0]

    # Save LoRA adapter before merging (small, useful for reference)
    adapter_dir = os.path.join(run_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)

    # Merge and save as standard HF model
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)

    return tokenizer, epoch_metrics, train_time


def _get_target_modules(model, target_modules_arg):
    """Determine which layers to apply LoRA to.

    If user passed explicit modules, use those.
    Otherwise auto-detect q_proj/v_proj for typical architectures.
    """
    if target_modules_arg:
        return target_modules_arg.split(",")

    # Auto-detect: look at the model's module names for common patterns
    module_names = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        if parts:
            module_names.add(parts[-1])

    # Qwen, Llama, Mistral all use q_proj, k_proj, v_proj, o_proj
    qkv_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    found_qkv = qkv_modules & module_names
    if found_qkv:
        # Apply to all attention projections for best quality
        targets = sorted(found_qkv)
        print(f"  Auto-detected LoRA targets: {targets}")
        return targets

    # Fallback for other architectures
    attention_patterns = {"query", "key", "value", "dense", "attention"}
    found = attention_patterns & module_names
    if found:
        targets = sorted(found)
        print(f"  Auto-detected LoRA targets: {targets}")
        return targets

    # Last resort: let peft auto-detect
    print(f"  Using peft auto-detection for LoRA target modules")
    return None


# ─── Main (grid search) ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train BC model (LoRA) with grid search over temperature & context window"
    )
    # Training args (fixed across grid)
    parser.add_argument("--data", "-d", type=str, default="data/offline_trajectories.json")
    parser.add_argument("--save_dir", "-o", type=str, default="checkpoints/bc_grid")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (LoRA typically needs higher LR, e.g. 2e-4)")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (higher = more capacity, default 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling (default 32, typically 2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default 0.05)")
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="Comma-separated module names (default: auto-detect q/k/v/o_proj)")

    # Grid search params
    parser.add_argument("--max_lengths", type=int, nargs="+", default=[1024, 1536, 2048],
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
    try:
        import peft
    except ImportError:
        print("Install: pip install peft")
        raise SystemExit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    combos = list(itertools.product(args.max_lengths, args.temperatures))
    print(f"\n{'═'*60}")
    print(f"  GRID SEARCH (LoRA): {len(args.max_lengths)} max_lengths × {len(args.temperatures)} temperatures")
    print(f"  = {len(combos)} evaluation points")
    print(f"  (Only {len(args.max_lengths)} training runs needed — temperature is eval-time)")
    print(f"  max_lengths:  {args.max_lengths}")
    print(f"  temperatures: {args.temperatures}")
    print(f"  epochs: {args.epochs}  lr: {args.lr}  batch_size: {args.batch_size}")
    print(f"  LoRA: r={args.lora_r}  alpha={args.lora_alpha}  dropout={args.lora_dropout}")
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
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
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
    print(f"  GRID SEARCH RESULTS (LoRA)")
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