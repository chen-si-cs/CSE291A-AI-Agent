#!/usr/bin/env python3
"""
Train a small LM (Qwen3-1.7B) for behavioral cloning on offline trajectories.

Reads offline_trajectories.json, formats each (observation, action) as (prompt, target),
and fine-tunes the model with causal LM so it learns to output the next command.

Usage:
  python -m scripts.train_bc_model --data data/offline_trajectories.json --save_dir checkpoints/bc_qwen --model_name Qwen/Qwen3-1.7B
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_parser import action_to_command
from observation_format import format_observation_for_prompt


def build_prompt_target_pairs(trajectories_path: str) -> list[tuple[str, str]]:
    """Load trajectories and return list of (prompt_text, action_command)."""
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


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-1.7B for BC on ARC DSL trajectories")
    parser.add_argument("--data", "-d", type=str, default="data/offline_trajectories.json")
    parser.add_argument("--save_dir", "-o", type=str, default="checkpoints/bc_qwen")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print("Install: pip install torch transformers datasets")
        raise SystemExit(1) from e

    pairs = build_prompt_target_pairs(args.data)
    print(f"Loaded {len(pairs)} (prompt, action) pairs")
    if not pairs:
        print("No (prompt, action) pairs found. Check trajectory JSON has 'steps' with 'observation' and 'action'.")
        raise SystemExit(1)

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
        input_ids_list = []
        labels_list = []
        real_lens = []
        for p, a in zip(examples["prompt"], examples["target"]):
            prompt_part = prompt_prefix + p + action_prefix
            target_part = a + tokenizer.eos_token
            prompt_enc = tokenizer(prompt_part, add_special_tokens=True, truncation=True, max_length=args.max_length - 64)
            target_enc = tokenizer(target_part, add_special_tokens=False, truncation=True, max_length=64)
            p_ids = prompt_enc["input_ids"]
            t_ids = target_enc["input_ids"]
            ids = p_ids + t_ids
            real_len = len(ids)
            pad_len = args.max_length - real_len
            if pad_len > 0:
                ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad_len
            else:
                ids = ids[: args.max_length]
                real_len = args.max_length
            input_ids_list.append(ids)
            real_lens.append(real_len)
            lab = [-100] * len(p_ids) + t_ids + [-100] * max(0, pad_len)
            lab = (lab + [-100] * args.max_length)[: args.max_length]
            labels_list.append(lab)
        return {
            "input_ids": input_ids_list,
            "attention_mask": [[1] * rl + [0] * (args.max_length - rl) for rl in real_lens],
            "labels": labels_list,
        }

    dataset = Dataset.from_dict({
        "prompt": [p for p, _ in pairs],
        "target": [a for _, a in pairs],
    })
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=20,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model and tokenizer saved to {args.save_dir}")


if __name__ == "__main__":
    main()
