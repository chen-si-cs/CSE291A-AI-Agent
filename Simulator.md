# ARC-AGI DSL Exploration Simulator (Team 1 Project for CSE291)

The `main` branch is the template environment for the ARC simulator.

## Branch Guide

### LLM API Agent
Switch to branch `zejia` for general LLM API agent tests.  
Switch to branch `jiarui` for ablations on LLM API agents.

### SFT Agent
Switch to branch `shuhao-SFT-modified`.

### RL Agent
Switch to branch `feiyang`.

---

# ARC-AGI DSL Exploration Simulator

A text-adventure-style environment where an AI agent solves ARC-AGI puzzles by composing Domain-Specific Language (DSL) operations on an inventory of intermediate results.

---

## Quick Start

```bash
# Human play
python -m scripts.play --puzzle 00d62c1b

# Evaluate random baseline
python -m scripts.evaluate -a random -d data/train -n 10

# Evaluate LLM agent (requires API)
python -m scripts.evaluate_llm_agent --model api-llama-4-scout -n 5
```

---

## Code Structure

```
finalPJ/
├── env.py              # ArcEnv: main simulator (Gym-like API)
├── dsl_engine.py       # DSLEngine: 160+ DSL functions, type-safe execution
├── inventory.py        # Inventory: named variable store (I, x1, x2, ...)
├── puzzle_db.py        # PuzzleDB: loads ARC JSON from data/train, data/evaluation
├── renderer.py         # Text/visual rendering of grids, objects
├── text_parser.py      # Parses text commands → action dicts (for LLM)
├── reward.py           # Exact match, partial match, step cost
├── solver_trajectory.py # Extracts action sequences from solvers.py
├── observation_format.py
├── constants.py, arc_types.py, dsl.py
│
├── agents/
│   ├── base_agent.py   # Abstract interface: setup(), act(), on_episode_end()
│   ├── random_agent.py # Random baseline
│   ├── llm_agent.py    # LLM agent (API call → text → parse)
│   ├── learning_agent.py # BC (Qwen3-1.7B) + lookup fallback
│   ├── rl_agent.py     # REINFORCE, 29 unary DSL actions
│   └── grid_rl_agent.py # GridRL: 57-dim state, 40 macro-actions
│
├── data/
│   ├── train/          # ARC training puzzles
│   ├── evaluation/     # ARC evaluation puzzles
│   └── offline_trajectories.json  # For BC (from build_offline_data)
│
└── scripts/
    ├── play.py         # Human interactive mode
    ├── demo.py         # Step-by-step demo of puzzle 00d62c1b
    ├── build_offline_data.py   # Solver trajectories → JSON
    ├── train_bc_model.py      # Train Qwen3-1.7B for BC
    ├── train_rl_agent.py     # Train RLAgent (REINFORCE)
    ├── train_grid_rl_agent.py # Train GridRLAgent
    ├── evaluate.py     # random / learning / rl
    ├── evaluate_llm_agent.py  # LLM agent
    └── evaluate_grid_rl_agent.py # GridRL agent
```

---

## Agents

| Agent | State | Action Space | Training | Evaluation Script |
|-------|-------|--------------|----------|-------------------|
| **Random** | — | Random DSL calls | — | `evaluate -a random` |
| **LLM** | Text observation | LLM generates text → parse | — | `evaluate_llm_agent` |
| **Learning** | Text observation | Qwen3-1.7B + lookup fallback | BC on trajectories | `evaluate -a learning` |
| **RL** | 4-dim (turn, budget, inv_size, last_success) | 29 unary DSL + submit | REINFORCE | `evaluate -a rl` |
| **GridRL** | 57-dim (grid features, symmetry, etc.) | 40 macro-actions | REINFORCE | `evaluate_grid_rl_agent` |

### RandomAgent

Baseline. Picks random DSL functions with random arguments. Almost never solves.

### LLMAgent

Uses an LLM API to generate text commands. Parses response into action dicts. Requires `llm_call` function. Evaluated via `evaluate_llm_agent.py` (supports `api-llama-4-scout`, etc.).

### LearningAgent (BC)

Behavioral cloning with Qwen3-1.7B. Trained on `offline_trajectories.json`. If model fails or is not loaded, falls back to (puzzle_id, step) lookup or random.

### RLAgent

Policy network: state (4 dims) → action logits. Actions: submit + 29 unary DSL (hmirror, rot90, objects, etc.). Trained with REINFORCE.

### GridRLAgent

Grid-aware RL. State: 57-dim vector (grid shape, colors, symmetry, inventory progress). Actions: 40 macro-actions — templated DSL sequences (e.g. "extract objects + colorfilter", "fill with least color"). Covers more of the DSL functions used in successful solutions.

---

## Core Concept

The agent is dropped into a puzzle. It can **observe** train input/output pairs, **manipulate** grids using DSL operations, and **submit** a final answer.

```
┌─────────────────────────────────────────────────┐
│                  PUZZLE STATE                    │
│                                                  │
│  Train Examples:  [{in→out}, {in→out}, ...]     │
│  Test Input:      stored as variable "I"        │
│                                                  │
│  ┌──────────── INVENTORY ──────────────┐        │
│  │  I   : Grid(6×6)  [test input]     │        │
│  │  x1  : Objects(5)                   │        │
│  │  x2  : Objects(3)  [filtered]       │        │
│  │  x3  : Indices(8)                   │        │
│  │  ...                                │        │
│  └─────────────────────────────────────┘        │
│                                                  │
│  DSL Tools: objects, fill, colorfilter, ...      │
│  Budget: 12/20 steps remaining                   │
└─────────────────────────────────────────────────┘
```

---

## Architecture

```
┌──────────────┐     actions      ┌──────────────────┐
│              │ ───────────────> │                    │
│    Agent     │                  │   ArcEnv           │
│  (LLM / RL) │ <─────────────── │   (Simulator)      │
│              │   observations   │                    │
└──────────────┘                  └────────┬───────────┘
                                           │
                                  ┌────────┴───────────┐
                                  │                     │
                              ┌───┴───┐          ┌──────┴──────┐
                              │  DSL  │          │  Puzzle DB  │
                              │Engine │          │ (JSON files)│
                              └───────┘          └─────────────┘
```

---

## How to Execute Tasks

### 1. Interactive / Demo

```bash
# Human play (random or specific puzzle)
python -m scripts.play
python -m scripts.play --puzzle 00d62c1b --data data/train

# Step-by-step demo (puzzle 00d62c1b)
python -m scripts.demo
python -m scripts.demo --puzzle 00d62c1b --data data/train
```

### 2. Evaluate Agents

```bash
# Random baseline
python -m scripts.evaluate -a random -d data/train -n 50

# Learning (BC) — requires trained model and/or offline data
python -m scripts.evaluate -a learning \
  --model-path checkpoints/bc_qwen \
  --offline-data data/offline_trajectories.json \
  -d data/train -n 50

# RL agent
python -m scripts.evaluate -a rl \
  --rl-checkpoint checkpoints/rl_agent/ckpt_final.pt \
  -d data/train -n 50

# LLM agent (requires API)
python -m scripts.evaluate_llm_agent --model api-llama-4-scout -n 5
python -m scripts.evaluate_llm_agent --puzzle 00d62c1b

# GridRL agent
python -m scripts.evaluate_grid_rl_agent --checkpoint checkpoints/grid_rl/ckpt_final.pt -d data/train -n 50
python -m scripts.evaluate_grid_rl_agent --checkpoint checkpoints/grid_rl/ckpt_final.pt --compare-all --rl-agent-ckpt checkpoints/rl_agent/ckpt_final.pt
```

### 3. Training

```bash
# Build offline trajectories for BC
python -m scripts.build_offline_data -d data/train data/evaluation -o data/offline_trajectories.json

# Train BC model (Qwen3-1.7B)
python -m scripts.train_bc_model -d data/offline_trajectories.json -o checkpoints/bc_qwen -m Qwen/Qwen3-1.7B --epochs 2

# Train RL agent
python -m scripts.train_rl_agent -d data/train -o checkpoints/rl_agent -e 500

# Train GridRL agent
python -m scripts.train_grid_rl_agent -d data/train --episodes 5000 --save-dir checkpoints/grid_rl
python -m scripts.train_grid_rl_agent -d data/train --smoke-test  # quick test
```

---

## Common Arguments

| Argument | Meaning | Example |
|----------|---------|---------|
| `-d` / `--data` | Data dirs | `data/train` `data/evaluation` |
| `-n` | Number of puzzles | `50` |
| `-p` / `--puzzle` | Single puzzle ID | `00d62c1b` |
| `-b` / `--budget` | Max steps per episode | `20` |
| `-o` / `--output` | Save results to JSON | `results.json` |

---

## Environment API (Gym-like)

### `env.reset(puzzle_id=None) → Observation`

Load a puzzle (random if `puzzle_id=None`). Returns the initial observation.

### `env.step(action: Action) → (Observation, Reward, Done, Info)`

Execute one action. Returns updated state.

### Action Types

- **execute**: `{"type": "execute", "function": "objects", "args": ["I", True, False, True], "store_as": "x1"}`
- **inspect**: `{"type": "inspect", "target": "x1"}`
- **submit**: `{"type": "submit", "answer": "O"}`
- **undo**: `{"type": "undo"}`
- **reset_inventory**: `{"type": "reset_inventory"}`
- **train_inspect**: `{"type": "train_inspect", "index": 0, "which": "input"}`
- **list_functions**: `{"type": "list_functions", "filter": "color"}`

### Text Mode (for LLM)

```
> execute objects(I, True, False, True) -> x1
> execute colorfilter(x1, 0) -> x2
> inspect x2
> submit O
```

---

## Reward Design

| Signal | Value | When |
|--------|-------|------|
| **Exact match** | `+1.0` | Submitted grid == expected output |
| **Partial match** | `0.0 < r < 1.0` | Fraction of cells correct |
| **Wrong dimensions** | `-0.1` | Submitted grid has wrong shape |
| **Invalid action** | `-0.05` | Type error, missing variable |
| **Step cost** | `-0.01` | Each step |
| **Budget exceeded** | `-0.5` | Exceeded max steps |

---

## Configuration

```python
env = ArcEnv(
    data_dirs=["data/train", "data/evaluation"],
    max_steps=20,
    step_cost=0.01,
    render_mode="text",       # "text" | "ansi" | "json"
)
```

---

## Dependencies

- Base: see `requirements.txt`
- BC (LearningAgent): `torch`, `transformers`, `datasets`
- LLM: API client (configured in `evaluate_llm_agent.py`)
