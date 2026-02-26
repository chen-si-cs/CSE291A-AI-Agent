# CSE291A-AI-Agent

# ARC-AGI DSL Exploration Simulator

A text-adventure-style environment where an AI agent solves ARC-AGI puzzles by composing Domain-Specific Language (DSL) operations on an inventory of intermediate results.

---

## Core Concept

The agent is dropped into a puzzle. It can **observe** train input/output pairs, **manipulate** grids using DSL operations, and **submit** a final answer. Think of it like a craftsman's workbench: the test input grid is placed on the bench, the DSL functions are the tools, and named variables are storage slots where the agent keeps intermediate work products.

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

### Components

| Component | Role |
|-----------|------|
| **`ArcEnv`** | Main simulator. Manages puzzle loading, inventory, action dispatch, scoring. |
| **`DSLEngine`** | Wraps all 160+ DSL functions. Validates argument types, executes operations safely. |
| **`Inventory`** | Named variable store. Each slot holds a typed DSL value (Grid, Object, Indices, etc.). |
| **`PuzzleDB`** | Loads ARC JSON files from `data/train/` and `data/evaluation/`. |
| **`Renderer`** | Converts grids/objects to text or visual representations for the agent. |

---

## Environment API (Gym-like)

### `env.reset(puzzle_id=None) → Observation`

Load a puzzle (random if `puzzle_id=None`). Returns the initial observation containing:
- Train input/output pairs (rendered as text grids)
- Test input grid dimensions and content
- Available DSL functions list
- Initial inventory (just `"I"` = test input)

### `env.step(action: Action) → (Observation, Reward, Done, Info)`

Execute one action. Returns updated state.

### `env.get_observation() → Observation`

Get current state without taking an action (for re-reading).

---

## Action Space

Actions are dictionaries with a `"type"` field:

### 1. `execute` — Call a DSL function

```python
{
    "type": "execute",
    "function": "objects",           # DSL function name
    "args": ["I", True, False, True], # mix of inventory refs (strings) and literals
    "store_as": "x1"                  # inventory slot for result (auto-assigned if omitted)
}
```

The engine resolves string arguments as inventory variable names. Booleans, ints, tuples are passed as literals.

### 2. `inspect` — View an inventory item in detail

```python
{
    "type": "inspect",
    "target": "x1"        # inventory variable name
}
```

Returns a detailed rendering: grid visualization, object cell list, set contents, etc.

### 3. `submit` — Submit an answer

```python
{
    "type": "submit",
    "answer": "x5"        # inventory variable holding the answer grid
}
```

Compares against expected test output. Returns reward.

### 4. `train_inspect` — View a train example more closely

```python
{
    "type": "train_inspect",
    "index": 0,            # which train pair
    "which": "input"       # "input" or "output" or "diff"
}
```

### 5. `undo` — Remove the last inventory entry

```python
{"type": "undo"}
```

### 6. `reset_inventory` — Clear all except `"I"`

```python
{"type": "reset_inventory"}
```

### 7. `list_functions` — Query available DSL functions

```python
{
    "type": "list_functions",
    "filter": "color"      # optional substring filter
}
```

Returns matching function names, signatures, and docstrings.

---

## Text-Adventure Mode (for LLM Agents)

For LLM agents, the environment also accepts **natural-language-like commands** that are parsed into the action dicts above:

```
> execute objects(I, True, False, True) -> x1
> execute colorfilter(x1, 0) -> x2
> inspect x2
> execute fill(I, 4, x3) -> O
> submit O
```

The `TextParser` component converts these strings into action dicts.

---

## Observation Format

Each observation is a dict with structured fields:

```python
{
    "puzzle_id": "00d62c1b",
    "turn": 3,
    "budget_remaining": 17,

    # Only on reset / first turn:
    "train_examples": [
        {
            "input": "6×6 grid\n0 0 0 0 0 0\n0 0 3 0 0 0\n...",
            "output": "6×6 grid\n0 0 0 0 0 0\n0 0 3 0 0 0\n...",
            "diff": "cells changed: (2,2)→4, (3,3)→4"
        },
        ...
    ],

    # Always present:
    "inventory": {
        "I":  {"type": "Grid", "shape": [6,6], "preview": "0 0 0 0 0 0 | 0 0 3 0 0 0 | ..."},
        "x1": {"type": "Objects", "count": 5, "preview": "5 objects, colors={0,3}"},
        "x2": {"type": "Objects", "count": 3, "preview": "3 objects, color=0"},
    },

    # Result of last action:
    "last_action_result": {
        "success": True,
        "stored": "x2",
        "value_summary": "Objects(3): 3 connected regions of color 0",
        "detail": "..."   # present only for inspect actions
    },

    # Feedback signals:
    "message": "Stored result of colorfilter(x1, 0) as x2"
}
```

---

## Reward Design

| Signal | Value | When |
|--------|-------|------|
| **Exact match** | `+1.0` | Submitted grid == expected output |
| **Partial match** | `0.0 < r < 1.0` | Fraction of cells correct (only on submit) |
| **Wrong dimensions** | `-0.1` | Submitted grid has wrong shape |
| **Invalid action** | `-0.05` | Type error, missing variable, bad function name |
| **Step cost** | `-0.01` | Each step to encourage efficiency |
| **Budget exceeded** | `-0.5` | Exceeded max steps (episode ends) |

---

## Constants

The DSL solvers reference named constants (`ZERO`, `ONE`, `THREE`, `T`, `F`, `ORIGIN`, etc.). These are pre-loaded in the engine:

```python
ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN = range(11)
NEG_ONE, NEG_TWO = -1, -2
T, F = True, False
ORIGIN = (0, 0)
UNITY = (1, 1)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)
# ... etc.
```

These are available as literal values the agent can use directly in `args`.

---

## Key Design Decisions

### Why an Inventory System?

The DSL solvers work by assigning intermediate values to variables (`x1`, `x2`, ...). The inventory mirrors this exactly. Each DSL call consumes inventory items and produces a new one — just like the solver scripts.

### Why Text-Adventure Style?

1. **LLM-friendly**: An LLM agent can read the observation as a prompt and emit an action as text.
2. **RL-friendly**: The action space is structured (function name + args), making it tractable for RL with a discrete action head + argument selection.
3. **Debuggable**: A human can play the same interface to understand puzzle strategies.

### How Are Higher-Order Functions Handled?

Many DSL functions (`compose`, `fork`, `lbind`, `rbind`, `matcher`, `chain`) produce **callables**. The inventory stores these as first-class values. The agent can:
- Create a composed function: `execute compose(flip, x3) -> x4`
- Apply it later: `execute mfilter(x2, x4) -> x5`

This is critical — many solvers build custom predicates this way.

### Type System

The engine tracks types for every inventory slot:
- `Grid` — tuple of tuples of ints
- `Object` — frozenset of (color, (row, col))
- `Objects` — frozenset of Object
- `Indices` — frozenset of (row, col)
- `Integer`, `Boolean`, `IntegerTuple`
- `Callable` — a function (from compose/fork/lbind/rbind/etc.)
- `Container` — generic

This allows the engine to give meaningful error messages when the agent passes wrong types.

---

## Typical Episode Flow

Here's how the agent would solve puzzle `00d62c1b` (fill enclosed regions with color 4):

```
RESET puzzle=00d62c1b
  → Agent sees 5 train examples showing green(3) outlines getting yellow(4) fills

STEP 1: execute objects(I, True, False, False) -> x1
  → x1 = 12 objects (connected single-color regions, non-diagonal, including bg)

STEP 2: execute colorfilter(x1, 0) -> x2
  → x2 = 7 objects (only the black/background regions)

STEP 3: execute rbind(bordering, I) -> x3
  → x3 = <callable: checks if a patch borders the grid edge>

STEP 4: execute compose(flip, x3) -> x4
  → x4 = <callable: checks if a patch does NOT border the grid edge>

STEP 5: execute mfilter(x2, x4) -> x5
  → x5 = Indices (all cells in non-bordering black regions = the enclosed areas)

STEP 6: execute fill(I, 4, x5) -> O
  → O = Grid(20×20) with enclosed regions filled yellow

STEP 7: submit O
  → reward = 1.0, done = True ✓
```

---

## File Structure

```
arc_dsl_sim/
├── README.md
├── env.py              # ArcEnv: main simulator class
├── dsl_engine.py       # DSLEngine: wraps dsl.py, safe execution, type tracking
├── inventory.py        # Inventory: named variable store with type metadata
├── renderer.py         # Text/visual rendering of grids, objects, diffs
├── puzzle_db.py        # Loads and indexes ARC JSON puzzle files
├── text_parser.py      # Parses text commands into action dicts
├── reward.py           # Reward computation (exact match, partial, shaping)
├── constants.py        # DSL constants (ZERO, ONE, T, F, ORIGIN, etc.)
├── dsl.py              # Original DSL (unchanged)
├── arc_types.py        # Original type definitions (unchanged)
├── agents/
│   ├── base_agent.py   # Abstract agent interface
│   ├── llm_agent.py    # LLM agent (calls OpenAI/Anthropic API)
│   ├── random_agent.py # Random baseline
│   └── rl_agent.py     # RL agent stub (action embedding + policy network)
├── data/
│   ├── train/          # ARC training puzzles (JSON)
│   └── evaluation/     # ARC evaluation puzzles (JSON)
└── scripts/
    ├── play.py         # Human interactive mode
    ├── evaluate.py     # Run agent on puzzle set, collect metrics
    └── demo.py         # Run a single puzzle with verbose output
```

---

## Agent Interface

```python
class BaseAgent:
    def setup(self, observation: dict) -> None:
        """Called on env.reset(). Agent studies the train examples."""
        ...

    def act(self, observation: dict) -> dict:
        """Given current observation, return an action dict."""
        ...

    def on_episode_end(self, total_reward: float, steps: int) -> None:
        """Called when episode ends. For learning agents."""
        ...
```

### LLM Agent Flow

```python
agent = LLMAgent(model="claude-sonnet-4-20250514")
obs = env.reset("00d62c1b")
agent.setup(obs)

done = False
while not done:
    action = agent.act(obs)       # LLM generates action from observation prompt
    obs, reward, done, info = env.step(action)

agent.on_episode_end(info["total_reward"], info["steps"])
```

### RL Agent Action Encoding

For RL, the action space is decomposed:
1. **Action type**: `[execute, inspect, submit, undo]` — discrete(4)
2. **Function selection**: index into DSL function list — discrete(~160)
3. **Argument slots** (up to 4): each is either an inventory ref (discrete over current vars) or a literal value (discrete over common constants)
4. **Store name**: auto-incremented or chosen from a small set

This gives a factored action space suitable for policy gradient methods.

---

## Configuration

```python
env = ArcEnv(
    data_dir="data/train",
    max_steps=20,             # budget per episode
    reward_shaping=True,      # partial credit on submit
    render_mode="text",       # "text" | "ansi_color" | "json"
    train_visible=True,       # whether agent sees train examples
    test_output_hidden=True,  # never leak test output during episode
)
```


