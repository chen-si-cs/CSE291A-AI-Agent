"""
GridRLAgent: a grid-aware RL agent for ARC-AGI.

Compared to RLAgent
-------------------------------
RLAgent:
  state  = 4 numbers (turn, budget, inventory size, last_success)
           → zero information about actual grid content
  actions = 29 unary DSL calls (hmirror(I), rot90(I), ...)
           → covers only 4 of the top-35 most-used functions in real solutions

This agent:
  state  = 57-dim vector encoding grid shape, color palette, symmetry signals,
           size-change direction, and inventory progress
           → the agent can distinguish puzzle types from the grid itself
  actions = 40 macro-actions — meaningful templated DSL sequences
           → includes object extraction, fill, recolor, tiling, spatial ops
           → covers the actual functions that appear in successful solutions

What a macro-action is
----------------------
A macro-action is a short fixed template (1-3 DSL calls) that accomplishes
one meaningful transformation step, using whatever variables are currently
in the inventory. For example:

  MACRO 8:  extract objects (diagonal)
      execute objects(I, True, True, True) -> x{n}

  MACRO 21: fill grid with least color
      execute leastcolor(I) -> tmp
      execute fill(I, tmp, asindices(I)) -> x{n}

  MACRO 31: tile horizontally ×2
      execute hconcat(I, I) -> x{n}

The agent learns WHICH macro to apply at each step given the current grid state.
This is sequential decision-making: the agent takes multiple steps per puzzle,
each step narrowing the gap between current state and target output.

This is exactly "Action = choose the next concept" from the proposal (p.15),
where concept = a DSL operation or short composition of operations.
"""

from __future__ import annotations
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from agents.base_agent import BaseAgent


# ── State representation ─────────────────────────────────────────────────────

STATE_DIM = 57


def obs_to_state(obs: dict) -> List[float]:
    """
    57-dimensional feature vector from a puzzle observation.

    Captures three things:
      1. Puzzle-level features (what does the target transformation look like?)
         — from train examples, visible throughout the episode
      2. Current progress (how far are we from the target?)
         — inventory size, last action success, steps taken
      3. Grid content signals (what kind of grid are we working with?)
         — colors, symmetry hints, size class
    """
    import re as _re
    from collections import Counter as _Counter

    def parse_shape(s):
        m = _re.match(r"(\d+)x(\d+)", s or "1x1")
        return (int(m.group(1)), int(m.group(2))) if m else (1, 1)

    def grid_nums(txt):
        return [int(x) for x in _re.findall(r"\b\d\b", txt)]

    feats = []

    # ── 1. Puzzle-level features (from train examples, 33 dims) ──────────
    for ex in (obs.get("train_examples") or [])[:3]:
        ih, iw = parse_shape(ex.get("input_shape",  "1x1"))
        oh, ow = parse_shape(ex.get("output_shape", "1x1"))
        feats += [
            oh / max(ih, 1),                            # height ratio
            ow / max(iw, 1),                            # width ratio
            1.0 if (oh == ih and ow == iw) else 0.0,   # same size?
            1.0 if (oh * ow > ih * iw)     else 0.0,   # output bigger?
        ]
        in_n  = grid_nums(ex.get("input_grid",  ""))
        out_n = grid_nums(ex.get("output_grid", ""))
        ic, oc = set(in_n), set(out_n)
        union  = ic | oc
        bg = _Counter(in_n).most_common(1)[0][0] if in_n else 0
        diff = ex.get("diff", "")
        feats += [
            len(ic) / 10.0,                                       # n input colors
            len(oc) / 10.0,                                       # n output colors
            len(ic & oc) / max(len(union), 1),                    # color overlap
            len(oc - ic)  / 10.0,                                 # new colors in output
            bg / 9.0,                                             # background color
            min(len(_re.findall(r"->", diff)) / 20.0, 1.0),      # n changed cells
            1.0 if "Shape changed" in diff else 0.0,             # shape changed?
        ]
    while len(feats) < 33:
        feats.append(0.0)
    feats = feats[:33]

    # ── 2. Test input size (2 dims) ───────────────────────────────────────
    test = obs.get("test_input") or {}
    th, tw = parse_shape(test.get("shape", "1x1"))
    feats += [th / 30.0, tw / 30.0]

    # ── 3. Episode progress (5 dims) ──────────────────────────────────────
    max_steps = 20
    feats += [
        obs.get("turn", 0) / max_steps,           # how far into episode
        obs.get("budget_remaining", max_steps) / max_steps,  # budget left
    ]
    inv = obs.get("inventory") or {}
    user_vars = [k for k in inv
                 if k != "I" and not k.startswith("train")]
    feats += [
        min(len(user_vars) / 10.0, 1.0),          # inventory fullness
    ]
    last = obs.get("last_action_result") or {}
    feats += [
        1.0 if last.get("action_ok", True) else 0.0,  # last action ok?
        float(last.get("total_reward", 0.0)),         # cumulative reward so far
    ]

    # ── 4. Latest inventory variable type hints (8 dims) ──────────────────
    # What type is the most recently created variable?
    type_flags = {
        "Grid":    0,
        "Objects": 1,
        "Indices": 2,
        "Integer": 3,
        "Set":     4,
        "Tuple":   5,
        "Boolean": 6,
        "Other":   7,
    }
    last_type_vec = [0.0] * 8
    if user_vars:
        last_var = user_vars[-1]
        vtype = (inv.get(last_var) or {}).get("type", "Other")
        for name, idx in type_flags.items():
            if name.lower() in vtype.lower():
                last_type_vec[idx] = 1.0
                break
        else:
            last_type_vec[7] = 1.0
    feats += last_type_vec

    # ── 5. Symmetry / tiling hints (9 dims, from test input grid text) ────
    test_grid = test.get("grid", "")
    test_nums = grid_nums(test_grid)
    rows = [r.strip() for r in test_grid.strip().split("\n") if r.strip()]
    # Remove "Grid(NxM):" header line
    rows = [r for r in rows if not r.startswith("Grid(")]

    def col_counts(nums):
        c = _Counter(nums)
        return len(c), c.most_common(1)[0][1] if c else 0

    n_colors, max_freq = col_counts(test_nums)
    feats += [
        n_colors / 10.0,
        max_freq / max(len(test_nums), 1),  # dominance of most common color
        th / max(tw, 1),                    # aspect ratio
        1.0 if th == tw else 0.0,           # square?
        1.0 if th <= 5 and tw <= 5 else 0.0,  # small grid (≤5×5)
        1.0 if th > 10 or tw > 10 else 0.0,  # large grid
    ]
    # Horizontal symmetry hint: do rows mirror?
    h_sym = 0.0
    if len(rows) >= 2:
        pairs = [(rows[i], rows[-(i+1)]) for i in range(len(rows)//2)]
        h_sym = sum(1 for a, b in pairs if a == b) / max(len(pairs), 1)
    feats.append(h_sym)
    # Vertical symmetry hint: do columns mirror?
    v_sym = 0.0
    if rows:
        cols_fwd  = [" ".join(r.split()[j] for r in rows if j < len(r.split()))
                     for j in range(tw)]
        cols_rev  = list(reversed(cols_fwd))
        pairs = [(cols_fwd[i], cols_rev[i]) for i in range(len(cols_fwd)//2)]
        v_sym = sum(1 for a, b in pairs if a == b) / max(len(pairs), 1)
    feats.append(v_sym)
    # Tiling hint: first row repeats?
    tiling = 0.0
    if len(rows) >= 2:
        tiling = 1.0 if rows[0] == rows[1] else 0.0
    feats.append(tiling)

    assert len(feats) == STATE_DIM, f"Expected {STATE_DIM}, got {len(feats)}"
    return feats


# ── Macro-action space ────────────────────────────────────────────────────────

def _next_var(obs: dict) -> str:
    """Return the next free variable name (x1, x2, ...)."""
    inv = obs.get("inventory") or {}
    existing = [k for k in inv
                if k.startswith("x") and k[1:].isdigit()]
    used = {int(k[1:]) for k in existing}
    n = 1
    while n in used:
        n += 1
    return f"x{n}"


def _last_var(obs: dict) -> str:
    """Return the most recently created user variable, or 'I'."""
    inv = obs.get("inventory") or {}
    user_vars = [k for k in inv
                 if k != "I" and not k.startswith("train")]
    return user_vars[-1] if user_vars else "I"


def get_macro_actions(obs: dict) -> List[List[dict]]:
    """
    Return the 40 macro-actions as lists of env action dicts,
    parameterised by the current inventory state.

    Each macro produces deterministic, useful DSL operations.
    Multi-step macros are stored as a list of actions executed in sequence.
    """
    nv  = _next_var(obs)   # next free variable, e.g. "x3"
    lv  = _last_var(obs)   # most recently created var, e.g. "x2"

    # Helper: integer for scaling (use output/input ratio from first train example)
    train = (obs.get("train_examples") or [{}])[0]
    def _ratio(dim):
        ih, iw = _pshape(train.get("input_shape",  "1x1"))
        oh, ow = _pshape(train.get("output_shape", "1x1"))
        r = (oh // max(ih,1)) if dim == "h" else (ow // max(iw,1))
        return max(2, min(r, 5))

    n2 = _next_after(obs, 2)   # variable after nv

    macros = [
        # ── 0-5  Simple grid transforms ──────────────────────────────────
        [_ex(f"hmirror(I)",                    nv)],
        [_ex(f"vmirror(I)",                    nv)],
        [_ex(f"rot90(I)",                      nv)],
        [_ex(f"rot180(I)",                     nv)],
        [_ex(f"rot270(I)",                     nv)],
        [_ex(f"trim(I)",                       nv)],

        # ── 6-9  Halving / splitting ──────────────────────────────────────
        [_ex(f"tophalf(I)",                    nv)],
        [_ex(f"bottomhalf(I)",                 nv)],
        [_ex(f"lefthalf(I)",                   nv)],
        [_ex(f"righthalf(I)",                  nv)],

        # ── 10-13  Tiling / concatenation ────────────────────────────────
        [_ex(f"hconcat(I, I)",                 nv)],
        [_ex(f"vconcat(I, I)",                 nv)],
        [_ex(f"hconcat(I, hmirror(I))",        nv)],
        [_ex(f"vconcat(I, vmirror(I))",        nv)],

        # ── 14-17  Scaling ────────────────────────────────────────────────
        [_ex(f"upscale(I, 2)",                 nv)],
        [_ex(f"upscale(I, 3)",                 nv)],
        [_ex(f"hupscale(I, {_ratio('w')})",    nv)],
        [_ex(f"vupscale(I, {_ratio('h')})",    nv)],

        # ── 18-21  Color extraction ───────────────────────────────────────
        [_ex(f"mostcolor(I)",                  nv)],
        [_ex(f"leastcolor(I)",                 nv)],
        [_ex(f"palette(I)",                    nv)],
        [_ex(f"numcolors(I)",                  nv)],

        # ── 22-25  Object extraction ──────────────────────────────────────
        [_ex(f"objects(I, True, False, True)",  nv)],
        [_ex(f"objects(I, True, True,  True)",  nv)],
        [_ex(f"objects(I, False, False, True)", nv)],
        [_ex(f"objects(I, True, False, False)", nv)],

        # ── 26-29  Fill / recolor using most/least color ──────────────────
        [_ex(f"mostcolor(I)",           nv),
         _ex(f"fill(I, {nv}, asindices(I))", n2)],

        [_ex(f"leastcolor(I)",          nv),
         _ex(f"fill(I, {nv}, asindices(I))", n2)],

        [_ex(f"mostcolor(I)",           nv),
         _ex(f"ofcolor(I, {nv})",       n2)],

        [_ex(f"leastcolor(I)",          nv),
         _ex(f"ofcolor(I, {nv})",       n2)],

        # ── 30-33  Operations on last variable ────────────────────────────
        [_ex(f"hmirror({lv})",          nv)],
        [_ex(f"vmirror({lv})",          nv)],
        [_ex(f"merge({lv})",            nv)],
        [_ex(f"first({lv})",            nv)],

        # ── 34-36  Combine / compose last var with I ──────────────────────
        [_ex(f"paint(I, {lv})",         nv)],
        [_ex(f"hconcat(I, {lv})",       nv)],
        [_ex(f"vconcat(I, {lv})",       nv)],

        # ── 37  Submit last variable ──────────────────────────────────────
        [{"type": "submit", "answer": lv}],

        # ── 38  Submit I (identity solution) ─────────────────────────────
        [{"type": "submit", "answer": "I"}],

        # ── 39  Inspect last variable (gather info, no penalty) ───────────
        [{"type": "inspect", "target": lv}],

        # ── 40-54  Combined transform+submit macros ───────────────────────
        # These collapse a 2-step sequence (execute → submit) into one macro
        # choice, giving the agent direct reward attribution for single-
        # transform solutions.  Critical for REINFORCE: without these, the
        # agent must learn a 2-step sequence before seeing any reward, and
        # the probability of randomly hitting the right pair is ~(1/40)² ≈ 0.06%.
        [_ex("hmirror(I)",             nv), {"type": "submit", "answer": nv}],
        [_ex("vmirror(I)",             nv), {"type": "submit", "answer": nv}],
        [_ex("rot90(I)",               nv), {"type": "submit", "answer": nv}],
        [_ex("rot180(I)",              nv), {"type": "submit", "answer": nv}],
        [_ex("rot270(I)",              nv), {"type": "submit", "answer": nv}],
        [_ex("tophalf(I)",             nv), {"type": "submit", "answer": nv}],
        [_ex("bottomhalf(I)",          nv), {"type": "submit", "answer": nv}],
        [_ex("lefthalf(I)",            nv), {"type": "submit", "answer": nv}],
        [_ex("righthalf(I)",           nv), {"type": "submit", "answer": nv}],
        [_ex("hconcat(I, I)",          nv), {"type": "submit", "answer": nv}],
        [_ex("vconcat(I, I)",          nv), {"type": "submit", "answer": nv}],
        [_ex("hconcat(I, hmirror(I))", nv), {"type": "submit", "answer": nv}],
        [_ex("vconcat(I, vmirror(I))", nv), {"type": "submit", "answer": nv}],
        [_ex("upscale(I, 2)",          nv), {"type": "submit", "answer": nv}],
        [_ex("trim(I)",                nv), {"type": "submit", "answer": nv}],

        # ── 55-58  Additional single-step transforms ───────────────────────
        # Each adds puzzles not covered by macros 40-54 (verified on data/train).
        [_ex("dmirror(I)",             nv), {"type": "submit", "answer": nv}],
        [_ex("compress(I)",            nv), {"type": "submit", "answer": nv}],
        [_ex("upscale(I, 3)",          nv), {"type": "submit", "answer": nv}],
        # 2-step: hmirror then rot90
        [_ex("hmirror(I)", nv), _ex(f"rot90({nv})", n2), {"type": "submit", "answer": n2}],
    ]

    assert len(macros) == NUM_MACROS, f"Expected {NUM_MACROS}, got {len(macros)}"
    return macros


NUM_MACROS = 59


def _ex(call: str, var: str) -> dict:
    """Build an execute action dict from a DSL call string."""
    # Parse the function name and args from the call string
    m = re.match(r"(\w+)\((.*)\)$", call.strip(), re.DOTALL)
    if not m:
        return {"type": "error", "message": f"Cannot parse: {call}"}
    func = m.group(1)
    raw  = m.group(2).strip()
    # Split top-level args (respecting nested parens)
    args = _split_args(raw)
    return {"type": "execute", "function": func, "args": args, "store_as": var}


def _split_args(s: str) -> List[str]:
    """Split comma-separated args respecting nested parentheses."""
    args, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1; cur.append(ch)
        elif ch == ")":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return [a for a in args if a]


def _pshape(s: str) -> Tuple[int, int]:
    m = re.match(r"(\d+)x(\d+)", s or "1x1")
    return (int(m.group(1)), int(m.group(2))) if m else (1, 1)


def _next_after(obs: dict, offset: int) -> str:
    """Return variable name `offset` steps after the next free variable."""
    inv = obs.get("inventory") or {}
    existing = {int(k[1:]) for k in inv
                if k.startswith("x") and k[1:].isdigit()}
    n = 1
    while n in existing:
        n += 1
    return f"x{n + offset - 1}"


# ── Policy network ────────────────────────────────────────────────────────────

def build_policy(state_dim: int, n_actions: int, hidden: int = 128):
    """3-layer MLP with dropout for regularisation."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, n_actions),
    )


# ── GridRLAgent ───────────────────────────────────────────────────────────────

class GridRLAgent(BaseAgent):
    """
    Grid-aware RL agent for ARC-AGI.

    Takes one macro-action per step (vs one DSL call per step in
    RLAgent). Each macro-action is a 1-3 step DSL template that produces a
    meaningful intermediate result. The agent observes the grid state at
    each step and learns which macro to apply.

    Training: REINFORCE with a simple moving-average baseline to reduce
    variance (the main instability of the RL agent implementation).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        baseline_decay: float = 0.95,
        entropy_coeff: float = 0.01,
        deterministic: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.hidden_dim      = hidden_dim
        self.gamma           = gamma
        self.baseline_decay  = baseline_decay
        self.entropy_coeff   = entropy_coeff   # entropy bonus keeps policy exploring
        self.deterministic   = deterministic

        self._policy         = None
        self._optimizer      = None
        self._device         = None
        self._baseline       = 0.0    # exponential moving average of returns
        self._build(lr)

        # Episode trajectory: (log_prob, reward, entropy) per macro step
        self._trajectory: List[Tuple] = []

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self.load(checkpoint_path)

    def _build(self, lr: float) -> None:
        import torch, torch.optim as optim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy = build_policy(STATE_DIM, NUM_MACROS, self.hidden_dim).to(
            self._device
        )
        self._optimizer = optim.Adam(self._policy.parameters(), lr=lr)

    # ── BaseAgent interface ───────────────────────────────────────────────

    def setup(self, observation: dict) -> None:
        self._trajectory = []

    def act(self, observation: dict) -> dict:
        """
        Select a macro-action given the current observation.
        Returns the FIRST DSL action of the chosen macro.
        Use get_macro_steps(obs) to retrieve the full sequence after calling act().
        """
        import torch
        state = obs_to_state(observation)
        x     = torch.tensor([state], dtype=torch.float32, device=self._device)
        logits = self._policy(x)
        dist   = torch.distributions.Categorical(logits=logits)

        if self.deterministic:
            idx = int(logits.argmax(dim=-1).item())
        else:
            t   = dist.sample()
            idx = int(t.item())

        log_prob = dist.log_prob(torch.tensor([idx], device=self._device))
        entropy  = dist.entropy()
        self._trajectory.append([log_prob, 0.0, entropy])  # reward filled in later
        self._last_macro_idx = idx
        self._last_obs = observation

        steps = get_macro_actions(observation)[idx]
        return steps[0]

    def get_macro_steps(self, observation: dict) -> List[dict]:
        """Return all DSL steps for the currently chosen macro."""
        if not hasattr(self, "_last_macro_idx"):
            return []
        return get_macro_actions(observation)[self._last_macro_idx]

    def on_step_result(self, observation: dict, reward: float,
                       done: bool, info: dict) -> None:
        if self._trajectory:
            self._trajectory[-1][1] += reward

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        pass  # update called explicitly

    # ── REINFORCE with baseline ────────────────────────────────────────────

    def reinforce_update(self) -> float:
        """
        REINFORCE update with EMA moving-average baseline.

        Advantages are computed as `return − baseline` where baseline is an
        exponential moving average of episode returns across training.  This
        works correctly for any episode length, including the 1-step combined
        macros (40–54):

          correct solve  (+0.98):  0.98 − 0.30 = +0.68  → strong ↑ gradient
          wrong submit   (+0.14):  0.14 − 0.30 = −0.16  → slight  ↓ gradient

        Within-episode normalisation was tried previously but breaks for short
        episodes: std=0 for 1-step episodes → zero gradient regardless of
        outcome, and ±1 for all 2-step episodes regardless of success.

        An entropy bonus keeps the policy from collapsing to one action.
        """
        import torch

        if not self._trajectory:
            return 0.0

        # Discounted returns
        rewards = [r for _, r, *_ in self._trajectory]
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # EMA baseline update (uses the return from the start of the episode)
        episode_return = returns[0]
        self._baseline = (self.baseline_decay * self._baseline
                          + (1 - self.baseline_decay) * episode_return)

        # Advantages: how much better/worse than average?
        advantages = [ret - self._baseline for ret in returns]
        returns_t = torch.tensor(advantages, device=self._device, dtype=torch.float32)

        # Policy gradient loss + entropy bonus
        loss = torch.tensor(0.0, device=self._device)
        for (log_prob, _, entropy), ret in zip(self._trajectory, returns_t):
            loss = loss + (-log_prob * ret - self.entropy_coeff * entropy)

        loss = loss / max(len(self._trajectory), 1)

        self._optimizer.zero_grad()
        loss.backward()
        import torch.nn as nn
        nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._optimizer.step()

        self._trajectory = []
        return float(loss.item())

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy":    self._policy.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "baseline":  self._baseline,
            "hidden":    self.hidden_dim,
        }, path)

    def load(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self._policy.load_state_dict(ckpt["policy"])
        try:
            self._optimizer.load_state_dict(ckpt["optimizer"])
            self._baseline = ckpt.get("baseline", 0.0)
        except Exception:
            pass
        self._policy.to(self._device)

    # ── Introspection ──────────────────────────────────────────────────────

    def top_macros(self, observation: dict, k: int = 5) -> List[Tuple[int, float, str]]:
        """Return top-k (macro_idx, probability, description) for debugging."""
        import torch
        state = obs_to_state(observation)
        x     = torch.tensor([state], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            probs = torch.softmax(self._policy(x), dim=-1)[0]
        top = torch.topk(probs, k)
        macros = get_macro_actions(observation)
        result = []
        for i, p in zip(top.indices.tolist(), top.values.tolist()):
            steps   = macros[i]
            desc    = _describe_macro(steps)
            result.append((i, p, desc))
        return result


def _describe_macro(steps: List[dict]) -> str:
    parts = []
    for s in steps:
        if s.get("type") == "execute":
            args = ", ".join(str(a) for a in s.get("args", []))
            parts.append(f"{s['function']}({args})")
        elif s.get("type") == "submit":
            parts.append(f"submit({s.get('answer', '?')})")
        elif s.get("type") == "inspect":
            parts.append(f"inspect({s.get('target', '?')})")
    return " → ".join(parts)