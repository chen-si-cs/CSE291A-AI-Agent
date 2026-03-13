"""
RLAgent v3: policy network trained with REINFORCE / GRPO on ArcEnv rewards.

Key changes from v2:
  1. Full ARC DSL function list (all unary + useful binary ops)
  2. Higher entropy_beta (0.10) + linear annealing → prevents policy collapse
  3. Larger network with LayerNorm → more stable gradients
  4. GRPO: skip update (not fall back to REINFORCE) when std_R < 1e-6
     The fallback was the collapse trigger — merging k=8 identical trajectories
     into one giant REINFORCE update wiped out learned policies.
  5. Advantage normalisation inside REINFORCE update
  6. AdamW + CosineAnnealingLR for better generalisation
  7. Save/load includes optimizer and scheduler state

State (110 dims):
  4   dynamic: turn, budget, chain_depth, last_ok
  30  color histograms: test_in, train_in, train_out  (10 each)
  75  pixel 5x5: test_in, train_in, train_out          (25 each)
  1   train_acc: accuracy of last_var vs first train output
"""

from __future__ import annotations
import os
from typing import Any, List, Optional, Tuple

from agents.base_agent import BaseAgent

# ── Full DSL function catalog ─────────────────────────────────────────────────
# Only functions that are truly unary (1 arg) and return something chainable.
# Verified against dsl.py.

# Unary grid→grid (1 argument, takes a Grid, returns a Grid)
UNARY_GRID = [
    "hmirror",    # grid→grid
    "vmirror",    # grid→grid
    "dmirror",    # grid→grid
    "cmirror",    # grid→grid
    "rot90",      # grid→grid
    "rot180",     # grid→grid
    "rot270",     # grid→grid
    "tophalf",    # grid→grid
    "bottomhalf", # grid→grid
    "lefthalf",   # grid→grid
    "righthalf",  # grid→grid
    "trim",       # grid→grid
    "compress",   # grid→grid (removes frontiers)
    "identity",   # any→any
]

# Unary grid→non-grid (1 argument, returns objects/indices/metadata)
UNARY_META = [
    "palette",       # element→frozenset of colors
    "numcolors",     # element→int
    "mostcolor",     # element→int
    "leastcolor",    # element→int
    "height",        # piece→int
    "width",         # piece→int
    "shape",         # piece→(h,w)
    "backdrop",      # patch→indices
    "delta",         # patch→indices
    "asindices",     # grid→indices (all cells)
    "toindices",     # patch→indices
    "partition",     # grid→objects (by color)
    "fgpartition",   # grid→objects (foreground only, no bg)
    "asobject",      # grid→object
    "objects",       # grid→objects — NOTE: needs 3 extra bool args; will use defaults via wrapper
    "frontiers",     # grid→objects
    "corners",       # patch→indices (4 corners)
    "outbox",        # patch→indices
    "inbox",         # patch→indices
]

# Binary ops: (var1, var2) → result
# Only ops that genuinely take exactly 2 args of inventory-variable type
BINARY_GG = [
    "paint",         # (grid, object)→grid  — most versatile
    "cover",         # (grid, patch)→grid   — remove object from grid
    "underpaint",    # (grid, object)→grid  — paint only on bg
    "hconcat",       # (grid, grid)→grid    — horizontal concat
    "vconcat",       # (grid, grid)→grid    — vertical concat
    "combine",       # (container, container)→container
    "intersection",  # (frozenset, frozenset)→frozenset
    "difference",    # (frozenset, frozenset)→frozenset
    "subgrid",       # (patch, grid)→grid   — crop to object
    "occurrences",   # (grid, object)→indices
]

_UG = len(UNARY_GRID)   # 14
_UM = len(UNARY_META)   # 19
_B  = len(BINARY_GG)    # 10

# Action layout:
#   0                            → submit
#   [1 .. UG]                    → unary_grid(func, I)
#   [UG+1 .. 2*UG]               → unary_grid(func, last_var)
#   [2*UG+1 .. 2*UG+UM]          → unary_meta(func, I)
#   [2*UG+UM+1 .. 2*UG+2*UM]     → unary_meta(func, last_var)
#   [2*UG+2*UM+1 .. 2*UG+2*UM+B] → binary(func, I, last_var)
#   [..+B+1 .. ..+2*B]           → binary(func, last_var, I)
NUM_ACTIONS = 1 + 2 * _UG + 2 * _UM + 2 * _B   # 87

# ── State dimensions ──────────────────────────────────────────────────────────
GRID_TARGET = 5
GRID_FEAT   = GRID_TARGET ** 2   # 25
STATE_DIM   = 4 + 10 * 3 + 25 * 3 + 1   # 110

ENTROPY_BETA_INIT  = 0.10
ENTROPY_BETA_FINAL = 0.02
GRAD_CLIP_NORM     = 0.5


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _parse_grid_str(grid_str: str, H: int, W: int) -> List[int]:
    digits = [int(c) for c in grid_str if c.isdigit()]
    target = H * W
    if len(digits) >= target:
        return digits[:target]
    return digits + [0] * (target - len(digits))


def _color_histogram(flat: List[int]) -> List[float]:
    total  = max(len(flat), 1)
    counts = [0] * 10
    for v in flat:
        if 0 <= v <= 9:
            counts[v] += 1
    return [c / total for c in counts]


def _downsample(flat: List[int], H: int, W: int,
                th: int = GRID_TARGET, tw: int = GRID_TARGET) -> List[float]:
    if H == 0 or W == 0:
        return [0.0] * (th * tw)
    result = []
    for i in range(th):
        for j in range(tw):
            si = min(int(i * H / th), H - 1)
            sj = min(int(j * W / tw), W - 1)
            result.append(flat[si * W + sj] / 9.0)
    return result


def _grid_features(grid_str: str, shape_str: str) -> Tuple[List[float], List[float]]:
    try:
        H, W = map(int, shape_str.split("x"))
    except Exception:
        H, W = 1, 1
    flat = _parse_grid_str(grid_str, H, W)
    return _downsample(flat, H, W), _color_histogram(flat)


# ── Puzzle feature extraction ─────────────────────────────────────────────────

def _extract_puzzle_features(obs: dict) -> dict:
    test_inp  = obs.get("test_input", {})
    test_pix, test_hist = _grid_features(
        test_inp.get("grid", ""), test_inp.get("shape", "1x1"))

    train_examples = obs.get("train_examples", [])
    train_in_pix   = [0.0] * GRID_FEAT
    train_out_pix  = [0.0] * GRID_FEAT
    train_in_hist  = [0.0] * 10
    train_out_hist = [0.0] * 10
    train_out_grid_raw = None

    if train_examples:
        ex = train_examples[0]
        train_in_pix,  train_in_hist  = _grid_features(
            ex.get("input_grid",  ""), ex.get("input_shape",  "1x1"))
        train_out_pix, train_out_hist = _grid_features(
            ex.get("output_grid", ""), ex.get("output_shape", "1x1"))
        try:
            oh, ow = map(int, ex.get("output_shape", "1x1").split("x"))
            flat = _parse_grid_str(ex.get("output_grid", ""), oh, ow)
            train_out_grid_raw = tuple(
                tuple(flat[i * ow:(i + 1) * ow]) for i in range(oh))
        except Exception:
            pass

    return {
        "static_vec": (test_hist + train_in_hist + train_out_hist
                       + test_pix + train_in_pix + train_out_pix),
        "train_out_grid": train_out_grid_raw,
    }


def _probe_train_accuracy(last_var_value, train_out_grid) -> float:
    if train_out_grid is None or not isinstance(last_var_value, tuple):
        return 0.0
    try:
        if len(last_var_value) != len(train_out_grid):
            return 0.0
        total = correct = 0
        for r1, r2 in zip(last_var_value, train_out_grid):
            if not isinstance(r1, tuple) or len(r1) != len(r2):
                return 0.0
            for a, b in zip(r1, r2):
                total += 1
                correct += int(a == b)
        return correct / total if total > 0 else 0.0
    except Exception:
        return 0.0


def _obs_to_state(obs: dict, max_steps: int, puzzle_cache: dict,
                  last_var_value=None) -> List[float]:
    turn        = obs.get("turn", 0)
    budget      = obs.get("budget_remaining", max_steps)
    inv         = obs.get("inventory") or {}
    chain_depth = len([k for k in inv if k != "I" and not k.startswith("train_")])
    last_ok     = 1.0 if (obs.get("last_action_result") or {}).get("action_ok", True) else 0.0

    dynamic   = [turn / max(1, max_steps), budget / max(1, max_steps),
                 min(chain_depth / 5.0, 1.0), last_ok]
    train_acc = [_probe_train_accuracy(last_var_value, puzzle_cache.get("train_out_grid"))]
    return dynamic + puzzle_cache.get("static_vec", [0.0] * 105) + train_acc


# ── Action mapping ────────────────────────────────────────────────────────────

def _action_index_to_env_action(idx: int, observation: dict) -> dict:
    inv       = observation.get("inventory") or {}
    non_input = [k for k in inv if k != "I" and not k.startswith("train_")]
    last_var  = non_input[-1] if non_input else "I"

    if idx == 0:
        return {"type": "submit", "answer": last_var,
                "_raw_input_submit": (not non_input)}

    idx -= 1
    if idx < _UG:
        return {"type": "execute", "function": UNARY_GRID[idx], "args": ["I"]}
    idx -= _UG
    if idx < _UG:
        return {"type": "execute", "function": UNARY_GRID[idx], "args": [last_var]}
    idx -= _UG
    if idx < _UM:
        func = UNARY_META[idx]
        arg  = "I"
        # objects() needs (grid, univalued, diagonal, without_bg) — use common defaults
        if func == "objects":
            return {"type": "execute", "function": func, "args": [arg, True, False, True]}
        return {"type": "execute", "function": func, "args": [arg]}
    idx -= _UM
    if idx < _UM:
        func = UNARY_META[idx]
        arg  = last_var
        if func == "objects":
            return {"type": "execute", "function": func, "args": [arg, True, False, True]}
        return {"type": "execute", "function": func, "args": [arg]}
    idx -= _UM
    if idx < _B:
        return {"type": "execute", "function": BINARY_GG[idx], "args": ["I", last_var]}
    idx -= _B
    if idx < _B:
        return {"type": "execute", "function": BINARY_GG[idx], "args": [last_var, "I"]}
    return {"type": "execute", "function": "identity", "args": ["I"]}


def _make_masks(turn: int, chain_depth: int) -> List[bool]:
    mask = [False] * NUM_ACTIONS
    if turn < 4:
        mask[0] = True
    if chain_depth == 0:
        # mask unary_grid(last_var)
        for i in range(1 + _UG, 1 + 2 * _UG):
            mask[i] = True
        # mask unary_meta(last_var)
        for i in range(1 + 2 * _UG + _UM, 1 + 2 * _UG + 2 * _UM):
            mask[i] = True
        # mask all binary ops
        for i in range(1 + 2 * _UG + 2 * _UM, NUM_ACTIONS):
            mask[i] = True
    return mask


# ── Agent class ───────────────────────────────────────────────────────────────

class RLAgent(BaseAgent):

    def __init__(
        self,
        state_dim:       int   = STATE_DIM,
        num_actions:     int   = NUM_ACTIONS,
        hidden_dim:      int   = 256,
        max_steps:       int   = 20,
        checkpoint_path: Optional[str] = None,
        entropy_beta:    float = ENTROPY_BETA_INIT,
    ):
        self.state_dim      = state_dim
        self.num_actions    = num_actions
        self.hidden_dim     = hidden_dim
        self.max_steps      = max_steps
        self.entropy_beta   = entropy_beta
        self._total_updates = 0
        self._puzzle_cache: dict = {}
        self._last_var_value     = None
        self._env_ref            = None
        self._policy   = None
        self._device   = None
        self._optimizer = None
        self._scheduler = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            self.load(checkpoint_path)
        else:
            self._build_policy()

    def _build_policy(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError("RLAgent requires: pip install torch") from e
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Policy(nn.Module):
            def __init__(self, state_dim, num_actions, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, num_actions),
                )
            def forward(self, x):
                return self.net(x)

        self._policy    = Policy(self.state_dim, self.num_actions, self.hidden_dim).to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._policy.parameters(), lr=3e-4, weight_decay=1e-4)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=10000, eta_min=1e-5)

    def _anneal_entropy(self) -> float:
        t = min(self._total_updates / 5000.0, 1.0)
        return ENTROPY_BETA_INIT + t * (ENTROPY_BETA_FINAL - ENTROPY_BETA_INIT)

    def _get_action(self, state: List[float], turn: int = 0,
                    chain_depth: int = 0, deterministic: bool = False):
        import torch
        x      = torch.tensor([state], dtype=torch.float32, device=self._device)
        logits = self._policy(x).squeeze(0).clone()
        masks  = _make_masks(turn, chain_depth)
        for i, m in enumerate(masks):
            if m:
                logits[i] = float("-inf")
        dist     = torch.distributions.Categorical(logits=logits)
        action_t = logits.argmax() if deterministic else dist.sample()
        return action_t.item(), dist.log_prob(action_t), dist.entropy()

    def setup(self, observation: dict) -> None:
        self._puzzle_cache   = _extract_puzzle_features(observation)
        self._last_var_value = None
        self._trajectory: List[Tuple] = []

    def act(self, observation: dict) -> dict:
        turn        = observation.get("turn", 0)
        inv         = observation.get("inventory") or {}
        non_input   = [k for k in inv if k != "I" and not k.startswith("train_")]
        chain_depth = len(non_input)
        state       = _obs_to_state(observation, self.max_steps,
                                    self._puzzle_cache, self._last_var_value)
        action_idx, log_prob, entropy = self._get_action(
            state, turn=turn, chain_depth=chain_depth)
        env_action = _action_index_to_env_action(action_idx, observation)
        self._trajectory.append((state, action_idx, log_prob, entropy, 0.0))
        return env_action

    def on_step_result(self, observation: dict, reward: float,
                       done: bool, info: dict) -> None:
        if self._trajectory:
            s, ai, lp, ent, _ = self._trajectory[-1]
            self._trajectory[-1] = (s, ai, lp, ent, reward)
        stored_name = (info or {}).get("stored")
        if stored_name and self._env_ref is not None:
            try:
                self._last_var_value = self._env_ref.inventory.get(stored_name)
            except Exception:
                self._last_var_value = None

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        pass

    def get_trajectory(self) -> List[Tuple]:
        return list(getattr(self, "_trajectory", []))

    # ── REINFORCE ─────────────────────────────────────────────────────────────

    def reinforce_update(self, gamma: float = 0.99,
                         baseline: Optional[float] = None) -> float:
        import torch
        traj = self.get_trajectory()
        if not traj:
            return 0.0

        rewards = [r for (*_, r) in traj]
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        bl  = baseline if baseline is not None else 0.0
        adv = [ret - bl for ret in returns]

        # Normalise advantages
        if len(adv) > 1:
            mean_a = sum(adv) / len(adv)
            std_a  = (sum((a - mean_a) ** 2 for a in adv) / len(adv)) ** 0.5
            if std_a > 1e-8:
                adv = [(a - mean_a) / (std_a + 1e-8) for a in adv]

        log_probs = torch.stack([lp  for (_, _, lp, _,  _) in traj])
        entropies = torch.stack([ent for (_, _, _,  ent, _) in traj])
        adv_t     = torch.tensor(adv, device=self._device, dtype=torch.float32)

        beta = self._anneal_entropy()
        loss = -(log_probs * adv_t).sum() - beta * entropies.sum()
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), GRAD_CLIP_NORM)
        self._optimizer.step()
        self._scheduler.step()
        self._total_updates += 1
        return loss.item()

    # ── GRPO ──────────────────────────────────────────────────────────────────

    def grpo_update(self, trajectories: List[List[Tuple]],
                    gamma: float = 0.99) -> float:
        import torch
        if not trajectories:
            return 0.0

        group_returns: List[float] = []
        for traj in trajectories:
            rewards = [r for (*_, r) in traj]
            R = 0.0
            for r in reversed(rewards):
                R = r + gamma * R
            group_returns.append(R)

        mean_R = sum(group_returns) / len(group_returns)
        std_R  = (sum((r - mean_R) ** 2 for r in group_returns) / len(group_returns)) ** 0.5

        # Skip — no learning signal when all outcomes are identical
        if std_R < 1e-6:
            self._total_updates += 1
            return 0.0

        all_log_probs: list       = []
        all_entropies: list       = []
        all_advantages: List[float] = []

        for traj, R in zip(trajectories, group_returns):
            adv = (R - mean_R) / (std_R + 1e-8)
            for (_, _, lp, ent, _) in traj:
                all_log_probs.append(lp)
                all_entropies.append(ent)
                all_advantages.append(adv)

        if not all_log_probs:
            return 0.0

        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        adv_t     = torch.tensor(all_advantages, device=self._device, dtype=torch.float32)

        beta = self._anneal_entropy()
        loss = -(log_probs * adv_t).sum() - beta * entropies.sum()
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), GRAD_CLIP_NORM)
        self._optimizer.step()
        self._scheduler.step()
        self._total_updates += 1
        return loss.item()

    def act_deterministic(self, observation: dict) -> dict:
        turn      = observation.get("turn", 0)
        inv       = observation.get("inventory") or {}
        non_input = [k for k in inv if k != "I" and not k.startswith("train_")]
        state     = _obs_to_state(observation, self.max_steps,
                                  self._puzzle_cache, self._last_var_value)
        action_idx, _, _ = self._get_action(
            state, turn=turn, chain_depth=len(non_input), deterministic=True)
        return _action_index_to_env_action(action_idx, observation)

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy":        self._policy.state_dict(),
            "optimizer":     self._optimizer.state_dict(),
            "scheduler":     self._scheduler.state_dict(),
            "total_updates": self._total_updates,
        }, path)

    def load(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        if self._policy is None:
            self._build_policy()
        self._policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            try:
                self._optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if "scheduler" in ckpt:
            try:
                self._scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        self._total_updates = ckpt.get("total_updates", 0)
        if self._device is not None:
            self._policy.to(self._device)