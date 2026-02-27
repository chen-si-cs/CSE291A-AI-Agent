"""
RLAgent: policy network trained with REINFORCE (or similar) on ArcEnv rewards.

State = compact vector (turn, budget, inv_size, last_success).
Action = discrete index into a fixed set of env actions (submit + execute with common DSL functions).
Training script: scripts/train_rl_agent.py.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent

# Subset of actions: 0 = submit(last_var), 1..N = execute(func_i, ["I"]) for unary functions
RL_UNARY_FUNCS = [
    "hmirror", "vmirror", "dmirror", "rot90", "rot180", "rot270",
    "tophalf", "bottomhalf", "lefthalf", "righthalf",
    "trim", "compress", "normalize", "toindices",
    "flip", "invert", "identity", "height", "width", "shape",
    "palette", "numcolors", "mostcolor", "leastcolor",
    "backdrop", "delta", "asindices", "partition", "objects",
]
NUM_ACTIONS = 1 + len(RL_UNARY_FUNCS)  # 0 = submit, 1.. = execute with I

STATE_DIM = 4  # turn_norm, budget_norm, inv_count_norm, last_success


def _obs_to_state(obs: dict, max_steps: int = 20) -> List[float]:
    """Convert observation to a fixed-size state vector."""
    turn = obs.get("turn", 0)
    budget = obs.get("budget_remaining", max_steps)
    inv = obs.get("inventory") or {}
    inv_count = len(inv)
    last_result = obs.get("last_action_result") or {}
    last_ok = 1.0 if last_result.get("success", True) else 0.0  # no result => ok
    return [
        turn / max(1, max_steps),
        budget / max(1, max_steps),
        min(inv_count / 10.0, 1.0),
        last_ok,
    ]


def _action_index_to_env_action(idx: int, observation: dict) -> dict:
    """Map discrete action index to env action dict."""
    inv = observation.get("inventory") or {}
    inv_list = list(inv.keys())
    if idx == 0:
        # Submit: use last inventory var or I
        answer = inv_list[-1] if inv_list else "I"
        return {"type": "submit", "answer": answer}
    # Execute unary with I
    func_idx = idx - 1
    if func_idx >= len(RL_UNARY_FUNCS):
        func_idx = 0
    func = RL_UNARY_FUNCS[func_idx]
    arg = "I" if "I" in inv_list else (inv_list[0] if inv_list else "I")
    return {
        "type": "execute",
        "function": func,
        "args": [arg],
    }


class RLAgent(BaseAgent):
    """
    Agent with a small policy network (MLP) that maps state -> action logits.
    Trained with REINFORCE using env rewards. Use scripts/train_rl_agent.py to train.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 64,
        max_steps: int = 20,
        checkpoint_path: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self._policy = None
        self._device = None
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
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_actions),
                )

            def forward(self, x):
                return self.net(x)

        self._policy = Policy(self.state_dim, self.num_actions, self.hidden_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=1e-3)

    def _get_logits_and_action(self, state: List[float], deterministic: bool = False) -> Tuple[Any, int, Any]:
        import torch
        x = torch.tensor([state], dtype=torch.float32, device=self._device)
        logits = self._policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action_idx = logits.argmax(dim=-1).item()
            log_prob = dist.log_prob(torch.tensor([action_idx], device=self._device)).item()
        else:
            action_t = dist.sample()
            action_idx = action_t.item()
            log_prob = dist.log_prob(action_t).item()
        return logits, action_idx, log_prob

    def setup(self, observation: dict) -> None:
        self._trajectory: List[Tuple[List[float], int, float, float]] = []  # (state, action_idx, log_prob, reward)

    def act(self, observation: dict) -> dict:
        state = _obs_to_state(observation, self.max_steps)
        _, action_idx, log_prob = self._get_logits_and_action(state, deterministic=False)
        env_action = _action_index_to_env_action(action_idx, observation)
        self._trajectory.append((state, action_idx, log_prob, 0.0))  # reward filled in on_step_result
        return env_action

    def on_step_result(self, observation: dict, reward: float, done: bool, info: dict) -> None:
        if self._trajectory:
            state, action_idx, log_prob, _ = self._trajectory[-1]
            self._trajectory[-1] = (state, action_idx, log_prob, reward)

    def on_episode_end(self, total_reward: float, steps: int, info: dict) -> None:
        # REINFORCE: use trajectory to compute gradient (done in train_rl_agent)
        pass

    def get_trajectory(self) -> List[Tuple[List[float], int, float, float]]:
        """Return collected (state, action_idx, log_prob, reward) for the current episode."""
        return getattr(self, "_trajectory", [])

    def reinforce_update(self, gamma: float = 0.99) -> float:
        """
        Compute REINFORCE gradient and update policy. Call after each episode.
        Returns mean loss.
        """
        import torch
        traj = self.get_trajectory()
        if not traj:
            return 0.0
        # Returns: G_t = r_t + gamma * r_{t+1} + ...
        rewards = [r for (_, _, _, r) in traj]
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        log_probs = torch.tensor([lp for (_, _, lp, _) in traj], device=self._device, dtype=torch.float32)
        returns_t = torch.tensor(returns, device=self._device, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = -(log_probs * returns_t).sum()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def act_deterministic(self, observation: dict) -> dict:
        """Act with argmax (no exploration). For evaluation."""
        state = _obs_to_state(observation, self.max_steps)
        _, action_idx, _ = self._get_logits_and_action(state, deterministic=True)
        return _action_index_to_env_action(action_idx, observation)

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"policy": self._policy.state_dict()}, path)

    def load(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        if self._policy is None:
            self._build_policy()
        self._policy.load_state_dict(ckpt["policy"])
        if self._device is not None:
            self._policy.to(self._device)
