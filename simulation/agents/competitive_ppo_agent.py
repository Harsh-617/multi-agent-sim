"""Competitive PPO agent — loads a trained CompetitivePolicyNetwork for inference.

Torch is imported lazily so the rest of the backend can start without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from simulation.agents.base import BaseAgent
from simulation.envs.competitive.actions import Action, ActionType

# Matches ACTION_TYPE_ORDER in adapters/competitive_pettingzoo.py
_IDX_TO_ACTION_TYPE = [
    ActionType.BUILD,
    ActionType.ATTACK,
    ActionType.DEFEND,
    ActionType.GAMBLE,
]

# Action types that use the amount parameter
_AMOUNT_TYPES = {ActionType.BUILD, ActionType.ATTACK}

_DEFAULT_AGENT_DIR = Path("storage/agents/competitive_ppo")

# Canonical key order matching training (competitive_ppo.py OBS_KEYS)
_OBS_KEYS = (
    "step", "own_score", "own_resources", "own_rank",
    "num_active_agents", "opponents_scores",
    "own_action_history", "opponents_recent_actions",
)


def _flatten_obs(obs: dict) -> np.ndarray:
    """Flatten a raw env observation dict into a 1-D float32 array.

    Must produce the same layout as competitive_ppo._flatten_obs used during
    training, but works with *raw* env observations (dicts/scalars) rather
    than pre-converted gymnasium arrays.
    """
    parts: list[np.ndarray] = []
    for key in _OBS_KEYS:
        val = obs.get(key)
        if val is None:
            continue

        if key == "own_action_history":
            # Raw obs is a list of dicts with "type" and "amount"
            if isinstance(val, list):
                rows = []
                for entry in val:
                    onehot = np.zeros(len(_IDX_TO_ACTION_TYPE), dtype=np.float32)
                    for j, at in enumerate(_IDX_TO_ACTION_TYPE):
                        if at.value == entry.get("type"):
                            onehot[j] = 1.0
                            break
                    rows.append(np.append(onehot, float(entry.get("amount", 0.0))))
                arr = np.concatenate(rows).astype(np.float32) if rows else np.zeros(0, dtype=np.float32)
            else:
                arr = np.asarray(val, dtype=np.float32).flatten()
        elif key == "opponents_scores":
            # Raw obs may be a dict {peer_id: float}
            if isinstance(val, dict):
                arr = np.array(list(val.values()), dtype=np.float32)
            else:
                arr = np.asarray(val, dtype=np.float32).flatten()
        elif key == "opponents_recent_actions":
            # Raw obs is a list of lists of dicts, or an ndarray from gym
            if isinstance(val, list):
                rows = []
                for opponent_actions in val:
                    if isinstance(opponent_actions, list):
                        for entry in opponent_actions:
                            onehot = np.zeros(len(_IDX_TO_ACTION_TYPE), dtype=np.float32)
                            if isinstance(entry, dict):
                                for j, at in enumerate(_IDX_TO_ACTION_TYPE):
                                    if at.value == entry.get("type"):
                                        onehot[j] = 1.0
                                        break
                            rows.append(onehot)
                    else:
                        rows.append(np.asarray(opponent_actions, dtype=np.float32).flatten())
                arr = np.concatenate(rows).astype(np.float32) if rows else np.zeros(0, dtype=np.float32)
            else:
                arr = np.asarray(val, dtype=np.float32).flatten()
        else:
            arr = np.asarray(val, dtype=np.float32).flatten()
        parts.append(arr)

    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class CompetitivePPOAgent(BaseAgent):
    """Inference-only agent backed by a trained CompetitivePolicyNetwork.

    Parameters
    ----------
    agent_dir : Path | str
        Directory containing ``policy.pt`` and ``metadata.json``.
    deterministic : bool
        If True (default), use argmax for action type and Beta mean for
        amount.  If False, sample stochastically.
    """

    def __init__(
        self,
        agent_dir: Path | str = _DEFAULT_AGENT_DIR,
        *,
        deterministic: bool = True,
    ) -> None:
        self._agent_dir = Path(agent_dir)
        self._deterministic = deterministic
        self._net: object | None = None  # CompetitivePolicyNetwork (lazy)
        self._device: object | None = None
        self._obs_dim: int | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        import torch
        from simulation.training.competitive_ppo import CompetitivePolicyNetwork

        meta_path = self._agent_dir / "metadata.json"
        policy_path = self._agent_dir / "policy.pt"

        if not meta_path.exists() or not policy_path.exists():
            raise FileNotFoundError(
                f"Competitive PPO artifacts not found in {self._agent_dir}. "
                "Train a policy first with `python -m simulation.training.competitive_ppo`."
            )

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self._obs_dim = metadata["obs_dim"]
        num_action_types = len(metadata.get("action_mapping", _IDX_TO_ACTION_TYPE))

        device = torch.device("cpu")
        self._device = device

        net = CompetitivePolicyNetwork(self._obs_dim, num_action_types)
        net.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
        net.eval()
        self._net = net
        self._loaded = True

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def reset(self, agent_id: str, seed: int) -> None:
        self._ensure_loaded()

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        import torch

        self._ensure_loaded()

        obs_flat = _flatten_obs(observation)

        # Pad or truncate to expected dim
        if self._obs_dim is not None and len(obs_flat) != self._obs_dim:
            padded = np.zeros(self._obs_dim, dtype=np.float32)
            n = min(len(obs_flat), self._obs_dim)
            padded[:n] = obs_flat[:n]
            obs_flat = padded

        obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

        with torch.no_grad():
            logits, alpha, beta_param, _value = self._net(obs_t)  # type: ignore[union-attr]

        if self._deterministic:
            action_idx = int(logits.argmax(dim=-1).item())
            # Beta mean = alpha / (alpha + beta)
            amount = float((alpha / (alpha + beta_param)).squeeze().item())
        else:
            from torch.distributions import Beta, Categorical
            action_idx = int(Categorical(logits=logits).sample().item())
            amount = float(Beta(alpha.squeeze(-1), beta_param.squeeze(-1)).sample().item())

        action_type = _IDX_TO_ACTION_TYPE[action_idx]
        amount = float(np.clip(amount, 0.0, 1.0))

        # Only pass amount for action types that use it
        if action_type in _AMOUNT_TYPES:
            return Action(type=action_type, amount=amount)
        return Action(type=action_type)
