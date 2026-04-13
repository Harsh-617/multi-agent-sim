"""Baseline agents for the Cooperative archetype.

Five deterministic/random policies for smoke-testing and training opponents.

1. RandomAgent    — random task type (uniform), random effort in [0, 1]
2. AlwaysWork     — picks the task type with the longest current queue, effort = 1.0
3. AlwaysIdle     — always IDLE
4. Specialist     — picks one task type at episode start, sticks to it at effort = 1.0
5. Balancer       — cycles through all task types round-robin, effort = 1.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from simulation.core.seeding import make_rng
from simulation.envs.cooperative.actions import Action

from simulation.agents.base import BaseAgent


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class CooperativeRandomAgent(BaseAgent):
    """Uniformly random cooperative policy, deterministic given its seed."""

    def __init__(self, num_task_types: int = 3) -> None:
        self._num_task_types = num_task_types
        self._rng: np.random.Generator | None = None

    def reset(self, agent_id: str, seed: int) -> None:
        self._rng = make_rng(seed)

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        assert self._rng is not None, "Must call reset() before act()"
        # Uniform over task types + IDLE
        choices = self._num_task_types + 1  # last index = IDLE
        idx = int(self._rng.integers(choices))
        if idx == self._num_task_types:
            return Action(task_type=None)
        effort = float(self._rng.uniform(0.0, 1.0))
        return Action(task_type=idx, effort_amount=effort)


class AlwaysWorkAgent(BaseAgent):
    """Always picks the task type with the longest current queue, effort = 1.0.

    Reads `obs["obs_vector"]` to find the task type with the highest queue depth
    (positions 1..T of the obs vector, after backlog_norm at position 0).
    Falls back to type 0 if observation is unavailable.
    """

    def __init__(self, num_task_types: int = 3) -> None:
        self._num_task_types = num_task_types

    def reset(self, agent_id: str, seed: int) -> None:
        self._num_task_types_episode = self._num_task_types

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        T = self._num_task_types
        chosen = 0
        if observation is not None:
            vec = observation.get("obs_vector")
            if vec is not None and len(vec) >= T + 1:
                # Positions 1..T are queue_norm per type
                queue_slice = vec[1 : T + 1]
                chosen = int(np.argmax(queue_slice))
        return Action(task_type=chosen, effort_amount=1.0)


class AlwaysIdleAgent(BaseAgent):
    """Always chooses IDLE."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        return Action(task_type=None)


class SpecialistAgent(BaseAgent):
    """Picks one task type randomly at episode start; sticks to it at effort = 1.0."""

    def __init__(self, num_task_types: int = 3) -> None:
        self._num_task_types = num_task_types
        self._chosen_type: int = 0

    def reset(self, agent_id: str, seed: int) -> None:
        rng = make_rng(seed)
        self._chosen_type = int(rng.integers(self._num_task_types))

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        return Action(task_type=self._chosen_type, effort_amount=1.0)


class BalancerAgent(BaseAgent):
    """Cycles through all task types round-robin, effort = 1.0."""

    def __init__(self, num_task_types: int = 3) -> None:
        self._num_task_types = num_task_types
        self._step = 0

    def reset(self, agent_id: str, seed: int) -> None:
        self._step = 0

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        chosen = self._step % self._num_task_types
        self._step += 1
        return Action(task_type=chosen, effort_amount=1.0)


_DEFAULT_COOP_PPO_DIR = Path("storage/agents/cooperative/ppo_shared")


class CooperativePPOAgent(BaseAgent):
    """Inference-only agent backed by a trained SharedPolicyNetwork for cooperative envs.

    Loads artifacts from ``storage/agents/cooperative/ppo_shared/`` by default.
    Train first with: ``python -m simulation.training.cooperative_train``
    """

    def __init__(
        self,
        agent_dir: Path | str = _DEFAULT_COOP_PPO_DIR,
        *,
        deterministic: bool = True,
    ) -> None:
        self._agent_dir = Path(agent_dir)
        self._deterministic = deterministic
        self._net: object | None = None
        self._device: object | None = None
        self._obs_dim: int | None = None
        self._num_action_types: int | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        import torch
        from simulation.training.ppo_shared import SharedPolicyNetwork

        meta_path = self._agent_dir / "metadata.json"
        policy_path = self._agent_dir / "policy.pt"

        if not meta_path.exists() or not policy_path.exists():
            raise FileNotFoundError(
                f"Cooperative PPO artifacts not found in {self._agent_dir}. "
                "Train a policy first with: python -m simulation.training.cooperative_train"
            )

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self._obs_dim = metadata["obs_dim"]
        self._num_action_types = metadata["num_action_types"]

        device = torch.device("cpu")
        self._device = device

        net = SharedPolicyNetwork(self._obs_dim, self._num_action_types)
        net.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=True)
        )
        net.eval()
        self._net = net
        self._loaded = True

    def reset(self, agent_id: str, seed: int) -> None:
        self._ensure_loaded()

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        import torch

        self._ensure_loaded()

        obs_flat = observation.get("obs_vector") if observation is not None else None
        if obs_flat is None:
            obs_flat = np.zeros(self._obs_dim or 1, dtype=np.float32)
        else:
            obs_flat = np.asarray(obs_flat, dtype=np.float32).flatten()

        if self._obs_dim is not None and len(obs_flat) != self._obs_dim:
            padded = np.zeros(self._obs_dim, dtype=np.float32)
            n = min(len(obs_flat), self._obs_dim)
            padded[:n] = obs_flat[:n]
            obs_flat = padded

        obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

        with torch.no_grad():
            logits, alpha, beta_param, _value = self._net(obs_t)  # type: ignore[union-attr]

        num_task_types = (self._num_action_types or 1) - 1  # last index = IDLE

        if self._deterministic:
            action_idx = int(logits.argmax(dim=-1).item())
            effort = float((alpha / (alpha + beta_param)).squeeze().item())
        else:
            from torch.distributions import Beta, Categorical
            action_idx = int(Categorical(logits=logits).sample().item())
            effort = float(Beta(alpha.squeeze(-1), beta_param.squeeze(-1)).sample().item())

        if action_idx >= num_task_types:
            return Action(task_type=None)
        return Action(task_type=action_idx, effort_amount=float(np.clip(effort, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

COOPERATIVE_POLICY_REGISTRY: dict[str, type[BaseAgent]] = {
    "random": CooperativeRandomAgent,
    "always_work": AlwaysWorkAgent,
    "always_idle": AlwaysIdleAgent,
    "specialist": SpecialistAgent,
    "balancer": BalancerAgent,
    "cooperative_ppo": CooperativePPOAgent,
}


def create_cooperative_agent(
    policy_name: str,
    num_task_types: int = 3,
    **kwargs: Any,
) -> BaseAgent:
    """Instantiate a cooperative baseline agent by policy name.

    Raises KeyError if policy_name is not in COOPERATIVE_POLICY_REGISTRY.
    """
    if policy_name not in COOPERATIVE_POLICY_REGISTRY:
        raise KeyError(policy_name)
    cls = COOPERATIVE_POLICY_REGISTRY[policy_name]
    # Agents that accept num_task_types pass it; others (AlwaysIdle) ignore it
    try:
        return cls(num_task_types=num_task_types, **kwargs)
    except TypeError:
        return cls(**kwargs)
