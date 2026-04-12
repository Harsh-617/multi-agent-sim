"""Baseline agents for the Cooperative archetype.

Five deterministic/random policies for smoke-testing and training opponents.

1. RandomAgent    — random task type (uniform), random effort in [0, 1]
2. AlwaysWork     — picks the task type with the longest current queue, effort = 1.0
3. AlwaysIdle     — always IDLE
4. Specialist     — picks one task type at episode start, sticks to it at effort = 1.0
5. Balancer       — cycles through all task types round-robin, effort = 1.0
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

COOPERATIVE_POLICY_REGISTRY: dict[str, type[BaseAgent]] = {
    "random": CooperativeRandomAgent,
    "always_work": AlwaysWorkAgent,
    "always_idle": AlwaysIdleAgent,
    "specialist": SpecialistAgent,
    "balancer": BalancerAgent,
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
