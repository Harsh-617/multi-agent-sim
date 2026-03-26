"""Baseline agents for the Competitive archetype.

Four deterministic/random policies for smoke-testing and as training opponents.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.core.seeding import make_rng
from simulation.envs.competitive.actions import Action, ActionType

from simulation.agents.base import BaseAgent

_ACTION_TYPES = [ActionType.BUILD, ActionType.ATTACK, ActionType.DEFEND, ActionType.GAMBLE]
_AMOUNT_TYPES = {ActionType.BUILD, ActionType.ATTACK}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class CompetitiveRandomAgent(BaseAgent):
    """Uniformly random competitive policy, deterministic given its seed."""

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, agent_id: str, seed: int) -> None:
        self._rng = make_rng(seed)

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        assert self._rng is not None, "Must call reset() before act()"
        idx = int(self._rng.integers(len(_ACTION_TYPES)))
        atype = _ACTION_TYPES[idx]
        amount = float(self._rng.uniform(0.0, 1.0)) if atype in _AMOUNT_TYPES else 0.0
        return Action(type=atype, amount=amount)


class AlwaysAttackAgent(BaseAgent):
    """Deterministic policy that always attacks at half intensity."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        return Action(type=ActionType.ATTACK, amount=0.5)


class AlwaysBuildAgent(BaseAgent):
    """Deterministic policy that always builds at half intensity."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        return Action(type=ActionType.BUILD, amount=0.5)


class AlwaysDefendAgent(BaseAgent):
    """Deterministic policy that always defends."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: Any) -> Action:  # type: ignore[override]
        return Action(type=ActionType.DEFEND)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

COMPETITIVE_POLICY_REGISTRY: dict[str, type[BaseAgent]] = {
    "random": CompetitiveRandomAgent,
    "always_attack": AlwaysAttackAgent,
    "always_build": AlwaysBuildAgent,
    "always_defend": AlwaysDefendAgent,
    "competitive_ppo": None,  # type: ignore[assignment]  # lazy — resolved in factory
}


def create_competitive_agent(policy_name: str, **kwargs: Any) -> BaseAgent:
    """Instantiate a competitive baseline agent by policy name.

    Raises KeyError if *policy_name* is not in COMPETITIVE_POLICY_REGISTRY.
    """
    if policy_name not in COMPETITIVE_POLICY_REGISTRY:
        raise KeyError(policy_name)

    if policy_name == "competitive_ppo":
        from simulation.agents.competitive_ppo_agent import CompetitivePPOAgent
        return CompetitivePPOAgent(**kwargs)

    cls = COMPETITIVE_POLICY_REGISTRY[policy_name]
    return cls(**kwargs)
