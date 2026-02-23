"""Agent policies package — registry and factory for pluggable policies."""

from __future__ import annotations

from simulation.agents.base import BaseAgent
from simulation.agents.random_agent import RandomAgent
from simulation.agents.always_cooperate import AlwaysCooperateAgent
from simulation.agents.always_extract import AlwaysExtractAgent
from simulation.agents.tit_for_tat import TitForTatAgent

POLICY_REGISTRY: dict[str, type[BaseAgent]] = {
    "random": RandomAgent,
    "always_cooperate": AlwaysCooperateAgent,
    "always_extract": AlwaysExtractAgent,
    "tit_for_tat": TitForTatAgent,
}

ALLOWED_POLICIES = frozenset((*POLICY_REGISTRY.keys(), "ppo_shared", "league_snapshot"))


def create_agent(policy: str, **kwargs) -> BaseAgent:
    """Instantiate an agent by policy name.

    ``ppo_shared`` and ``league_snapshot`` are resolved lazily to avoid
    importing torch at module level, so the server can start even without
    torch installed.

    Raises KeyError if the policy name is not registered.
    """
    if policy == "ppo_shared":
        from simulation.agents.ppo_shared_agent import PPOSharedAgent
        return PPOSharedAgent(**kwargs)
    if policy == "league_snapshot":
        from simulation.agents.league_snapshot_agent import LeaguePolicyAgent
        return LeaguePolicyAgent(**kwargs)
    cls = POLICY_REGISTRY[policy]
    return cls()


__all__ = [
    "BaseAgent",
    "ALLOWED_POLICIES",
    "POLICY_REGISTRY",
    "create_agent",
    "RandomAgent",
    "AlwaysCooperateAgent",
    "AlwaysExtractAgent",
    "TitForTatAgent",
]
