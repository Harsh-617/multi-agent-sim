"""Base agent interface for pluggable action policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from simulation.envs.mixed.actions import Action


class BaseAgent(ABC):
    """Interface that all agent policies must implement."""

    @abstractmethod
    def reset(self, agent_id: str, seed: int) -> None:
        """Initialise or re-initialise the agent for a new episode."""

    @abstractmethod
    def act(self, observation: dict) -> Action:
        """Choose an action given the current observation."""
