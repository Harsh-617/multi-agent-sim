"""Abstract base environment — the environment contract.

Every environment (Mixed, or any future archetype) must implement this
interface.  The contract is defined in design/Defining_framework_of_environment.md
(Steps 2-7).

This class enforces:
  1. Lifecycle  — reset() produces initial observations
  2. Step       — step() accepts actions, returns per-agent StepResults
  3. Obs spec   — observation_spec() declared before training
  4. Action spec— action_spec() declared before training
  5. Termination— done signals + termination_reason
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from simulation.core.types import AgentID, StepResult, TerminationReason


class BaseEnvironment(ABC):
    """Abstract environment contract.  Domain-agnostic."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[AgentID, dict[str, Any]]:
        """Reset to initial state. Return initial observations keyed by agent ID."""
        ...

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @abstractmethod
    def step(
        self, actions: dict[AgentID, Any]
    ) -> dict[AgentID, StepResult]:
        """Advance one timestep given all agents' actions.

        Returns a StepResult for every *active* agent.
        """
        ...

    # ------------------------------------------------------------------
    # Specs (declared before training)
    # ------------------------------------------------------------------

    @abstractmethod
    def observation_spec(self) -> dict[str, Any]:
        """Describe the observation structure agents will receive."""
        ...

    @abstractmethod
    def action_spec(self) -> dict[str, Any]:
        """Describe the valid action structure agents must submit."""
        ...

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @abstractmethod
    def active_agents(self) -> list[AgentID]:
        """Return IDs of agents that are still active (not terminated)."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """True if the episode has terminated."""
        ...

    @abstractmethod
    def termination_reason(self) -> TerminationReason | None:
        """Why the episode ended, or None if still running."""
        ...

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Current timestep (0-indexed, incremented after each step)."""
        ...
