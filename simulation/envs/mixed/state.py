"""State representations for the Mixed archetype.

All mutable simulation state lives here — explicit, typed, no unstructured dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any

from simulation.core.types import AgentID
from simulation.envs.mixed.actions import Action


@dataclass(slots=True)
class AgentState:
    """Per-agent mutable state."""

    agent_id: AgentID
    resources: float
    active: bool = True
    # Rolling history of this agent's actions (most recent last)
    action_history: deque[Action] = field(default_factory=deque)
    # Rolling history of rewards received
    reward_history: deque[float] = field(default_factory=deque)

    def record(self, action: Action, reward: float, max_depth: int) -> None:
        """Append to history buffers, enforcing max depth."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        while len(self.action_history) > max_depth:
            self.action_history.popleft()
        while len(self.reward_history) > max_depth:
            self.reward_history.popleft()


@dataclass(slots=True)
class RelationalState:
    """Pairwise relational metrics between agents.

    cooperation_score tracks the rolling cooperation tendency between two
    agents.  Updated each step based on whether both cooperated.
    """

    cooperation_score: float = 0.0

    def update(self, both_cooperated: bool, sensitivity: float) -> None:
        """Exponential moving average update."""
        target = 1.0 if both_cooperated else 0.0
        self.cooperation_score += sensitivity * (target - self.cooperation_score)


@dataclass(slots=True)
class GlobalState:
    """Complete mutable state of a Mixed environment episode."""

    shared_pool: float
    step: int = 0
    agents: dict[AgentID, AgentState] = field(default_factory=dict)
    # Relational state keyed by sorted (agent_a, agent_b) tuple
    relations: dict[tuple[AgentID, AgentID], RelationalState] = field(
        default_factory=dict
    )

    @staticmethod
    def _pair_key(a: AgentID, b: AgentID) -> tuple[AgentID, AgentID]:
        return (a, b) if a < b else (b, a)

    def get_relation(self, a: AgentID, b: AgentID) -> RelationalState:
        key = self._pair_key(a, b)
        if key not in self.relations:
            self.relations[key] = RelationalState()
        return self.relations[key]

    def active_agent_ids(self) -> list[AgentID]:
        return [aid for aid, s in self.agents.items() if s.active]

    def to_observation(self, agent_id: AgentID, memory_steps: int) -> dict[str, Any]:
        """Build the observation dict for a single agent."""
        agent = self.agents[agent_id]
        active_ids = self.active_agent_ids()

        # Cooperation scores this agent has with every other active agent
        coop_scores = {
            other: self.get_relation(agent_id, other).cooperation_score
            for other in active_ids
            if other != agent_id
        }

        # Action history as serialisable list of dicts
        hist = [
            {"type": a.type.value, "amount": a.amount}
            for a in agent.action_history
        ][-memory_steps:]

        return {
            "step": self.step,
            "shared_pool": self.shared_pool,
            "own_resources": agent.resources,
            "num_active_agents": len(active_ids),
            "cooperation_scores": coop_scores,
            "action_history": hist,
        }
