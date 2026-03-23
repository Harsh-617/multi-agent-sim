"""State representations for the Competitive archetype.

All mutable simulation state lives here — explicit, typed, no unstructured dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any

from simulation.core.types import AgentID
from simulation.envs.competitive.actions import Action, ActionType


@dataclass(slots=True)
class AgentState:
    """Per-agent mutable state."""

    agent_id: AgentID
    score: float
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
class OpponentHistoryState:
    """Pairwise opponent action history between agents.

    Tracks a rolling window of action types observed for each
    opponent pair.  No trust or cooperation scores — only action
    history for opponent modelling.
    """

    action_history: deque[ActionType] = field(default_factory=deque)

    def record(self, action_type: ActionType, max_depth: int) -> None:
        """Append an observed action, enforcing max window depth."""
        self.action_history.append(action_type)
        while len(self.action_history) > max_depth:
            self.action_history.popleft()


@dataclass(slots=True)
class GlobalState:
    """Complete mutable state of a Competitive environment episode."""

    step: int = 0
    agents: dict[AgentID, AgentState] = field(default_factory=dict)
    # Opponent history keyed by sorted (agent_a, agent_b) tuple
    opponent_histories: dict[tuple[AgentID, AgentID], OpponentHistoryState] = field(
        default_factory=dict
    )

    @staticmethod
    def _pair_key(a: AgentID, b: AgentID) -> tuple[AgentID, AgentID]:
        return (a, b) if a < b else (b, a)

    def get_opponent_history(
        self, a: AgentID, b: AgentID
    ) -> OpponentHistoryState:
        key = self._pair_key(a, b)
        if key not in self.opponent_histories:
            self.opponent_histories[key] = OpponentHistoryState()
        return self.opponent_histories[key]

    def active_agent_ids(self) -> list[AgentID]:
        return [aid for aid, s in self.agents.items() if s.active]

    def rankings(self) -> list[tuple[AgentID, float]]:
        """Return agents sorted by score descending (highest first)."""
        return sorted(
            ((aid, s.score) for aid, s in self.agents.items()),
            key=lambda x: x[1],
            reverse=True,
        )

    def to_observation(
        self,
        agent_id: AgentID,
        opponent_obs_window: int,
        observation_memory_steps: int,
    ) -> dict[str, Any]:
        """Build the observation dict for a single agent."""
        agent = self.agents[agent_id]
        active_ids = self.active_agent_ids()
        ranked = self.rankings()

        # Own action history as serialisable list of dicts
        own_hist = [
            {"type": a.type.value, "amount": a.amount}
            for a in agent.action_history
        ][-observation_memory_steps:]

        # Opponent observations: scores and recent action types (windowed)
        opponent_obs: dict[str, Any] = {}
        for other in active_ids:
            if other == agent_id:
                continue
            other_state = self.agents[other]
            # Recent actions from opponent history (this pair)
            pair_history = self.get_opponent_history(agent_id, other)
            recent_actions = [
                at.value for at in pair_history.action_history
            ][-opponent_obs_window:]
            opponent_obs[other] = {
                "score": other_state.score,
                "recent_actions": recent_actions,
            }

        return {
            "step": self.step,
            "own_score": agent.score,
            "own_resources": agent.resources,
            "num_active_agents": len(active_ids),
            "rankings": [(aid, sc) for aid, sc in ranked],
            "action_history": own_hist,
            "opponents": opponent_obs,
        }
