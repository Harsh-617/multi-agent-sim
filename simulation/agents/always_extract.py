"""Always-extract agent — always plays EXTRACT with amount=0.6."""

from __future__ import annotations

from simulation.envs.mixed.actions import Action, ActionType

from simulation.agents.base import BaseAgent


class AlwaysExtractAgent(BaseAgent):
    """Deterministic policy that always extracts."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: dict) -> Action:
        return Action(type=ActionType.EXTRACT, amount=0.6)
