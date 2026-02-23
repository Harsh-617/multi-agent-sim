"""Always-cooperate agent — always plays COOPERATE with amount=0.6."""

from __future__ import annotations

from simulation.envs.mixed.actions import Action, ActionType

from simulation.agents.base import BaseAgent


class AlwaysCooperateAgent(BaseAgent):
    """Deterministic policy that always cooperates."""

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: dict) -> Action:
        return Action(type=ActionType.COOPERATE, amount=0.6)
