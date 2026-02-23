"""Tit-for-tat agent — cooperates when cooperation ratio is high, else defends."""

from __future__ import annotations

from simulation.envs.mixed.actions import Action, ActionType

from simulation.agents.base import BaseAgent


class TitForTatAgent(BaseAgent):
    """Cooperates if the average cooperation score >= 0.5, otherwise defends.

    On the first step (no cooperation scores yet) defaults to cooperating.
    """

    def reset(self, agent_id: str, seed: int) -> None:
        pass

    def act(self, observation: dict) -> Action:
        coop_scores: dict = observation.get("cooperation_scores", {})
        if coop_scores:
            coop_ratio = sum(coop_scores.values()) / len(coop_scores)
        else:
            # No data yet — default to cooperation
            coop_ratio = 1.0

        if coop_ratio >= 0.5:
            return Action(type=ActionType.COOPERATE, amount=0.6)
        return Action(type=ActionType.DEFEND)
