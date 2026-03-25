"""Metrics collector for the Competitive archetype.

Ingests environment state, actions, and StepResults each step to produce:
  - structured step metric dicts
  - accumulated episode-level metrics
  - semantic event records

Respects InstrumentationConfig flags and step_log_frequency.
"""

from __future__ import annotations

from typing import Any

from simulation.config.competitive_schema import InstrumentationConfig
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.competitive.actions import Action, ActionType
from simulation.metrics.competitive_definitions import EventType


class CompetitiveMetricsCollector:
    """Collects and structures metrics for a single competitive episode."""

    def __init__(self, config: InstrumentationConfig) -> None:
        self._config = config
        self._total_rewards: dict[AgentID, float] = {}
        self._events: list[dict[str, Any]] = []
        self._prev_active: set[AgentID] = set()
        self._prev_rankings: dict[AgentID, int] = {}

    # ------------------------------------------------------------------
    # Step metrics
    # ------------------------------------------------------------------

    def collect_step(
        self,
        step: int,
        actions: dict[AgentID, Action],
        results: dict[AgentID, StepResult],
        agent_scores: dict[AgentID, float],
        agent_resources: dict[AgentID, float],
        active_agents: list[AgentID],
        rankings: list[tuple[AgentID, float]],
    ) -> list[dict[str, Any]]:
        """Build per-agent step metric records and accumulate totals.

        Returns a list of step metric dicts (one per agent), or an empty
        list if step metrics are disabled or this step is skipped by
        step_log_frequency.
        """
        # Accumulate total rewards regardless of logging flags
        for aid, sr in results.items():
            self._total_rewards[aid] = self._total_rewards.get(aid, 0.0) + sr.reward

        # Current rankings as agent_id -> 1-based rank
        current_rank: dict[AgentID, int] = {
            aid: rank for rank, (aid, _score) in enumerate(rankings, start=1)
        }

        # Detect agent eliminations
        current_active = set(active_agents)
        if self._config.enable_event_log:
            for aid in self._prev_active - current_active:
                rank = current_rank.get(aid, len(rankings))
                self._events.append({
                    "event": EventType.AGENT_ELIMINATED.value,
                    "step": step,
                    "agent_id": aid,
                    "final_score": agent_scores.get(aid, 0.0),
                    "final_rank": rank,
                })

        # Detect rank changes
        if self._config.enable_event_log and self._prev_rankings:
            for aid, new_rank in current_rank.items():
                old_rank = self._prev_rankings.get(aid)
                if old_rank is not None and old_rank != new_rank:
                    self._events.append({
                        "event": EventType.RANK_CHANGE.value,
                        "step": step,
                        "agent_id": aid,
                        "old_rank": old_rank,
                        "new_rank": new_rank,
                    })

        # Update previous state
        self._prev_active = current_active
        self._prev_rankings = current_rank

        # Check if we should emit step metrics
        if not self._config.enable_step_metrics:
            return []
        if step % self._config.step_log_frequency != 0:
            return []

        # Compute action ratios across all agents this step
        n_agents = len(actions)
        n_attack = sum(1 for a in actions.values() if a.type == ActionType.ATTACK)
        n_defend = sum(1 for a in actions.values() if a.type == ActionType.DEFEND)
        n_build = sum(1 for a in actions.values() if a.type == ActionType.BUILD)
        n_gamble = sum(1 for a in actions.values() if a.type == ActionType.GAMBLE)
        attack_ratio = n_attack / n_agents if n_agents > 0 else 0.0
        defend_ratio = n_defend / n_agents if n_agents > 0 else 0.0
        build_ratio = n_build / n_agents if n_agents > 0 else 0.0
        gamble_ratio = n_gamble / n_agents if n_agents > 0 else 0.0

        records: list[dict[str, Any]] = []
        for aid, sr in results.items():
            action = actions.get(aid, Action(type=ActionType.DEFEND))
            records.append({
                "step": step,
                "agent_id": aid,
                "reward": sr.reward,
                "action_type": action.type.value,
                "action_amount": action.amount,
                "own_score": agent_scores.get(aid, 0.0),
                "own_resources": agent_resources.get(aid, 0.0),
                "own_rank": current_rank.get(aid, len(rankings)),
                "num_active_agents": len(current_active),
                "attack_ratio": attack_ratio,
                "defend_ratio": defend_ratio,
                "build_ratio": build_ratio,
                "gamble_ratio": gamble_ratio,
            })

        return records

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_attack_succeeded(
        self, step: int, attacker_id: AgentID, score_gained: float
    ) -> None:
        """Record a successful attack event."""
        if not self._config.enable_event_log:
            return
        self._events.append({
            "event": EventType.ATTACK_SUCCEEDED.value,
            "step": step,
            "attacker_id": attacker_id,
            "score_gained": score_gained,
        })

    def record_attack_defended(
        self, step: int, attacker_id: AgentID, defender_id: AgentID, cost_paid: float
    ) -> None:
        """Record an attack that was fully blocked by a defender."""
        if not self._config.enable_event_log:
            return
        self._events.append({
            "event": EventType.ATTACK_DEFENDED.value,
            "step": step,
            "attacker_id": attacker_id,
            "defender_id": defender_id,
            "cost_paid": cost_paid,
        })

    def record_gamble_resolved(
        self,
        step: int,
        agent_id: AgentID,
        outcome: float,
        resources_before: float,
        resources_after: float,
    ) -> None:
        """Record a resolved gamble action."""
        if not self._config.enable_event_log:
            return
        self._events.append({
            "event": EventType.GAMBLE_RESOLVED.value,
            "step": step,
            "agent_id": agent_id,
            "outcome": outcome,
            "resources_before": resources_before,
            "resources_after": resources_after,
        })

    @property
    def events(self) -> list[dict[str, Any]]:
        """All semantic events collected so far."""
        return list(self._events)

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(
        self,
        episode_length: int,
        termination_reason: TerminationReason | None,
        final_scores: dict[AgentID, float],
        final_rankings: list[tuple[AgentID, float]],
    ) -> dict[str, Any]:
        """Build episode-level summary metrics.

        Returns an empty dict if episode metrics are disabled.
        """
        if not self._config.enable_episode_metrics:
            return {}

        scores = list(final_scores.values())
        score_spread = (max(scores) - min(scores)) if scores else 0.0

        # Winner: agent with highest score, None on draw
        winner_id: str | None = None
        if final_rankings:
            top_score = final_rankings[0][1]
            # Check for draw: more than one agent with top score
            top_agents = [aid for aid, sc in final_rankings if sc == top_score]
            if len(top_agents) == 1:
                winner_id = top_agents[0]

        # Count eliminations: agents that were eliminated before episode end
        num_eliminations = sum(
            1 for e in self._events
            if e["event"] == EventType.AGENT_ELIMINATED.value
        )

        return {
            "episode_length": episode_length,
            "termination_reason": termination_reason.value if termination_reason else None,
            "final_rankings": [aid for aid, _sc in final_rankings],
            "final_scores": dict(final_scores),
            "score_spread": score_spread,
            "winner_id": winner_id,
            "num_eliminations": num_eliminations,
            "total_reward_per_agent": dict(self._total_rewards),
        }
