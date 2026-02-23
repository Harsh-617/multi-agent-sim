"""Metrics collector for the Mixed archetype.

Ingests environment state, actions, and StepResults each step to produce:
  - structured step metric dicts
  - accumulated episode-level metrics
  - semantic event records

Respects InstrumentationConfig flags and step_log_frequency.
"""

from __future__ import annotations

from typing import Any

from simulation.config.schema import InstrumentationConfig
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.mixed.actions import Action, ActionType
from simulation.metrics.definitions import EventType


class MetricsCollector:
    """Collects and structures metrics for a single episode."""

    def __init__(self, config: InstrumentationConfig) -> None:
        self._config = config
        self._total_rewards: dict[AgentID, float] = {}
        self._events: list[dict[str, Any]] = []
        self._prev_active: set[AgentID] = set()

    # ------------------------------------------------------------------
    # Step metrics
    # ------------------------------------------------------------------

    def collect_step(
        self,
        step: int,
        actions: dict[AgentID, Action],
        results: dict[AgentID, StepResult],
        shared_pool: float,
        agent_resources: dict[AgentID, float],
        active_agents: list[AgentID],
    ) -> list[dict[str, Any]]:
        """Build per-agent step metric records and accumulate totals.

        Returns a list of step metric dicts (one per agent), or an empty
        list if step metrics are disabled or this step is skipped by
        step_log_frequency.
        """
        # Accumulate total rewards regardless of logging flags
        for aid, sr in results.items():
            self._total_rewards[aid] = self._total_rewards.get(aid, 0.0) + sr.reward

        # Detect agent deactivations
        current_active = set(active_agents)
        if self._config.enable_event_log:
            for aid in self._prev_active - current_active:
                self._events.append({
                    "event": EventType.AGENT_DEACTIVATED.value,
                    "step": step,
                    "agent_id": aid,
                    "resources": agent_resources.get(aid, 0.0),
                })
        self._prev_active = current_active

        # Check if we should emit step metrics
        if not self._config.enable_step_metrics:
            return []
        if step % self._config.step_log_frequency != 0:
            return []

        # Cooperation / extraction ratios among agents in this step
        n_agents = len(results)
        n_coop = sum(
            1 for a in actions.values() if a.type == ActionType.COOPERATE
        )
        n_extract = sum(
            1 for a in actions.values() if a.type == ActionType.EXTRACT
        )
        coop_ratio = n_coop / n_agents if n_agents > 0 else 0.0
        extraction_ratio = n_extract / n_agents if n_agents > 0 else 0.0

        records: list[dict[str, Any]] = []
        for aid, sr in results.items():
            action = actions.get(aid, Action(type=ActionType.DEFEND))
            records.append({
                "step": step,
                "agent_id": aid,
                "reward": sr.reward,
                "action_type": action.type.value,
                "action_amount": action.amount,
                "shared_pool": shared_pool,
                "agent_resources": agent_resources.get(aid, 0.0),
                "coop_ratio": coop_ratio,
                "extraction_ratio": extraction_ratio,
            })

        return records

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_collapse(self, step: int, shared_pool: float) -> None:
        """Record a system collapse event."""
        if not self._config.enable_event_log:
            return
        self._events.append({
            "event": EventType.COLLAPSE.value,
            "step": step,
            "shared_pool": shared_pool,
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
        final_shared_pool: float,
    ) -> dict[str, Any]:
        """Build episode-level summary metrics.

        Returns an empty dict if episode metrics are disabled.
        """
        if not self._config.enable_episode_metrics:
            return {}
        return {
            "episode_length": episode_length,
            "termination_reason": termination_reason.value if termination_reason else None,
            "final_shared_pool": final_shared_pool,
            "total_reward_per_agent": dict(self._total_rewards),
        }
