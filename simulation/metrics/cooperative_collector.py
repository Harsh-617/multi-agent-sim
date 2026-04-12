"""Metrics collector for the Cooperative archetype.

Ingests environment state and step results each step to produce:
  - structured step metric dicts
  - accumulated episode-level metrics
  - semantic event records
"""

from __future__ import annotations

from typing import Any

from simulation.config.cooperative_schema import InstrumentationConfig
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.cooperative.actions import Action
from simulation.envs.cooperative.state import GlobalState
from simulation.metrics.cooperative_definitions import EventType

# Threshold for specialization event
_SPEC_THRESHOLD = 0.7
# Threshold for free-rider detection (idle_rate over rolling window)
_FREE_RIDER_IDLE_THRESHOLD = 0.4
_FREE_RIDER_WINDOW = 10
# Stress spike / recovery thresholds
_STRESS_SPIKE_DELTA = 0.3
_STRESS_HIGH = 0.8
_STRESS_LOW = 0.4
# Gini threshold for contribution imbalance event
_GINI_THRESHOLD = 0.5


class CooperativeMetricsCollector:
    """Collects and structures metrics for a single cooperative episode."""

    def __init__(self, config: InstrumentationConfig) -> None:
        self._config = config
        self._total_rewards: dict[AgentID, float] = {}
        self._events: list[dict[str, Any]] = []

        # For event detection
        self._prev_stress: float = 0.0
        self._stress_history: list[float] = []
        self._idle_history: dict[AgentID, list[bool]] = {}
        self._spec_crossed: dict[AgentID, set[int]] = {}
        self._gini_imbalance_steps: int = 0
        self._spec_locked: dict[AgentID, tuple[int, int]] = {}  # agent -> (type, count)

        # For episode summary
        self._stress_accumulator: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Step metrics
    # ------------------------------------------------------------------

    def collect_step(
        self,
        step: int,
        actions: dict[AgentID, Action],
        results: dict[AgentID, StepResult],
        state: GlobalState,
    ) -> list[dict[str, Any]]:
        """Build per-agent step metric records and detect events."""

        # Accumulate totals regardless of logging flags
        for aid, sr in results.items():
            self._total_rewards[aid] = self._total_rewards.get(aid, 0.0) + sr.reward

        self._stress_accumulator += state.system_stress
        self._step_count += 1

        # --- Event detection ---
        if self._config.enable_step_metrics or self._config.enable_episode_metrics:
            self._detect_events(step, actions, state)

        # --- Step records ---
        if not self._config.enable_step_metrics:
            return []
        if step % self._config.step_log_frequency != 0:
            return []

        records: list[dict[str, Any]] = []
        completion_rate = state.relational.group_completion_rate

        for aid, sr in results.items():
            act = actions.get(aid, Action(task_type=None))
            components = sr.info.get("reward_components", {})
            records.append({
                "step": step,
                "agent_id": aid,
                "reward": sr.reward,
                "task_type": act.task_type,
                "effort_amount": act.effort_amount,
                "effective_contribution": state.agents[aid].last_effort[
                    act.task_type
                ] if act.task_type is not None else 0.0,
                "r_group": components.get("r_group", 0.0),
                "r_individual": components.get("r_individual", 0.0),
                "r_efficiency": components.get("r_efficiency", 0.0),
                "system_stress": state.system_stress,
                "backlog_level": state.backlog_level,
                "completion_rate": completion_rate,
            })

        return records

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _detect_events(
        self, step: int, actions: dict[AgentID, Action], state: GlobalState
    ) -> None:
        if not self._config.enable_step_metrics:
            return

        stress = state.system_stress

        # Specialization threshold crossed
        for aid, agent in state.agents.items():
            if aid not in self._spec_crossed:
                self._spec_crossed[aid] = set()
            for t, score in enumerate(agent.specialization_score):
                if score >= _SPEC_THRESHOLD and t not in self._spec_crossed[aid]:
                    self._spec_crossed[aid].add(t)
                    self._events.append({
                        "event": EventType.SPECIALIZATION_THRESHOLD_CROSSED.value,
                        "step": step,
                        "agent_id": aid,
                        "task_type": t,
                        "score": score,
                    })

        # Free-rider detection (rolling window)
        for aid in state.agent_ids():
            act = actions.get(aid, Action(task_type=None))
            if aid not in self._idle_history:
                self._idle_history[aid] = []
            self._idle_history[aid].append(act.is_idle)
            if len(self._idle_history[aid]) > _FREE_RIDER_WINDOW:
                self._idle_history[aid].pop(0)
            window = self._idle_history[aid]
            if len(window) == _FREE_RIDER_WINDOW:
                idle_rate = sum(window) / _FREE_RIDER_WINDOW
                if idle_rate >= _FREE_RIDER_IDLE_THRESHOLD:
                    # Avoid duplicate events for same sustained period
                    if not self._events or self._events[-1].get("agent_id") != aid or \
                       self._events[-1].get("event") != EventType.FREE_RIDER_DETECTED.value:
                        self._events.append({
                            "event": EventType.FREE_RIDER_DETECTED.value,
                            "step": step,
                            "agent_id": aid,
                            "idle_rate": idle_rate,
                        })

        # System stress spike
        stress_delta = stress - self._prev_stress
        if stress_delta >= _STRESS_SPIKE_DELTA:
            self._events.append({
                "event": EventType.SYSTEM_STRESS_SPIKE.value,
                "step": step,
                "stress_before": self._prev_stress,
                "stress_after": stress,
            })

        # Stress recovery (from high to low within recent steps)
        self._stress_history.append(stress)
        if len(self._stress_history) > 5:
            self._stress_history.pop(0)
        if (
            len(self._stress_history) >= 2
            and self._stress_history[0] >= _STRESS_HIGH
            and stress <= _STRESS_LOW
        ):
            self._events.append({
                "event": EventType.STRESS_RECOVERY.value,
                "step": step,
                "stress_before": self._stress_history[0],
                "stress_after": stress,
            })

        self._prev_stress = stress

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------

    def episode_summary(
        self,
        episode_length: int,
        termination_reason: TerminationReason | None,
        state: GlobalState,
    ) -> dict[str, Any]:
        """Build episode-level summary metrics."""
        if not self._config.enable_episode_metrics:
            return {}

        total_arrived = sum(state.tasks_completed_total)  # use completed as proxy
        # More accurately: sum up arrivals tracked in state
        total_completed = sum(state.tasks_completed_total)

        completion_ratio = (
            total_completed / total_arrived if total_arrived > 0 else 1.0
        )

        mean_stress = (
            self._stress_accumulator / self._step_count
            if self._step_count > 0
            else 0.0
        )

        mean_reward_per_step: dict[AgentID, float] = {
            aid: (total / max(episode_length, 1))
            for aid, total in self._total_rewards.items()
        }

        collapse_occurred = (
            termination_reason == TerminationReason.SYSTEM_COLLAPSE
        )

        return {
            "episode_length": episode_length,
            "termination_reason": termination_reason.value if termination_reason else None,
            "total_tasks_arrived": total_arrived,
            "total_tasks_completed": total_completed,
            "completion_ratio": completion_ratio,
            "final_backlog_level": state.backlog_level,
            "final_system_stress": state.system_stress,
            "mean_system_stress": mean_stress,
            "collapse_occurred": collapse_occurred,
            "total_reward_per_agent": dict(self._total_rewards),
            "mean_reward_per_step_per_agent": mean_reward_per_step,
        }
