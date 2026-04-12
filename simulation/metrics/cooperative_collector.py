"""Metrics collector for the Cooperative archetype.

Ingests environment state and step results each step to produce:
  - structured step metric dicts
  - accumulated episode-level metrics
  - semantic event records

Step E additions:
  - Per-agent extended metrics: idle_rate, effort_utilization, dominant_task_type,
    role_stability, final/peak specialization scores
  - Social/group metrics: Gini coefficient, free-rider stats, specialization divergence
  - peak_system_stress, group_efficiency_ratio
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative floats. Bounded [0, 1]."""
    n = len(values)
    if n <= 1:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += abs(values[i] - values[j])
    return s / (2.0 * n * total)


def _variance(values: list[float]) -> float:
    """Population variance."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


class CooperativeMetricsCollector:
    """Collects and structures metrics for a single cooperative episode."""

    def __init__(
        self,
        config: InstrumentationConfig,
        *,
        agent_effort_capacity: float = 1.0,
        mean_task_difficulty: float = 1.0,
    ) -> None:
        self._config = config
        self._agent_effort_capacity = max(agent_effort_capacity, 1e-9)
        self._mean_task_difficulty = max(mean_task_difficulty, 1e-9)

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

        # ---- Step E: additional per-agent tracking ----
        self._idle_count: dict[AgentID, int] = {}
        self._work_count: dict[AgentID, int] = {}
        self._work_effort: dict[AgentID, float] = {}   # total effort when working
        self._effort_by_type: dict[AgentID, list[float]] = {}  # cumulative per task type
        self._steps_on_type: dict[AgentID, dict[int, int]] = {}  # steps per task type
        self._peak_spec: dict[AgentID, float] = {}     # max spec score any type
        self._peak_stress: float = 0.0

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
        self._peak_stress = max(self._peak_stress, state.system_stress)

        # ---- Per-agent tracking (Step E) ----
        num_types = state.num_task_types()
        for aid, sr in results.items():
            act = actions.get(aid, Action(task_type=None))
            if act.is_idle:
                self._idle_count[aid] = self._idle_count.get(aid, 0) + 1
            else:
                self._work_count[aid] = self._work_count.get(aid, 0) + 1
                self._work_effort[aid] = self._work_effort.get(aid, 0.0) + act.effort_amount
                # Effort by type
                if aid not in self._effort_by_type:
                    self._effort_by_type[aid] = [0.0] * num_types
                tt = act.task_type
                if tt is not None and 0 <= tt < len(self._effort_by_type[aid]):
                    self._effort_by_type[aid][tt] += act.effort_amount
                # Steps on type
                if aid not in self._steps_on_type:
                    self._steps_on_type[aid] = {}
                if tt is not None:
                    self._steps_on_type[aid][tt] = self._steps_on_type[aid].get(tt, 0) + 1

            # Peak specialization
            if aid in state.agents and state.agents[aid].specialization_score:
                mx = max(state.agents[aid].specialization_score)
                self._peak_spec[aid] = max(self._peak_spec.get(aid, 0.0), mx)

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
        """Build episode-level summary metrics (basic + Step E extended)."""
        if not self._config.enable_episode_metrics:
            return {}

        total_arrived = sum(state.tasks_completed_total)  # use completed as proxy
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

        summary: dict[str, Any] = {
            "episode_length": episode_length,
            "termination_reason": termination_reason.value if termination_reason else None,
            "total_tasks_arrived": total_arrived,
            "total_tasks_completed": total_completed,
            "completion_ratio": float(max(0.0, min(1.0, completion_ratio))),
            "final_backlog_level": state.backlog_level,
            "final_system_stress": state.system_stress,
            "mean_system_stress": mean_stress,
            "peak_system_stress": self._peak_stress,
            "collapse_occurred": collapse_occurred,
            "total_reward_per_agent": dict(self._total_rewards),
            "mean_reward_per_step_per_agent": mean_reward_per_step,
        }

        # ---- Step E: extended metrics ----
        num_agents = state.num_agents()
        num_types = state.num_task_types()

        # --- Per-agent metrics ---
        agent_metrics: dict[str, dict[str, Any]] = {}
        for aid in state.agents.keys():
            idle_c = self._idle_count.get(aid, 0)
            work_c = self._work_count.get(aid, 0)
            total_c = idle_c + work_c

            idle_rate = idle_c / total_c if total_c > 0 else 0.0

            # Effort utilization (mean effort / capacity when working)
            effort_util = (
                self._work_effort.get(aid, 0.0)
                / (work_c * self._agent_effort_capacity)
                if work_c > 0
                else 0.0
            )
            effort_util = float(max(0.0, min(1.0, effort_util)))

            # Dominant task type & fraction
            eby_type = self._effort_by_type.get(aid, [])
            total_effort = sum(eby_type)
            if eby_type and total_effort > 0:
                dominant_type = int(
                    max(range(len(eby_type)), key=lambda i: eby_type[i])
                )
                dominant_type_fraction = eby_type[dominant_type] / total_effort
            else:
                dominant_type = None
                dominant_type_fraction = 0.0

            # Role stability (fraction of steps on dominant type)
            steps_on = self._steps_on_type.get(aid, {})
            if dominant_type is not None and total_c > 0:
                role_stability = float(
                    steps_on.get(dominant_type, 0) / total_c
                )
            else:
                role_stability = 0.0
            role_stability = max(0.0, min(1.0, role_stability))

            # Specialization scores at episode end
            spec_scores = state.agents[aid].specialization_score if aid in state.agents else []
            if dominant_type is not None and dominant_type < len(spec_scores):
                final_spec = float(spec_scores[dominant_type])
            else:
                final_spec = 0.0

            peak_spec = float(self._peak_spec.get(aid, 0.0))

            agent_metrics[aid] = {
                "cumulative_reward": float(self._total_rewards.get(aid, 0.0)),
                "mean_reward_per_step": float(
                    self._total_rewards.get(aid, 0.0) / max(episode_length, 1)
                ),
                "effort_utilization": effort_util,
                "idle_rate": float(max(0.0, min(1.0, idle_rate))),
                "dominant_task_type": dominant_type,
                "dominant_type_fraction": float(
                    max(0.0, min(1.0, dominant_type_fraction))
                ),
                "final_specialization_score": float(
                    max(0.0, min(1.0, final_spec))
                ),
                "peak_specialization_score": float(max(0.0, min(1.0, peak_spec))),
                "role_stability": role_stability,
                "strategy_label": "\u2014",  # placeholder — computed by clustering
            }

        # --- Social / group metrics ---
        total_effort_per_agent = [
            self._work_effort.get(aid, 0.0) for aid in state.agents.keys()
        ]

        effort_gini = _gini(total_effort_per_agent)
        contribution_variance = _variance(total_effort_per_agent)

        free_rider_count = sum(
            1 for aid in state.agents.keys()
            if agent_metrics[aid]["idle_rate"] >= _FREE_RIDER_IDLE_THRESHOLD
        )
        free_rider_fraction = (
            free_rider_count / num_agents if num_agents > 0 else 0.0
        )

        dominant_types = [
            agent_metrics[aid]["dominant_task_type"]
            for aid in state.agents.keys()
            if agent_metrics[aid]["dominant_task_type"] is not None
        ]
        if len(dominant_types) >= 2:
            pairs_total = len(dominant_types) * (len(dominant_types) - 1) / 2
            diff_pairs = sum(
                1
                for i in range(len(dominant_types))
                for j in range(i + 1, len(dominant_types))
                if dominant_types[i] != dominant_types[j]
            )
            specialization_divergence = (
                diff_pairs / pairs_total if pairs_total > 0 else 0.0
            )
        else:
            specialization_divergence = 0.0

        mean_role_stability = (
            sum(agent_metrics[aid]["role_stability"] for aid in state.agents.keys())
            / num_agents
            if num_agents > 0
            else 0.0
        )

        # Group efficiency ratio
        if (
            num_agents > 0
            and episode_length > 0
            and self._agent_effort_capacity > 0
            and self._mean_task_difficulty > 0
        ):
            theoretical_max = (
                num_agents
                * self._agent_effort_capacity
                * episode_length
                / self._mean_task_difficulty
            )
            group_efficiency_ratio = float(
                min(1.0, total_completed / theoretical_max)
                if theoretical_max > 0
                else 0.0
            )
        else:
            group_efficiency_ratio = 0.0

        summary.update({
            "group_efficiency_ratio": max(0.0, min(1.0, group_efficiency_ratio)),
            "contribution_variance": float(contribution_variance),
            "specialization_divergence": float(
                max(0.0, min(1.0, specialization_divergence))
            ),
            "mean_role_stability": float(
                max(0.0, min(1.0, mean_role_stability))
            ),
            "free_rider_count": free_rider_count,
            "free_rider_fraction": float(
                max(0.0, min(1.0, free_rider_fraction))
            ),
            "effort_gini_coefficient": float(
                max(0.0, min(1.0, effort_gini))
            ),
            "agent_metrics": agent_metrics,
        })

        return summary
