"""Metric names and schemas for the Cooperative archetype.

Three categories:
  - Step metrics    : per-agent, per-step measurements
  - Episode metrics : summary of an entire episode
  - Event types     : semantic events (specialization, free-riding, collapse)
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Step metric keys (one record per agent per step)
# ---------------------------------------------------------------------------

STEP_METRIC_KEYS: list[str] = [
    "step",
    "agent_id",
    "reward",
    "task_type",        # int | None (None = IDLE)
    "effort_amount",
    "effective_contribution",
    "r_group",
    "r_individual",
    "r_efficiency",
    "system_stress",
    "backlog_level",
    "completion_rate",
]

STEP_METRIC_SCHEMA: dict[str, str] = {
    "step": "int",
    "agent_id": "str",
    "reward": "float",
    "task_type": "int | None",
    "effort_amount": "float",
    "effective_contribution": "float",
    "r_group": "float",
    "r_individual": "float",
    "r_efficiency": "float",
    "system_stress": "float",
    "backlog_level": "int",
    "completion_rate": "float",
}


# ---------------------------------------------------------------------------
# Episode metric keys (one record per episode)
# ---------------------------------------------------------------------------

EPISODE_METRIC_KEYS: list[str] = [
    "episode_length",
    "termination_reason",
    "total_tasks_arrived",
    "total_tasks_completed",
    "completion_ratio",
    "final_backlog_level",
    "final_system_stress",
    "mean_system_stress",
    "collapse_occurred",
    "total_reward_per_agent",
    "mean_reward_per_step_per_agent",
]

EPISODE_METRIC_SCHEMA: dict[str, str] = {
    "episode_length": "int",
    "termination_reason": "str",
    "total_tasks_arrived": "int",
    "total_tasks_completed": "int",
    "completion_ratio": "float",
    "final_backlog_level": "int",
    "final_system_stress": "float",
    "mean_system_stress": "float",
    "collapse_occurred": "bool",
    "total_reward_per_agent": "dict[str, float]",
    "mean_reward_per_step_per_agent": "dict[str, float]",
}


# ---------------------------------------------------------------------------
# Semantic event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Semantic events emitted during a cooperative simulation run."""

    SPECIALIZATION_THRESHOLD_CROSSED = "specialization_threshold_crossed"
    FREE_RIDER_DETECTED = "free_rider_detected"
    SYSTEM_STRESS_SPIKE = "system_stress_spike"
    STRESS_RECOVERY = "stress_recovery"
    SYSTEM_COLLAPSE = "system_collapse"
    PERFECT_CLEARANCE = "perfect_clearance"
    CONTRIBUTION_IMBALANCE = "contribution_imbalance"


EVENT_SCHEMAS: dict[str, dict[str, str]] = {
    EventType.SPECIALIZATION_THRESHOLD_CROSSED.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "task_type": "int",
        "score": "float",
    },
    EventType.FREE_RIDER_DETECTED.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "idle_rate": "float",
    },
    EventType.SYSTEM_STRESS_SPIKE.value: {
        "event": "str",
        "step": "int",
        "stress_before": "float",
        "stress_after": "float",
    },
    EventType.STRESS_RECOVERY.value: {
        "event": "str",
        "step": "int",
        "stress_before": "float",
        "stress_after": "float",
    },
    EventType.SYSTEM_COLLAPSE.value: {
        "event": "str",
        "step": "int",
        "backlog_level": "int",
    },
    EventType.PERFECT_CLEARANCE.value: {
        "event": "str",
        "step": "int",
    },
    EventType.CONTRIBUTION_IMBALANCE.value: {
        "event": "str",
        "step": "int",
        "gini_coefficient": "float",
    },
}
