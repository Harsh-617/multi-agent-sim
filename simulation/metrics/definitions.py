"""Metric names and minimal schemas for the Mixed archetype.

Defines three categories:
  - Step metrics: per-agent, per-step measurements
  - Episode metrics: summary of an entire episode
  - Event types: semantic events (collapse, agent deactivation)

All schemas are plain dicts describing expected keys and types,
used for documentation and optional runtime validation.
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
    "action_type",
    "action_amount",
    "shared_pool",
    "agent_resources",
    "coop_ratio",
    "extraction_ratio",
]

STEP_METRIC_SCHEMA: dict[str, str] = {
    "step": "int",
    "agent_id": "str",
    "reward": "float",
    "action_type": "str",
    "action_amount": "float",
    "shared_pool": "float",
    "agent_resources": "float",
    "coop_ratio": "float",
    "extraction_ratio": "float",
}


# ---------------------------------------------------------------------------
# Episode metric keys (one record per episode)
# ---------------------------------------------------------------------------

EPISODE_METRIC_KEYS: list[str] = [
    "episode_length",
    "termination_reason",
    "final_shared_pool",
    "total_reward_per_agent",
]

EPISODE_METRIC_SCHEMA: dict[str, str] = {
    "episode_length": "int",
    "termination_reason": "str",
    "final_shared_pool": "float",
    "total_reward_per_agent": "dict[str, float]",
}


# ---------------------------------------------------------------------------
# Semantic event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Semantic events emitted during a simulation run."""

    COLLAPSE = "collapse"
    AGENT_DEACTIVATED = "agent_deactivated"


EVENT_SCHEMAS: dict[str, dict[str, str]] = {
    EventType.COLLAPSE.value: {
        "event": "str",
        "step": "int",
        "shared_pool": "float",
    },
    EventType.AGENT_DEACTIVATED.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "resources": "float",
    },
}
