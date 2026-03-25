"""Metric names and minimal schemas for the Competitive archetype.

Defines three categories:
  - Step metrics: per-agent, per-step measurements
  - Episode metrics: summary of an entire episode
  - Event types: semantic events (elimination, rank changes, combat)

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
    "own_score",
    "own_resources",
    "own_rank",
    "num_active_agents",
    "attack_ratio",
    "defend_ratio",
    "build_ratio",
    "gamble_ratio",
]

STEP_METRIC_SCHEMA: dict[str, str] = {
    "step": "int",
    "agent_id": "str",
    "reward": "float",
    "action_type": "str",
    "action_amount": "float",
    "own_score": "float",
    "own_resources": "float",
    "own_rank": "int",
    "num_active_agents": "int",
    "attack_ratio": "float",
    "defend_ratio": "float",
    "build_ratio": "float",
    "gamble_ratio": "float",
}


# ---------------------------------------------------------------------------
# Episode metric keys (one record per episode)
# ---------------------------------------------------------------------------

EPISODE_METRIC_KEYS: list[str] = [
    "episode_length",
    "termination_reason",
    "final_rankings",
    "final_scores",
    "score_spread",
    "winner_id",
    "num_eliminations",
    "total_reward_per_agent",
]

EPISODE_METRIC_SCHEMA: dict[str, str] = {
    "episode_length": "int",
    "termination_reason": "str",
    "final_rankings": "list[str]",
    "final_scores": "dict[str, float]",
    "score_spread": "float",
    "winner_id": "str | None",
    "num_eliminations": "int",
    "total_reward_per_agent": "dict[str, float]",
}


# ---------------------------------------------------------------------------
# Semantic event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Semantic events emitted during a competitive simulation run."""

    AGENT_ELIMINATED = "agent_eliminated"
    RANK_CHANGE = "rank_change"
    ATTACK_SUCCEEDED = "attack_succeeded"
    ATTACK_DEFENDED = "attack_defended"
    GAMBLE_RESOLVED = "gamble_resolved"


EVENT_SCHEMAS: dict[str, dict[str, str]] = {
    EventType.AGENT_ELIMINATED.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "final_score": "float",
        "final_rank": "int",
    },
    EventType.RANK_CHANGE.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "old_rank": "int",
        "new_rank": "int",
    },
    EventType.ATTACK_SUCCEEDED.value: {
        "event": "str",
        "step": "int",
        "attacker_id": "str",
        "score_gained": "float",
    },
    EventType.ATTACK_DEFENDED.value: {
        "event": "str",
        "step": "int",
        "attacker_id": "str",
        "defender_id": "str",
        "cost_paid": "float",
    },
    EventType.GAMBLE_RESOLVED.value: {
        "event": "str",
        "step": "int",
        "agent_id": "str",
        "outcome": "float",
        "resources_before": "float",
        "resources_after": "float",
    },
}
