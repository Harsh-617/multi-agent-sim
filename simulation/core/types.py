"""Framework-level types used across all environments.

These are the shared vocabulary of the simulation framework.
Domain-specific types (e.g., Mixed archetype actions) live in their
respective env packages, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

AgentID = str  # unique within an episode


# ---------------------------------------------------------------------------
# Step result (what the environment returns per agent per step)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StepResult:
    """Per-agent output of a single environment step."""

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class TerminationReason(Enum):
    """Why an episode ended.  V1 supports exactly three reasons."""

    MAX_STEPS = "max_steps"
    SYSTEM_COLLAPSE = "system_collapse"
    NO_ACTIVE_AGENTS = "no_active_agents"
