"""State representations for the Cooperative archetype.

Three-tier state model:
  - GlobalState   : true world state (task queue, stress, step counter)
  - AgentState    : per-agent private state (specialization, history)
  - RelationalState: group-level aggregate signals (no per-agent breakdown)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.core.types import AgentID
from simulation.envs.cooperative.actions import Action


# ---------------------------------------------------------------------------
# Agent-local state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentState:
    """Per-agent private mutable state."""

    agent_id: AgentID
    # Specialization score per task type — EMA-updated each step; bounded [0, 1]
    specialization_score: list[float]
    # Rolling window: each entry is a list[float] of effort per task type for that step
    contribution_history: deque[list[float]] = field(default_factory=deque)
    # Accumulated reward this episode
    cumulative_reward: float = 0.0
    # How many steps this agent has been active
    steps_active: int = 0
    # Effort allocated per task type in the most recent step (set by transition)
    last_effort: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Relational / group-level state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RelationalState:
    """Group-level aggregate signals — never per-agent."""

    # Total effective effort contributed by all agents last step
    group_contribution_last_step: float = 0.0
    # Effective effort per task type last step
    group_contribution_by_type: list[float] = field(default_factory=list)
    # Rolling average of per-step completion rates
    group_completion_rate: float = 1.0
    # Normalised signal: how far group contribution fell below expected
    free_rider_pressure: float = 0.0
    # History of per-step completion rates for rolling average
    completion_rate_history: deque[float] = field(default_factory=deque)


# ---------------------------------------------------------------------------
# Global / world state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GlobalState:
    """Complete mutable state of a Cooperative environment episode."""

    # Task world
    task_queue: list[int]            # pending tasks per type
    task_difficulty: list[float]     # effort required per task per type (drawn at reset)
    tasks_completed_this_step: list[int]
    tasks_completed_total: list[int]
    tasks_arrived_this_step: list[int]

    # Scalar world state
    backlog_level: int               # sum(task_queue) — kept in sync
    system_stress: float             # backlog / collapse_threshold, clamped [0, 1]
    step: int

    # Termination window counters
    consecutive_collapse_steps: int = 0
    consecutive_clearance_steps: int = 0

    # Per-agent state
    agents: dict[AgentID, AgentState] = field(default_factory=dict)

    # Group relational state
    relational: RelationalState = field(default_factory=RelationalState)

    def agent_ids(self) -> list[AgentID]:
        return list(self.agents.keys())

    def num_agents(self) -> int:
        return len(self.agents)

    def num_task_types(self) -> int:
        return len(self.task_queue)
