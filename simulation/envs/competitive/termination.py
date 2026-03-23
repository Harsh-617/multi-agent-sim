"""Termination checks for the Competitive archetype.

V1 supports exactly three termination reasons (checked in this order):
  1. no_active_agents — all agents eliminated
  2. elimination      — one survivor remains
  3. max_steps        — episode reached configured step limit

DOMINANCE is explicitly not implemented in V1.
"""

from __future__ import annotations

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.competitive.state import GlobalState


def check_termination(
    state: GlobalState,
    config: CompetitiveEnvironmentConfig,
) -> TerminationReason | None:
    """Return the first applicable termination reason, or None."""
    pop = config.population

    # 1. No active agents (all eliminated simultaneously)
    if not state.active_agent_ids():
        return TerminationReason.NO_ACTIVE_AGENTS

    # 2. Elimination: exactly one survivor
    if len(state.active_agent_ids()) == 1:
        return TerminationReason.ELIMINATION

    # 3. Max steps reached
    if state.step >= pop.max_steps:
        return TerminationReason.MAX_STEPS

    return None
