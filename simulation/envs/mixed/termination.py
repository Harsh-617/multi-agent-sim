"""Termination checks for the Mixed archetype.

V1 supports exactly three termination reasons:
  1. max_steps      — episode reached configured step limit
  2. system_collapse — shared pool fell to or below collapse_threshold
  3. no_active_agents — all agents deactivated (resources <= 0)
"""

from __future__ import annotations

from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.mixed.state import GlobalState


def check_termination(
    state: GlobalState,
    config: MixedEnvironmentConfig,
) -> TerminationReason | None:
    """Return the first applicable termination reason, or None."""
    pop = config.population

    # 1. System collapse (pool at or below threshold)
    if state.shared_pool <= pop.collapse_threshold:
        return TerminationReason.SYSTEM_COLLAPSE

    # 2. No active agents
    if not state.active_agent_ids():
        return TerminationReason.NO_ACTIVE_AGENTS

    # 3. Max steps reached
    if state.step >= pop.max_steps:
        return TerminationReason.MAX_STEPS

    return None
