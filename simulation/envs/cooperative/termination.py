"""Termination checks for the Cooperative archetype.

Three conditions (spec Part 8):
  1. max_steps          — timestep >= max_steps
  2. system_collapse    — backlog >= collapse_threshold for N consecutive steps
  3. perfect_clearance  — backlog == 0 for N consecutive steps (optional, off by default)

Ambiguity #5 resolution: collapse condition uses integer comparison
backlog >= collapse_threshold (not float equality on system_stress).
"""

from __future__ import annotations

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.cooperative.state import GlobalState


def check_termination(
    state: GlobalState,
    config: CooperativeEnvironmentConfig,
) -> TerminationReason | None:
    """Update sustain-window counters and return termination reason or None.

    Must be called once per step *after* transition has updated state.
    Mutates state.consecutive_collapse_steps and state.consecutive_clearance_steps.
    """
    pop = config.population
    task = config.task

    # --- Update collapse window counter ---
    if state.backlog_level >= task.collapse_threshold:
        state.consecutive_collapse_steps += 1
    else:
        state.consecutive_collapse_steps = 0

    # --- Update clearance window counter ---
    if state.backlog_level == 0:
        state.consecutive_clearance_steps += 1
    else:
        state.consecutive_clearance_steps = 0

    # Condition 2: System collapse (failure)
    if state.consecutive_collapse_steps >= pop.collapse_sustain_window:
        return TerminationReason.SYSTEM_COLLAPSE

    # Condition 3: Perfect clearance (optional success)
    if (
        pop.enable_early_success
        and state.consecutive_clearance_steps >= pop.clearance_sustain_window
    ):
        return TerminationReason.PERFECT_CLEARANCE

    # Condition 1: Max steps reached
    if state.step >= pop.max_steps:
        return TerminationReason.MAX_STEPS

    return None
