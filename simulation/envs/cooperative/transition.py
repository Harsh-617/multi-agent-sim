"""Transition dynamics for the Cooperative archetype.

12-step fixed sequence (spec Part 6):
  1.  New tasks arrive (with bounded noise)
  2.  Collect & validate actions
  3.  Normalize effort amounts (already clamped in Action.__post_init__)
  4.  Apply specialization multipliers
  5.  Pool contributions by task type
  6.  Resolve task completions
  7.  Update task queue
  8.  Update system stress
  9.  Update agent-local state (contribution history, specialization scores)
  10. Update relational state (group-level signals)
  11. (Termination check is done in env.py)
  12. Advance timestep (done in env.py)

Resolutions from spec ambiguity section:
  - contribution_share is always 1.0 for the chosen type, 0.0 for others.
  - task_difficulty is always per-type (broadcasted at env init).
  - Division-by-zero in completion rate: if no tasks arrived, rate = 1.0.
  - Collapse condition uses integer comparison: backlog >= collapse_threshold.
"""

from __future__ import annotations

import numpy as np

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.cooperative.actions import Action
from simulation.envs.cooperative.state import GlobalState


def resolve_step(
    state: GlobalState,
    actions: dict[AgentID, Action],
    config: CooperativeEnvironmentConfig,
    rng: np.random.Generator,
) -> dict[AgentID, dict]:
    """Resolve one cooperative timestep.  Mutates *state* in place.

    Returns per-agent metadata dict (for reward computation):
      {agent_id: {"effort_amount": float, "chosen_type": int | None,
                  "effective_contribution": float, "completion_rate": float}}
    """
    pop = config.population
    layers = config.layers
    T = state.num_task_types()
    agent_ids = state.agent_ids()
    N = len(agent_ids)
    capacity = pop.agent_effort_capacity

    # -----------------------------------------------------------------------
    # Step 1: New task arrivals
    # -----------------------------------------------------------------------
    arrival_rates = _broadcast(config.task.task_arrival_rate, T)
    arrivals: list[int] = []
    for t in range(T):
        base = arrival_rates[t]
        noise_frac = layers.task_arrival_noise
        raw = base + rng.uniform(-noise_frac, noise_frac) * base
        arrivals.append(max(0, int(round(raw))))

    state.tasks_arrived_this_step = arrivals

    # -----------------------------------------------------------------------
    # Steps 2-3: Collect actions (already normalized in Action.__post_init__)
    # -----------------------------------------------------------------------
    # Fall back to IDLE for any agent with no action submitted
    effective_actions: dict[AgentID, Action] = {
        aid: actions.get(aid, Action(task_type=None))
        for aid in agent_ids
    }

    # -----------------------------------------------------------------------
    # Step 4: Compute effective contributions (with specialization bonus)
    # -----------------------------------------------------------------------
    effective_contrib: dict[AgentID, float] = {}
    for aid in agent_ids:
        act = effective_actions[aid]
        if act.is_idle:
            effective_contrib[aid] = 0.0
        else:
            t = act.task_type
            spec_bonus = (
                state.agents[aid].specialization_score[t]
                * layers.specialization_scale
            )
            effective_contrib[aid] = act.effort_amount * capacity * (1.0 + spec_bonus)

    # -----------------------------------------------------------------------
    # Step 5: Pool contributions by task type
    # -----------------------------------------------------------------------
    pooled: list[float] = [0.0] * T
    for aid in agent_ids:
        act = effective_actions[aid]
        if not act.is_idle:
            pooled[act.task_type] += effective_contrib[aid]

    # -----------------------------------------------------------------------
    # Step 6: Resolve task completions
    # -----------------------------------------------------------------------
    # Add arrivals first (arrivals happen before agents act — spec Part 3, step 1)
    for t in range(T):
        state.task_queue[t] += arrivals[t]

    completed: list[int] = []
    difficulties = state.task_difficulty
    for t in range(T):
        if difficulties[t] > 0 and pooled[t] > 0:
            c = min(int(pooled[t] / difficulties[t]), state.task_queue[t])
        else:
            c = 0
        completed.append(c)

    # -----------------------------------------------------------------------
    # Step 7: Update task queue
    # -----------------------------------------------------------------------
    for t in range(T):
        state.task_queue[t] -= completed[t]
        state.tasks_completed_this_step[t] = completed[t]
        state.tasks_completed_total[t] += completed[t]

    state.backlog_level = sum(state.task_queue)

    # -----------------------------------------------------------------------
    # Step 8: Update system stress
    # -----------------------------------------------------------------------
    collapse_thresh = config.task.collapse_threshold
    state.system_stress = min(state.backlog_level / collapse_thresh, 1.0)

    # -----------------------------------------------------------------------
    # Step 9: Update agent-local state
    # -----------------------------------------------------------------------
    decay = layers.specialization_decay
    history_window = layers.history_window

    for aid in agent_ids:
        agent = state.agents[aid]
        act = effective_actions[aid]
        chosen_type = act.task_type

        # Update last_effort record
        effort_record = [0.0] * T
        if chosen_type is not None:
            effort_record[chosen_type] = act.effort_amount

        agent.last_effort = effort_record
        agent.steps_active += 1

        # Append to contribution history, enforce max depth
        agent.contribution_history.append(effort_record)
        while len(agent.contribution_history) > history_window:
            agent.contribution_history.popleft()

        # Update specialization scores via EMA
        # Ambiguity #2 resolution: contribution_share = 1.0 for chosen, 0.0 for others
        for t in range(T):
            target = 1.0 if (chosen_type == t) else 0.0
            agent.specialization_score[t] = (
                (1.0 - decay) * agent.specialization_score[t] + decay * target
            )

    # -----------------------------------------------------------------------
    # Step 10: Update relational state (group-level signals)
    # -----------------------------------------------------------------------
    total_tasks_arrived = sum(arrivals)
    total_tasks_completed = sum(completed)

    # Per-step completion rate (Ambiguity #4: 1.0 when nothing arrived)
    if total_tasks_arrived == 0:
        step_completion_rate = 1.0
    else:
        step_completion_rate = min(total_tasks_completed / total_tasks_arrived, 1.0)

    rel = state.relational
    rel.completion_rate_history.append(step_completion_rate)
    while len(rel.completion_rate_history) > history_window:
        rel.completion_rate_history.popleft()

    rel.group_completion_rate = (
        sum(rel.completion_rate_history) / len(rel.completion_rate_history)
        if rel.completion_rate_history
        else 1.0
    )

    rel.group_contribution_last_step = sum(effective_contrib.values())
    rel.group_contribution_by_type = [
        sum(
            effective_contrib[aid]
            for aid in agent_ids
            if not effective_actions[aid].is_idle
            and effective_actions[aid].task_type == t
        )
        for t in range(T)
    ]

    expected_contribution = N * capacity * 0.5
    if expected_contribution > 0:
        shortfall = max(0.0, expected_contribution - rel.group_contribution_last_step)
        rel.free_rider_pressure = min(shortfall / expected_contribution, 1.0)
    else:
        rel.free_rider_pressure = 0.0

    # -----------------------------------------------------------------------
    # Build per-agent metadata for reward computation
    # -----------------------------------------------------------------------
    meta: dict[AgentID, dict] = {}
    for aid in agent_ids:
        act = effective_actions[aid]
        meta[aid] = {
            "effort_amount": act.effort_amount,
            "chosen_type": act.task_type,
            "effective_contribution": effective_contrib[aid],
            "completion_rate": step_completion_rate,
        }

    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broadcast(value: float | list[float], n: int) -> list[float]:
    """Return a list of length n: scalar is broadcast, list is used as-is."""
    if isinstance(value, (int, float)):
        return [float(value)] * n
    return [float(v) for v in value]
