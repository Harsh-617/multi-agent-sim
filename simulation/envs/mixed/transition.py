"""Transition logic — collective resolution of simultaneous actions.

Deterministic given config + seed.  All randomness flows through the
provided NumPy Generator.

Action semantics (V1, no targeting):
  COOPERATE(amount): Agent contributes `amount * own_resources` to the shared pool.
  EXTRACT(amount):   Agent takes `amount * extract_capacity` from the shared pool.
                     extract_capacity = shared_pool / num_active_agents.
  DEFEND:            Agent does nothing; gains a small defensive bonus and avoids
                     extraction penalties.
  CONDITIONAL:       Acts like COOPERATE if group cooperation ratio last step >= 0.5,
                     else acts like DEFEND.
"""

from __future__ import annotations

import numpy as np

from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.mixed.actions import Action, ActionType
from simulation.envs.mixed.state import GlobalState


def resolve_actions(
    state: GlobalState,
    actions: dict[AgentID, Action],
    config: MixedEnvironmentConfig,
    rng: np.random.Generator,
) -> dict[AgentID, dict]:
    """Resolve one step of simultaneous actions.  Mutates *state* in place.

    Returns per-agent action metadata (for reward computation).
    """
    pop = config.population
    layers = config.layers
    active = state.active_agent_ids()
    n_active = len(active)

    if n_active == 0:
        return {}

    # --- Phase 0: resolve CONDITIONAL into effective types ----------------
    effective: dict[AgentID, Action] = {}
    for aid in active:
        act = actions.get(aid, Action(type=ActionType.DEFEND))
        if act.type == ActionType.CONDITIONAL:
            # Look at last-step cooperation ratio
            coop_ratio = _last_step_coop_ratio(state, active)
            if coop_ratio >= 0.5:
                act = Action(type=ActionType.COOPERATE, amount=0.5)
            else:
                act = Action(type=ActionType.DEFEND)
        effective[aid] = act

    # --- Phase 1: contributions (COOPERATE) -------------------------------
    contributions: dict[AgentID, float] = {}
    for aid in active:
        act = effective[aid]
        if act.type == ActionType.COOPERATE:
            agent = state.agents[aid]
            contrib = act.amount * agent.resources
            contributions[aid] = contrib
            agent.resources -= contrib
            state.shared_pool += contrib
        else:
            contributions[aid] = 0.0

    # --- Phase 2: extractions (EXTRACT) -----------------------------------
    # Fair-share capacity: each extractor can pull from pool / n_active
    extract_capacity = state.shared_pool / n_active if n_active > 0 else 0.0
    extractions: dict[AgentID, float] = {}
    for aid in active:
        act = effective[aid]
        if act.type == ActionType.EXTRACT:
            desired = act.amount * extract_capacity
            actual = min(desired, state.shared_pool)
            extractions[aid] = actual
            state.shared_pool -= actual
            state.agents[aid].resources += actual
        else:
            extractions[aid] = 0.0

    # --- Phase 3: defensive bonus (DEFEND) --------------------------------
    defend_bonus = 0.01 * pop.initial_agent_resources
    for aid in active:
        if effective[aid].type == ActionType.DEFEND:
            state.agents[aid].resources += defend_bonus

    # --- Phase 4: relational updates -------------------------------------
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            both_coop = (
                effective[a].type == ActionType.COOPERATE
                and effective[b].type == ActionType.COOPERATE
            )
            state.get_relation(a, b).update(both_coop, layers.reputation_sensitivity)

    # --- Phase 5: inject noise -------------------------------------------
    noise_scale = layers.uncertainty_intensity * pop.initial_shared_pool * 0.01
    if noise_scale > 0:
        pool_noise = rng.normal(0, noise_scale)
        state.shared_pool = max(0.0, state.shared_pool + pool_noise)

    # --- Phase 6: deactivate broke agents --------------------------------
    for aid in active:
        if state.agents[aid].resources <= 0:
            state.agents[aid].resources = 0.0
            state.agents[aid].active = False

    # --- Build metadata for reward computation ----------------------------
    meta: dict[AgentID, dict] = {}
    for aid in active:
        meta[aid] = {
            "effective_action": effective[aid],
            "contributed": contributions[aid],
            "extracted": extractions[aid],
        }

    return meta


def _last_step_coop_ratio(state: GlobalState, active: list[AgentID]) -> float:
    """Fraction of active agents whose last action was COOPERATE."""
    if not active:
        return 0.0
    coop_count = 0
    total = 0
    for aid in active:
        hist = state.agents[aid].action_history
        if hist:
            total += 1
            if hist[-1].type == ActionType.COOPERATE:
                coop_count += 1
    return coop_count / total if total > 0 else 0.5  # default 0.5 on step 0
