"""Transition logic — collective resolution of simultaneous actions.

Deterministic given config + seed.  All randomness flows through the
provided NumPy Generator.

Action semantics (V1, untargeted):
  BUILD(amount):  Convert resources to score (1:1, capped by budget).
  ATTACK(amount): Pay resources, distribute score damage across non-defending
                  opponents.  Negated by DEFEND; reduced against GAMBLE.
  DEFEND:         Negate incoming attacks.  +20 % bonus if attacker is
                  predictable (history sensitivity rule).
  GAMBLE:         No resource cost.  Score = initial_resources × U(0, 2.5).
"""

from __future__ import annotations

import numpy as np

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.competitive.actions import Action, ActionType
from simulation.envs.competitive.state import GlobalState


def resolve_actions(
    state: GlobalState,
    actions: dict[AgentID, Action],
    config: CompetitiveEnvironmentConfig,
    rng: np.random.Generator,
) -> dict[AgentID, dict]:
    """Resolve one step of simultaneous actions.  Mutates *state* in place.

    Returns per-agent action metadata (for reward computation).
    """
    pop = config.population
    layers = config.layers
    active = state.active_agent_ids()

    if not active:
        return {}

    build_base = pop.initial_resources

    # --- Phase 0: validate and normalise actions ----------------------------
    effective: dict[AgentID, Action] = {}
    for aid in active:
        act = actions.get(aid, Action(type=ActionType.DEFEND))
        if act.type in (ActionType.BUILD, ActionType.ATTACK):
            clamped = max(0.0, min(1.0, act.amount))
            if clamped != act.amount:
                act = Action(type=act.type, amount=clamped)
        effective[aid] = act

    # Initialise per-agent metadata
    meta: dict[AgentID, dict] = {}
    for aid in active:
        meta[aid] = {
            "effective_action": effective[aid],
            "score_gain": 0.0,
            "resource_cost": 0.0,
            "attack_failed": False,
        }

    # --- Phase 1: resolve GAMBLE actions ------------------------------------
    for aid in active:
        if effective[aid].type == ActionType.GAMBLE:
            multiplier = rng.uniform(0.0, 2.5)
            gain = build_base * multiplier
            state.agents[aid].score += gain
            meta[aid]["score_gain"] += gain

    # --- Phase 2: resolve ATTACK vs DEFEND clashes --------------------------
    attackers = [aid for aid in active if effective[aid].type == ActionType.ATTACK]

    # Compute all attack effects on a snapshot basis (simultaneous)
    score_deltas: dict[AgentID, float] = {aid: 0.0 for aid in active}
    resource_costs: dict[AgentID, float] = {aid: 0.0 for aid in active}

    for atk_id in attackers:
        agent = state.agents[atk_id]
        amt = effective[atk_id].amount

        # Resource cost (capped by current resources)
        desired_cost = amt * build_base
        actual_cost = min(desired_cost, agent.resources)
        resource_costs[atk_id] += actual_cost
        attack_strength = actual_cost  # 1:1 resource→attack power

        opponents = [oid for oid in active if oid != atk_id]
        non_defending = [
            oid for oid in opponents if effective[oid].type != ActionType.DEFEND
        ]
        defending = [
            oid for oid in opponents if effective[oid].type == ActionType.DEFEND
        ]

        # History sensitivity: is the attacker predictable?
        atk_history = list(agent.action_history)[-layers.opponent_history_depth :]
        attacker_predictable = False
        if atk_history:
            atk_ratio = sum(
                1 for a in atk_history if a.type == ActionType.ATTACK
            ) / len(atk_history)
            attacker_predictable = atk_ratio >= 0.5

        if not non_defending:
            # All opponents defending — attack fully negated
            meta[atk_id]["attack_failed"] = True
            if attacker_predictable and defending:
                would_be_share = attack_strength / len(opponents) if opponents else 0.0
                bonus = 0.20 * would_be_share
                for def_id in defending:
                    score_deltas[def_id] += bonus
            continue

        # Distribute attack evenly across non-defending opponents
        share = attack_strength / len(non_defending)
        total_atk_gain = 0.0

        for opp_id in non_defending:
            opp_type = effective[opp_id].type
            if opp_type == ActionType.GAMBLE:
                actual_share = share * 0.5  # Partial success
            else:
                actual_share = share  # Full success (BUILD or ATTACK)

            score_deltas[atk_id] += actual_share
            score_deltas[opp_id] -= actual_share
            total_atk_gain += actual_share

        # Predictable attacker — defenders still get bonus
        if attacker_predictable and defending:
            would_be_share = attack_strength / len(opponents) if opponents else 0.0
            bonus = 0.20 * would_be_share
            for def_id in defending:
                score_deltas[def_id] += bonus

        if total_atk_gain == 0.0:
            meta[atk_id]["attack_failed"] = True

    # Apply Phase-2 results atomically
    for aid in active:
        state.agents[aid].score += score_deltas[aid]
        state.agents[aid].resources -= resource_costs[aid]
        meta[aid]["score_gain"] += score_deltas[aid]
        meta[aid]["resource_cost"] += resource_costs[aid]

    # --- Phase 3: resolve BUILD actions (independent) -----------------------
    for aid in active:
        if effective[aid].type == ActionType.BUILD:
            agent = state.agents[aid]
            desired_cost = effective[aid].amount * build_base
            actual_cost = min(desired_cost, agent.resources)
            score_gain = actual_cost  # 1:1 resource→score
            agent.resources -= actual_cost
            agent.score += score_gain
            meta[aid]["score_gain"] += score_gain
            meta[aid]["resource_cost"] += actual_cost

    # --- Phase 4: update scores and resources -------------------------------
    for aid in active:
        agent = state.agents[aid]
        agent.resources += pop.resource_regeneration_rate
        agent.resources = max(0.0, agent.resources)
        agent.score = max(0.0, agent.score)

    # --- Phase 5: update opponent history state -----------------------------
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            hist = state.get_opponent_history(a, b)
            hist.record(effective[a].type, layers.opponent_history_depth)
            hist.record(effective[b].type, layers.opponent_history_depth)

    # --- Phase 6: inject controlled noise -----------------------------------
    noise_scale = layers.uncertainty_intensity * build_base * 0.01
    if noise_scale > 0:
        for aid in active:
            score_noise = rng.normal(0, noise_scale)
            state.agents[aid].score = max(
                0.0, state.agents[aid].score + score_noise
            )

    # --- Phase 7: deactivate eliminated agents ------------------------------
    for aid in active:
        if state.agents[aid].resources <= pop.elimination_threshold:
            state.agents[aid].resources = 0.0
            state.agents[aid].active = False

    return meta
