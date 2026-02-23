"""Reward computation for the Mixed archetype.

Computes per-agent reward as a weighted sum of:
  - individual: change in personal resources
  - group: shared pool health (normalised)
  - relational: mean cooperation score with other active agents

Also returns a component breakdown dict for logging; the env only
surfaces the scalar reward in StepResult.
"""

from __future__ import annotations

from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.mixed.actions import ActionType
from simulation.envs.mixed.state import GlobalState


def compute_rewards(
    state: GlobalState,
    prev_resources: dict[AgentID, float],
    prev_pool: float,
    action_meta: dict[AgentID, dict],
    config: MixedEnvironmentConfig,
) -> dict[AgentID, tuple[float, dict]]:
    """Return {agent_id: (scalar_reward, component_dict)} for every active agent.

    Must be called *after* transition has mutated state.
    """
    rw = config.rewards
    pop = config.population
    results: dict[AgentID, tuple[float, dict]] = {}

    for aid in action_meta:
        agent = state.agents[aid]

        # --- Individual component: normalised resource change ---
        delta = agent.resources - prev_resources[aid]
        individual = delta / pop.initial_agent_resources

        # --- Group component: normalised pool health ---
        pool_health = state.shared_pool / pop.initial_shared_pool
        group = pool_health

        # --- Relational component: mean cooperation score with others ---
        active_others = [o for o in state.active_agent_ids() if o != aid]
        if active_others:
            relational = sum(
                state.get_relation(aid, o).cooperation_score
                for o in active_others
            ) / len(active_others)
        else:
            relational = 0.0

        # --- Penalty for extraction when pool is low ---
        penalty = 0.0
        eff_action = action_meta[aid]["effective_action"]
        if eff_action.type == ActionType.EXTRACT:
            pool_ratio = state.shared_pool / pop.initial_shared_pool
            if pool_ratio < 0.3:
                penalty = rw.penalty_scaling * (0.3 - pool_ratio) * eff_action.amount

        # --- Weighted sum ---
        scalar = (
            rw.individual_weight * individual
            + rw.group_weight * group
            + rw.relational_weight * relational
            - penalty
        )

        components = {
            "individual": individual,
            "group": group,
            "relational": relational,
            "penalty": penalty,
        }
        results[aid] = (scalar, components)

    return results
