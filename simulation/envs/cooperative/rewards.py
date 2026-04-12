"""Reward computation for the Cooperative archetype.

reward[agent] = w_group * R_group + w_individual * R_individual + w_efficiency * R_efficiency

All three components are bounded [0, 1].  The weighted sum is also bounded
[0, 1] because weights sum to 1.0.

Component definitions (spec Part 7):
  R_group      = α * completion_rate_this_step + (1-α) * (1 - system_stress)
  R_individual = effort_amount if WORK, else 0.0
  R_efficiency = specialization_score[agent][chosen_type] * completion_rate_this_step
"""

from __future__ import annotations

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.cooperative.state import GlobalState

# Balance within R_group between throughput and stress relief
_ALPHA = 0.6


def compute_rewards(
    state: GlobalState,
    action_meta: dict[AgentID, dict],
    config: CooperativeEnvironmentConfig,
) -> dict[AgentID, tuple[float, dict]]:
    """Return {agent_id: (scalar_reward, component_dict)} for every agent.

    Must be called *after* transition has mutated state for this step.
    """
    rw = config.rewards

    # Group component is identical for all agents
    completion_rate = action_meta[next(iter(action_meta))]["completion_rate"]
    r_group = _ALPHA * completion_rate + (1.0 - _ALPHA) * (1.0 - state.system_stress)
    r_group = max(0.0, min(1.0, r_group))

    results: dict[AgentID, tuple[float, dict]] = {}

    for aid, meta in action_meta.items():
        chosen_type = meta["chosen_type"]
        effort = meta["effort_amount"]

        # Individual component
        r_individual = effort if chosen_type is not None else 0.0

        # Efficiency component
        if chosen_type is not None:
            spec = state.agents[aid].specialization_score[chosen_type]
        else:
            spec = 0.0
        r_efficiency = spec * completion_rate
        r_efficiency = max(0.0, min(1.0, r_efficiency))

        # Weighted sum — bounded [0, 1] by construction (weights sum to 1)
        scalar = (
            rw.w_group * r_group
            + rw.w_individual * r_individual
            + rw.w_efficiency * r_efficiency
        )
        scalar = max(0.0, min(1.0, scalar))

        components = {
            "r_group": r_group,
            "r_individual": r_individual,
            "r_efficiency": r_efficiency,
        }
        results[aid] = (scalar, components)

    return results
