"""Reward computation for the Competitive archetype.

Computes per-agent reward as a weighted sum of:
  - absolute gain: normalised score change this step
  - relative gain: change in score gap vs opponents (fixed denominator =
    initial agent count; eliminated agents count as last-ranked)
  - efficiency: score gained per resource spent
  - penalty: failed-attack resource cost

Terminal bonus is NOT issued here — that belongs in env.py at episode end.
"""

from __future__ import annotations

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.types import AgentID
from simulation.envs.competitive.state import GlobalState


def compute_rewards(
    state: GlobalState,
    prev_scores: dict[AgentID, float],
    prev_resources: dict[AgentID, float],
    action_meta: dict[AgentID, dict],
    config: CompetitiveEnvironmentConfig,
) -> dict[AgentID, tuple[float, dict]]:
    """Return {agent_id: (scalar_reward, component_dict)} for every agent in *action_meta*.

    Must be called *after* transition has mutated state.
    """
    rw = config.rewards
    pop = config.population
    initial_count = pop.num_agents
    initial_resources = pop.initial_resources

    # Current scores for ALL agents (active and eliminated)
    all_scores_now: dict[AgentID, float] = {
        aid: s.score for aid, s in state.agents.items()
    }

    results: dict[AgentID, tuple[float, dict]] = {}

    for aid in action_meta:
        current_score = all_scores_now[aid]
        prev_score = prev_scores.get(aid, 0.0)

        # --- Absolute gain: normalised score change -------------------------
        score_delta = current_score - prev_score
        absolute_gain = score_delta / initial_resources if initial_resources > 0 else 0.0

        # --- Relative gain: change in score gap vs opponents ----------------
        # Always computed against INITIAL agent count (eliminated = frozen score)
        other_scores_now = [s for oid, s in all_scores_now.items() if oid != aid]
        other_scores_prev = [s for oid, s in prev_scores.items() if oid != aid]

        denom = initial_count - 1 if initial_count > 1 else 1
        mean_others_now = sum(other_scores_now) / denom
        mean_others_prev = sum(other_scores_prev) / denom

        gap_now = current_score - mean_others_now
        gap_prev = prev_score - mean_others_prev
        relative_gain = (
            (gap_now - gap_prev) / initial_resources if initial_resources > 0 else 0.0
        )

        # --- Efficiency: score gained per resource spent --------------------
        resource_spent = action_meta[aid]["resource_cost"]
        if resource_spent > 0:
            efficiency = score_delta / resource_spent
        else:
            # No resources spent — efficiency equals absolute gain (no penalty)
            efficiency = absolute_gain

        # --- Penalty: failed attack cost ------------------------------------
        penalty = 0.0
        if action_meta[aid]["attack_failed"] and resource_spent > 0:
            penalty = rw.penalty_scaling * (resource_spent / initial_resources)

        # --- Weighted sum ---------------------------------------------------
        scalar = (
            rw.absolute_gain_weight * absolute_gain
            + rw.relative_gain_weight * relative_gain
            + rw.efficiency_weight * efficiency
            - penalty
        )

        components = {
            "absolute_gain": absolute_gain,
            "relative_gain": relative_gain,
            "efficiency": efficiency,
            "penalty": penalty,
        }
        results[aid] = (scalar, components)

    return results
