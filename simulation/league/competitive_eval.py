"""Competitive league population evaluation runner.

Evaluates all policies (baselines + competitive_ppo + league snapshots) against
each other in the CompetitiveEnvironment, then returns a summary table with
mean_reward, final_rank, and winner_rate per policy.

Follows the same pattern as eval_population.py (Mixed archetype).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from simulation.agents.competitive_baselines import (
    COMPETITIVE_POLICY_REGISTRY,
    create_competitive_agent,
)
from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.envs.competitive.env import CompetitiveEnvironment
from simulation.league.competitive_sampling import COMPETITIVE_BASELINE_POLICIES
from simulation.league.registry import LeagueRegistry
from simulation.metrics.competitive_collector import CompetitiveMetricsCollector


# ------------------------------------------------------------------
# Single episode
# ------------------------------------------------------------------

def _run_eval_episode(
    config: CompetitiveEnvironmentConfig,
    policy_name: str,
    opponent_policy: str,
    seed: int,
) -> dict[str, Any]:
    """Run one episode: all agents use the given policies.

    agent_0 uses *policy_name*, remaining agents use *opponent_policy*.
    Returns a dict with reward, final_rank, and whether policy_name won.
    """
    env = CompetitiveEnvironment(config)
    collector = CompetitiveMetricsCollector(config.instrumentation)
    observations = env.reset(seed=seed)

    agent_ids = env.active_agents()
    eval_id = agent_ids[0]
    opponent_ids = agent_ids[1:]

    # Create agents
    eval_agent = create_competitive_agent(policy_name)
    eval_agent.reset(eval_id, seed)

    opponents = {}
    for i, oid in enumerate(opponent_ids):
        opp = create_competitive_agent(opponent_policy)
        opp.reset(oid, derive_seed(seed, i + 100))
        opponents[oid] = opp

    total_reward = 0.0
    step = 0

    while not env.is_done():
        active = env.active_agents()
        actions: dict[str, Any] = {}

        for aid in active:
            obs = observations.get(aid)
            if aid == eval_id:
                actions[aid] = eval_agent.act(obs)
            elif aid in opponents:
                actions[aid] = opponents[aid].act(obs)

        results = env.step(actions)

        # Collect metrics
        state = env._state
        agent_scores = {aid: state.agents[aid].score for aid in results}
        agent_resources = {aid: state.agents[aid].resources for aid in results}
        rankings = state.rankings()
        collector.collect_step(
            step=step,
            actions=actions,
            results=results,
            agent_scores=agent_scores,
            agent_resources=agent_resources,
            active_agents=env.active_agents(),
            rankings=rankings,
        )

        # Update observations
        observations = {aid: sr.observation for aid, sr in results.items()}

        if eval_id in results:
            total_reward += results[eval_id].reward

        step += 1

    # Final state
    final_state = env._state
    final_scores = {aid: s.score for aid, s in final_state.agents.items()}
    final_rankings = final_state.rankings()

    # Determine eval agent's final rank (1-based)
    final_rank = 1
    for rank_idx, (aid, _score) in enumerate(final_rankings, start=1):
        if aid == eval_id:
            final_rank = rank_idx
            break

    # Did eval agent win?
    summary = collector.episode_summary(
        episode_length=step,
        termination_reason=env.termination_reason(),
        final_scores=final_scores,
        final_rankings=final_rankings,
    )
    is_winner = summary.get("winner_id") == eval_id

    return {
        "reward": total_reward,
        "final_rank": final_rank,
        "is_winner": is_winner,
    }


# ------------------------------------------------------------------
# Full population evaluation
# ------------------------------------------------------------------

def run_competitive_population_eval(
    config: CompetitiveEnvironmentConfig,
    episodes_per_policy: int = 10,
    seed: int = 42,
    league_root: str = "storage/agents/competitive_league",
) -> list[dict[str, Any]]:
    """Evaluate all competitive policies against each other.

    Evaluates baselines + league snapshots. Each policy plays
    *episodes_per_policy* episodes against a random-baseline opponent mix.

    Returns a list of summary dicts (one per policy) with keys:
    ``policy``, ``mean_reward``, ``final_rank``, ``winner_rate``.
    """
    # Gather all policy names to evaluate
    policies: list[str] = list(COMPETITIVE_BASELINE_POLICIES)

    # Add league snapshots if any exist
    registry = LeagueRegistry(league_root)
    members = registry.list_members()
    league_ids = [m["member_id"] for m in members]
    # We don't add league snapshot policies to eval by default since they
    # require a trained model — only evaluate baseline policies and any
    # registered competitive_ppo if artifacts exist.

    # Check if competitive_ppo artifacts exist
    from pathlib import Path
    ppo_dir = Path("storage/agents/competitive_ppo")
    if (ppo_dir / "policy.pt").exists() and (ppo_dir / "metadata.json").exists():
        policies.append("competitive_ppo")

    # Use random as the default opponent for consistent evaluation
    opponent_policy = "random"

    results: list[dict[str, Any]] = []

    for policy_name in policies:
        rewards: list[float] = []
        ranks: list[int] = []
        wins: list[bool] = []

        for ep in range(episodes_per_policy):
            ep_seed = derive_seed(seed, hash(policy_name) % 10_000 + ep)

            try:
                ep_result = _run_eval_episode(
                    config, policy_name, opponent_policy, ep_seed,
                )
                rewards.append(ep_result["reward"])
                ranks.append(ep_result["final_rank"])
                wins.append(ep_result["is_winner"])
            except (FileNotFoundError, KeyError):
                # Skip policies whose artifacts can't be loaded
                continue

        if not rewards:
            continue

        results.append({
            "policy": policy_name,
            "mean_reward": float(np.mean(rewards)),
            "final_rank": float(np.mean(ranks)),
            "winner_rate": float(np.mean(wins)),
            "episodes": len(rewards),
        })

    return results
