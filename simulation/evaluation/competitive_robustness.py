"""Robustness evaluation for competitive policies across environment variants."""

from __future__ import annotations

import statistics
from typing import Any

from simulation.agents.competitive_baselines import create_competitive_agent
from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.envs.competitive.env import CompetitiveEnvironment
from simulation.metrics.competitive_collector import CompetitiveMetricsCollector

from .competitive_policy_set import CompetitivePolicySpec
from .competitive_sweeps import CompetitiveSweepSpec, apply_competitive_sweep


def _run_competitive_episode(
    config: CompetitiveEnvironmentConfig,
    policy_name: str,
    seed: int,
) -> dict[str, Any]:
    """Run one competitive episode where all agents use *policy_name*.

    Returns dict with reward, final_rank, is_winner, episode_length.
    """
    env = CompetitiveEnvironment(config)
    collector = CompetitiveMetricsCollector(config.instrumentation)
    observations = env.reset(seed=seed)

    agent_ids = env.active_agents()

    # Create agents — all use same policy
    agents = {}
    for i, aid in enumerate(agent_ids):
        agent = create_competitive_agent(policy_name)
        agent.reset(aid, derive_seed(seed, i))
        agents[aid] = agent

    eval_id = agent_ids[0]
    total_reward = 0.0
    step = 0

    while not env.is_done():
        active = env.active_agents()
        actions: dict[str, Any] = {}
        for aid in active:
            obs = observations.get(aid)
            if aid in agents:
                actions[aid] = agents[aid].act(obs)

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

        observations = {aid: sr.observation for aid, sr in results.items()}
        if eval_id in results:
            total_reward += results[eval_id].reward

        step += 1

    # Final state
    final_state = env._state
    final_scores = {aid: s.score for aid, s in final_state.agents.items()}
    final_rankings = final_state.rankings()

    final_rank = 1
    for rank_idx, (aid, _score) in enumerate(final_rankings, start=1):
        if aid == eval_id:
            final_rank = rank_idx
            break

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
        "episode_length": step,
    }


def run_competitive_robustness(
    config: CompetitiveEnvironmentConfig,
    policy_specs: list[CompetitivePolicySpec],
    seeds: list[int],
    episodes_per_seed: int = 1,
    sweeps: list[tuple[str, CompetitiveSweepSpec]] | None = None,
) -> dict[str, Any]:
    """Evaluate competitive policies across multiple environment variants.

    Parameters
    ----------
    config:
        Base environment config (not mutated).
    policy_specs:
        Policies to evaluate.
    seeds:
        Root seeds for reproducibility.
    episodes_per_seed:
        Episodes per seed per policy per sweep.
    sweeps:
        List of (sweep_name, sweep_spec) tuples. If None, uses defaults.

    Returns
    -------
    dict with keys: metadata, per_sweep_results, per_policy_robustness.
    """
    if sweeps is None:
        from .competitive_sweeps import build_competitive_default_sweeps
        sweeps = build_competitive_default_sweeps(config)

    result: dict[str, Any] = {}

    # Metadata
    result["metadata"] = {
        "sweeps": [name for name, _ in sweeps],
        "sweep_count": len(sweeps),
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "policy_count": len(policy_specs),
    }

    # Track per-policy rewards across sweeps for aggregation
    policy_sweep_rewards: dict[str, list[float]] = {}
    policy_sweep_winner_rates: dict[str, list[float]] = {}
    policy_sweep_names: dict[str, list[str]] = {}

    per_sweep_results: dict[str, dict[str, dict[str, Any]]] = {}

    for sweep_name, sweep_spec in sweeps:
        swept_config = apply_competitive_sweep(config, sweep_spec)

        sweep_data: dict[str, dict[str, Any]] = {}

        for spec in policy_specs:
            if not spec.available:
                sweep_data[spec.name] = {
                    "policy_name": spec.name,
                    "source": spec.source,
                    "available": False,
                    "skip_reason": spec.skip_reason,
                    "mean_reward": 0.0,
                    "winner_rate": 0.0,
                    "n_episodes": 0,
                }
                continue

            rewards: list[float] = []
            wins: list[bool] = []

            for seed in seeds:
                for ep_idx in range(episodes_per_seed):
                    ep_seed = derive_seed(seed, ep_idx)
                    try:
                        ep_result = _run_competitive_episode(
                            swept_config, spec.agent_policy, ep_seed,
                        )
                        rewards.append(ep_result["reward"])
                        wins.append(ep_result["is_winner"])
                    except (FileNotFoundError, KeyError):
                        continue

            mean_reward = (
                round(statistics.mean(rewards), 4) if rewards else 0.0
            )
            winner_rate = (
                round(sum(wins) / len(wins), 4) if wins else 0.0
            )

            sweep_data[spec.name] = {
                "policy_name": spec.name,
                "source": spec.source,
                "available": True,
                "mean_reward": mean_reward,
                "winner_rate": winner_rate,
                "n_episodes": len(rewards),
            }

            if rewards:
                policy_sweep_rewards.setdefault(spec.name, []).append(
                    mean_reward
                )
                policy_sweep_winner_rates.setdefault(spec.name, []).append(
                    winner_rate
                )
                policy_sweep_names.setdefault(spec.name, []).append(
                    sweep_name
                )

        per_sweep_results[sweep_name] = sweep_data

    result["per_sweep_results"] = per_sweep_results

    # Aggregate per-policy robustness
    per_policy_robustness: dict[str, dict[str, Any]] = {}

    for spec in policy_specs:
        name = spec.name
        rewards = policy_sweep_rewards.get(name, [])
        winner_rates = policy_sweep_winner_rates.get(name, [])
        sweep_names_for_policy = policy_sweep_names.get(name, [])

        if rewards:
            overall_mean = round(statistics.mean(rewards), 4)
            worst_case = round(min(rewards), 4)
            robustness_score = round(
                0.7 * overall_mean + 0.3 * worst_case, 4
            )
            mean_winner_rate = round(statistics.mean(winner_rates), 4)

            # Best and worst sweep
            best_idx = rewards.index(max(rewards))
            worst_idx = rewards.index(min(rewards))
            best_sweep = sweep_names_for_policy[best_idx]
            worst_sweep = sweep_names_for_policy[worst_idx]
        else:
            overall_mean = 0.0
            worst_case = 0.0
            robustness_score = 0.0
            mean_winner_rate = 0.0
            best_sweep = None
            worst_sweep = None

        per_policy_robustness[name] = {
            "policy_name": name,
            "overall_mean_reward": overall_mean,
            "worst_case_mean_reward": worst_case,
            "robustness_score": robustness_score,
            "mean_winner_rate": mean_winner_rate,
            "n_sweeps_evaluated": len(rewards),
            "best_sweep": best_sweep,
            "worst_sweep": worst_sweep,
        }

    result["per_policy_robustness"] = per_policy_robustness

    return result
