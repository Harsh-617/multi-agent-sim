"""Core evaluation loop: run policies across seeds and aggregate stats."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from simulation.agents import create_agent
from simulation.agents.base import BaseAgent
from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.envs.mixed import MixedEnvironment
from simulation.metrics.collector import MetricsCollector

from .policy_set import PolicySpec


@dataclass
class EpisodeStat:
    """Raw stats from one episode."""

    seed: int
    episode_idx: int
    total_reward: float
    final_shared_pool: float
    episode_length: int
    termination_reason: str | None


@dataclass
class PolicyResult:
    """Aggregated evaluation results for one policy."""

    spec: PolicySpec
    episodes: list[EpisodeStat] = field(default_factory=list)

    # --- aggregated (filled by _aggregate) ---
    mean_total_reward: float = 0.0
    std_total_reward: float = 0.0
    mean_final_shared_pool: float = 0.0
    std_final_shared_pool: float = 0.0
    collapse_rate: float = 0.0
    mean_episode_length: float = 0.0

    # Per-seed breakdown for reproducibility reporting.
    per_seed: list[dict[str, Any]] = field(default_factory=list)

    def aggregate(self) -> None:
        rewards = [e.total_reward for e in self.episodes]
        pools = [e.final_shared_pool for e in self.episodes]
        lengths = [e.episode_length for e in self.episodes]
        collapses = sum(
            1 for e in self.episodes if e.termination_reason == "system_collapse"
        )
        n = max(len(self.episodes), 1)

        self.mean_total_reward = round(statistics.mean(rewards), 4) if rewards else 0.0
        self.std_total_reward = (
            round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0
        )
        self.mean_final_shared_pool = (
            round(statistics.mean(pools), 4) if pools else 0.0
        )
        self.std_final_shared_pool = (
            round(statistics.stdev(pools), 4) if len(pools) > 1 else 0.0
        )
        self.collapse_rate = round(collapses / n, 4)
        self.mean_episode_length = round(statistics.mean(lengths), 2) if lengths else 0

        # Per-seed breakdown
        seed_map: dict[int, list[EpisodeStat]] = {}
        for e in self.episodes:
            seed_map.setdefault(e.seed, []).append(e)

        self.per_seed = []
        for seed, eps in sorted(seed_map.items()):
            r = [e.total_reward for e in eps]
            self.per_seed.append(
                {
                    "seed": seed,
                    "n_episodes": len(eps),
                    "mean_total_reward": round(statistics.mean(r), 4),
                    "collapse_count": sum(
                        1 for e in eps if e.termination_reason == "system_collapse"
                    ),
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.spec.name,
            "source": self.spec.source,
            "league_member_id": self.spec.league_member_id,
            "available": self.spec.available,
            "skip_reason": self.spec.skip_reason,
            "mean_total_reward": self.mean_total_reward,
            "std_total_reward": self.std_total_reward,
            "mean_final_shared_pool": self.mean_final_shared_pool,
            "std_final_shared_pool": self.std_final_shared_pool,
            "collapse_rate": self.collapse_rate,
            "mean_episode_length": self.mean_episode_length,
            "per_seed": self.per_seed,
            "n_episodes": len(self.episodes),
        }


def _run_episode(
    config: MixedEnvironmentConfig,
    agent_policy: str,
    seed: int,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one episode and return the MetricsCollector summary."""
    env = MixedEnvironment(config)
    collector = MetricsCollector(config.instrumentation)

    initial_obs = env.reset(seed=seed)
    kw = agent_kwargs or {}
    agents: dict[str, BaseAgent] = {}
    for i, aid in enumerate(env.active_agents()):
        agent = create_agent(agent_policy, **kw)
        agent.reset(aid, derive_seed(seed, i))
        agents[aid] = agent

    observations: dict[str, dict] = dict(initial_obs)
    shared_pool = config.population.initial_shared_pool

    while not env.is_done():
        actions: dict[str, Any] = {}
        for aid in env.active_agents():
            obs = observations.get(aid, {})
            actions[aid] = agents[aid].act(obs)

        results = env.step(actions)
        for aid, sr in results.items():
            observations[aid] = sr.observation

        agent_resources = {
            aid: results[aid].observation.get("own_resources", 0.0) for aid in results
        }
        shared_pool = next(iter(results.values())).observation.get(
            "shared_pool", 0.0
        )

        collector.collect_step(
            step=env.current_step,
            actions=actions,
            results=results,
            shared_pool=shared_pool,
            agent_resources=agent_resources,
            active_agents=env.active_agents(),
        )

    reason = env.termination_reason()
    return collector.episode_summary(
        episode_length=env.current_step,
        termination_reason=reason,
        final_shared_pool=shared_pool,
    )


def evaluate_policies(
    config: MixedEnvironmentConfig,
    policy_specs: list[PolicySpec],
    *,
    seeds: list[int],
    episodes_per_seed: int = 1,
    max_steps_override: int | None = None,
) -> list[PolicyResult]:
    """Evaluate each available policy across seeds and episodes.

    Parameters
    ----------
    config:
        Environment configuration (not mutated).
    policy_specs:
        Policies to evaluate (unavailable ones are included in output
        but not run).
    seeds:
        Root seeds for reproducibility.
    episodes_per_seed:
        How many episodes to run per seed.
    max_steps_override:
        If set, temporarily override ``config.population.max_steps``
        for faster evaluation.
    """
    # Optionally override max_steps without mutating the original config
    if max_steps_override is not None:
        config = config.model_copy(
            update={
                "population": config.population.model_copy(
                    update={"max_steps": max_steps_override}
                )
            }
        )

    results: list[PolicyResult] = []

    for spec in policy_specs:
        pr = PolicyResult(spec=spec)

        if not spec.available:
            pr.aggregate()
            results.append(pr)
            continue

        for seed in seeds:
            for ep_idx in range(episodes_per_seed):
                ep_seed = derive_seed(seed, ep_idx)
                summary = _run_episode(
                    config,
                    spec.agent_policy,
                    ep_seed,
                    agent_kwargs=spec.agent_kwargs or None,
                )
                rewards = summary.get("total_reward_per_agent", {})
                mean_r = sum(rewards.values()) / len(rewards) if rewards else 0.0

                pr.episodes.append(
                    EpisodeStat(
                        seed=seed,
                        episode_idx=ep_idx,
                        total_reward=round(mean_r, 4),
                        final_shared_pool=summary.get("final_shared_pool", 0.0),
                        episode_length=summary.get("episode_length", 0),
                        termination_reason=(
                            summary.get("termination_reason")
                            if isinstance(summary.get("termination_reason"), str)
                            else (
                                summary["termination_reason"].value
                                if summary.get("termination_reason") is not None
                                else None
                            )
                        ),
                    )
                )

        pr.aggregate()
        results.append(pr)

    return results
