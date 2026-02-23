"""League population evaluation runner.

CLI usage::

    python -m simulation.league.eval_population \\
        --config-id default --episodes 20 --agent-policy tit_for_tat

Evaluates an agent policy against a sampled population of opponents drawn
from league snapshots, baselines, and optionally a fixed trained policy.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.agents import create_agent
from simulation.agents.base import BaseAgent
from simulation.config.defaults import default_config
from simulation.core.seeding import derive_seed
from simulation.envs.mixed.env import MixedEnvironment
from simulation.league.registry import LeagueRegistry
from simulation.league.sampling import (
    BASELINE_POLICIES,
    OpponentSampler,
    OpponentSpec,
    SamplingWeights,
)


# ------------------------------------------------------------------
# Episode result
# ------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""

    episode: int
    opponent_spec: OpponentSpec
    return_total: float
    final_shared_pool: float
    termination_reason: str | None
    episode_length: int


# ------------------------------------------------------------------
# Agent instantiation helpers
# ------------------------------------------------------------------

def _make_agent(policy: str, **kwargs: Any) -> BaseAgent:
    """Create an agent from a policy name string."""
    return create_agent(policy, **kwargs)


def _agent_from_spec(spec: OpponentSpec, registry: LeagueRegistry) -> BaseAgent:
    """Create an agent instance from an OpponentSpec."""
    if spec.source == "baseline":
        return _make_agent(spec.policy)
    if spec.source == "league":
        member_dir = registry.load_member(spec.policy)
        return _make_agent("league_snapshot", member_dir=member_dir)
    if spec.source == "fixed":
        return _make_agent(spec.policy)
    raise ValueError(f"Unknown opponent source: {spec.source!r}")


# ------------------------------------------------------------------
# Single episode
# ------------------------------------------------------------------

def run_episode(
    env: MixedEnvironment,
    eval_agent: BaseAgent,
    opponent_agent: BaseAgent,
    episode_idx: int,
    seed: int,
    opponent_spec: OpponentSpec,
) -> EpisodeResult:
    """Run one episode: agent_0 is the evaluated agent, rest are opponents."""
    obs = env.reset(seed=seed)
    agents_ids = list(obs.keys())

    eval_id = agents_ids[0]  # agent_0
    opponent_ids = agents_ids[1:]

    eval_agent.reset(eval_id, seed)
    for oid in opponent_ids:
        opponent_agent.reset(oid, derive_seed(seed, int(oid.split("_")[1])))

    total_return = 0.0
    step = 0

    while not env.is_done():
        actions: dict[str, Any] = {}
        # Evaluated agent
        if eval_id in obs:
            actions[eval_id] = eval_agent.act(obs[eval_id])
        # Opponents
        for oid in opponent_ids:
            if oid in obs:
                actions[oid] = opponent_agent.act(obs[oid])

        results = env.step(actions)

        # Update obs for next step
        obs = {aid: sr.observation for aid, sr in results.items()}

        if eval_id in results:
            total_return += results[eval_id].reward

        step += 1

    reason = env.termination_reason()
    return EpisodeResult(
        episode=episode_idx,
        opponent_spec=opponent_spec,
        return_total=total_return,
        final_shared_pool=env._state.shared_pool if env._state else 0.0,
        termination_reason=reason.value if reason else None,
        episode_length=step,
    )


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------

def evaluate(
    agent_policy: str,
    num_episodes: int = 20,
    seed: int = 42,
    config_id: str = "default",
    league_root: str = "storage/agents/league",
    agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run population evaluation and return a summary dict."""
    config = default_config(seed=seed)
    registry = LeagueRegistry(league_root)

    sampler = OpponentSampler(
        registry,
        seed=seed,
        weights=SamplingWeights(),
        include_fixed=["ppo_shared"],
    )

    eval_agent = _make_agent(agent_policy, **(agent_kwargs or {}))

    results: list[EpisodeResult] = []
    for ep in range(num_episodes):
        ep_seed = derive_seed(seed, ep)
        spec = sampler.sample_opponent_policy()

        try:
            opponent = _agent_from_spec(spec, registry)
        except (FileNotFoundError, KeyError):
            # Fallback to a baseline if the sampled spec can't be loaded
            spec = OpponentSpec(source="baseline", policy="random")
            opponent = _make_agent("random")

        env = MixedEnvironment(config)
        er = run_episode(env, eval_agent, opponent, ep, ep_seed, spec)
        results.append(er)

    return _build_summary(results, agent_policy)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def _build_summary(
    results: list[EpisodeResult], agent_policy: str
) -> dict[str, Any]:
    returns = np.array([r.return_total for r in results])
    pools = np.array([r.final_shared_pool for r in results])
    lengths = np.array([r.episode_length for r in results])
    collapses = sum(1 for r in results if r.termination_reason == "system_collapse")

    # Per-opponent breakdown
    by_opponent: dict[str, list[float]] = defaultdict(list)
    opponent_counts: dict[str, int] = defaultdict(int)
    for r in results:
        key = f"{r.opponent_spec.source}:{r.opponent_spec.policy}"
        by_opponent[key].append(r.return_total)
        opponent_counts[key] += 1

    opponent_breakdown = {
        k: {
            "count": opponent_counts[k],
            "mean_return": float(np.mean(v)),
        }
        for k, v in by_opponent.items()
    }

    return {
        "agent_policy": agent_policy,
        "num_episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_final_pool": float(np.mean(pools)),
        "collapse_rate": collapses / len(results) if results else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
        "opponent_breakdown": opponent_breakdown,
    }


# ------------------------------------------------------------------
# Pretty printing
# ------------------------------------------------------------------

def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"  Population Evaluation: {summary['agent_policy']}")
    print("=" * 60)
    print(f"  Episodes          : {summary['num_episodes']}")
    print(f"  Mean return       : {summary['mean_return']:.3f}")
    print(f"  Std return        : {summary['std_return']:.3f}")
    print(f"  Mean final pool   : {summary['mean_final_pool']:.3f}")
    print(f"  Collapse rate     : {summary['collapse_rate']:.1%}")
    print(f"  Mean episode len  : {summary['mean_episode_length']:.1f}")
    print("-" * 60)
    print("  Opponent breakdown:")
    for key, info in summary["opponent_breakdown"].items():
        print(f"    {key:30s}  n={info['count']:3d}  mean_ret={info['mean_return']:.3f}")
    print("=" * 60 + "\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent against sampled population opponents."
    )
    parser.add_argument(
        "--config-id",
        default="default",
        help="Config identifier (currently only 'default' is supported).",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--agent-policy",
        default="tit_for_tat",
        help="Policy for the evaluated agent (e.g. tit_for_tat, random, ppo_shared).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Root seed for reproducibility."
    )
    parser.add_argument(
        "--league-root",
        default="storage/agents/league",
        help="Path to the league snapshot directory.",
    )
    args = parser.parse_args(argv)

    summary = evaluate(
        agent_policy=args.agent_policy,
        num_episodes=args.episodes,
        seed=args.seed,
        config_id=args.config_id,
        league_root=args.league_root,
    )
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
