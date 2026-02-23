"""Robustness evaluation: run policies across multiple environment variants."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from simulation.config.schema import MixedEnvironmentConfig

from .evaluator import PolicyResult, evaluate_policies
from .policy_set import PolicySpec
from .sweeps import SweepSpec, apply_sweep


@dataclass
class PolicyRobustness:
    """Aggregated robustness metrics for one policy across all sweeps."""

    policy_name: str
    overall_mean_reward: float = 0.0
    worst_case_mean_reward: float = 0.0
    robustness_score: float = 0.0
    collapse_rate_overall: float = 0.0
    n_sweeps_evaluated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "overall_mean_reward": self.overall_mean_reward,
            "worst_case_mean_reward": self.worst_case_mean_reward,
            "robustness_score": self.robustness_score,
            "collapse_rate_overall": self.collapse_rate_overall,
            "n_sweeps_evaluated": self.n_sweeps_evaluated,
        }


@dataclass
class RobustnessResult:
    """Complete robustness evaluation output."""

    metadata: dict[str, Any] = field(default_factory=dict)
    per_sweep_results: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    per_policy_robustness: dict[str, PolicyRobustness] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "per_sweep_results": self.per_sweep_results,
            "per_policy_robustness": {
                k: v.to_dict() for k, v in self.per_policy_robustness.items()
            },
        }


def evaluate_robustness(
    base_config: MixedEnvironmentConfig,
    policy_specs: list[PolicySpec],
    sweeps: list[SweepSpec],
    *,
    seeds: list[int],
    episodes_per_seed: int = 1,
    max_steps_override: int | None = None,
) -> RobustnessResult:
    """Evaluate policies across multiple environment variants.

    Parameters
    ----------
    base_config:
        Base environment config (not mutated).
    policy_specs:
        Policies to evaluate.
    sweeps:
        Environment variants to test.
    seeds:
        Root seeds for reproducibility.
    episodes_per_seed:
        Episodes per seed per policy per sweep.
    max_steps_override:
        If set, cap max_steps for faster runs.
    """
    result = RobustnessResult()

    # Metadata
    result.metadata = {
        "sweeps": [s.name for s in sweeps],
        "sweep_count": len(sweeps),
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "max_steps_override": max_steps_override,
        "policy_count": len(policy_specs),
    }

    # Track per-policy rewards across sweeps for aggregation
    policy_sweep_rewards: dict[str, list[float]] = {}
    policy_sweep_collapses: dict[str, list[float]] = {}

    for sweep in sweeps:
        swept_config = apply_sweep(base_config, sweep)

        sweep_results = evaluate_policies(
            swept_config,
            policy_specs,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_steps_override=max_steps_override,
        )

        sweep_data: dict[str, dict[str, Any]] = {}
        for pr in sweep_results:
            name = pr.spec.name
            sweep_data[name] = pr.to_dict()

            if pr.spec.available and pr.episodes:
                policy_sweep_rewards.setdefault(name, []).append(
                    pr.mean_total_reward
                )
                policy_sweep_collapses.setdefault(name, []).append(
                    pr.collapse_rate
                )

        result.per_sweep_results[sweep.name] = sweep_data

    # Aggregate per-policy robustness
    for spec in policy_specs:
        name = spec.name
        pr = PolicyRobustness(policy_name=name)

        rewards = policy_sweep_rewards.get(name, [])
        collapses = policy_sweep_collapses.get(name, [])

        if rewards:
            overall_mean = round(statistics.mean(rewards), 4)
            worst_case = round(min(rewards), 4)
            pr.overall_mean_reward = overall_mean
            pr.worst_case_mean_reward = worst_case
            pr.robustness_score = round(
                0.7 * overall_mean + 0.3 * worst_case, 4
            )
            pr.collapse_rate_overall = (
                round(statistics.mean(collapses), 4) if collapses else 0.0
            )
            pr.n_sweeps_evaluated = len(rewards)

        result.per_policy_robustness[name] = pr

    return result
