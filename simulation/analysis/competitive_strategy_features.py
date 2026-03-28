"""Extract per-policy behavioral features from competitive robustness results."""

from __future__ import annotations

from typing import Dict, Optional


Features = Dict[str, Optional[float]]
PolicyFeatures = Dict[str, Features]


def extract_competitive_strategy_features(
    policy_results: dict,
) -> PolicyFeatures:
    """Return per-policy feature dicts from competitive policy results.

    Parameters
    ----------
    policy_results:
        ``{policy_name: {mean_reward, robustness_score, winner_rate,
        worst_case_reward, sweep_rewards}}``.

    Missing fields are set to ``None``.  ``mean_reward`` and
    ``worst_case_reward`` are min-max normalised across policies.
    """
    if not policy_results:
        return {}

    # Collect raw values for normalisation.
    raw_mean = {
        name: data.get("mean_reward")
        for name, data in policy_results.items()
    }
    raw_worst = {
        name: data.get("worst_case_reward")
        for name, data in policy_results.items()
    }

    mean_vals = [v for v in raw_mean.values() if v is not None]
    worst_vals = [v for v in raw_worst.values() if v is not None]

    mean_min, mean_max = (
        (min(mean_vals), max(mean_vals)) if mean_vals else (0.0, 0.0)
    )
    worst_min, worst_max = (
        (min(worst_vals), max(worst_vals)) if worst_vals else (0.0, 0.0)
    )

    mean_range = mean_max - mean_min if mean_max != mean_min else 1.0
    worst_range = worst_max - worst_min if worst_max != worst_min else 1.0

    out: PolicyFeatures = {}
    for name, data in policy_results.items():
        mr = data.get("mean_reward")
        wc = data.get("worst_case_reward")
        sweeps = data.get("sweep_rewards")

        norm_mean: Optional[float] = None
        if mr is not None:
            norm_mean = (mr - mean_min) / mean_range

        norm_worst: Optional[float] = None
        if wc is not None:
            norm_worst = (wc - worst_min) / worst_range

        reward_variance: Optional[float] = None
        best_sweep: Optional[float] = None
        worst_sweep: Optional[float] = None
        if sweeps and len(sweeps) > 0:
            s_mean = sum(sweeps) / len(sweeps)
            reward_variance = sum((x - s_mean) ** 2 for x in sweeps) / len(sweeps)
            best_sweep = max(sweeps)
            worst_sweep = min(sweeps)

        out[name] = {
            "mean_reward": norm_mean,
            "robustness_score": data.get("robustness_score"),
            "winner_rate": data.get("winner_rate"),
            "worst_case_reward": norm_worst,
            "reward_variance": reward_variance,
            "best_sweep_score": best_sweep,
            "worst_sweep_score": worst_sweep,
        }

    return out
