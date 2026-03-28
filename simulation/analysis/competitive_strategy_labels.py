"""Rule-based labeling for competitive strategy clusters."""

from __future__ import annotations

from typing import Dict, Optional

from simulation.analysis.competitive_strategy_features import PolicyFeatures

LABELS = ("Dominant", "Aggressive", "Consistent", "Weak")

SUMMARIES = {
    "Dominant": "High mean reward and high winner rate across sweeps.",
    "Aggressive": "High winner rate but poor robustness under perturbation.",
    "Consistent": "Strong robustness score with low reward variance.",
    "Weak": "Low mean reward and low winner rate overall.",
}


def get_competitive_strategy_label(cluster_features: PolicyFeatures) -> str:
    """Assign a deterministic label to a cluster based on aggregate features.

    Parameters
    ----------
    cluster_features:
        ``{policy_name: feature_dict}`` for the policies in one cluster.

    Rules (applied in order, first match wins):

    1. **Dominant** – high mean_reward *and* high winner_rate.
    2. **Aggressive** – high winner_rate but low robustness_score.
    3. **Consistent** – high robustness_score *and* low reward_variance.
    4. **Weak** – everything else.
    """
    if not cluster_features:
        return "Weak"

    stats = _aggregate(cluster_features)

    mr = stats.get("mean_reward")
    wr = stats.get("winner_rate")
    rs = stats.get("robustness_score")
    rv = stats.get("reward_variance")

    # Thresholds — use 0.5 as midpoint for normalised / bounded features.
    high_mr = mr is not None and mr >= 0.5
    high_wr = wr is not None and wr >= 0.5
    high_rs = rs is not None and rs >= 0.5
    low_rs = rs is not None and rs < 0.5
    low_rv = rv is not None and rv < _median_val(cluster_features, "reward_variance")

    if high_mr and high_wr:
        return "Dominant"
    if high_wr and low_rs:
        return "Aggressive"
    if high_rs and low_rv:
        return "Consistent"
    return "Weak"


def competitive_cluster_summaries(labels: Dict[int, str]) -> Dict[int, str]:
    """Return a short text summary for each cluster based on its label."""
    return {cid: SUMMARIES[label] for cid, label in labels.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate(
    cluster_features: PolicyFeatures,
) -> Dict[str, Optional[float]]:
    """Compute mean of each feature across cluster members."""
    keys = [
        "mean_reward",
        "robustness_score",
        "winner_rate",
        "worst_case_reward",
        "reward_variance",
        "best_sweep_score",
        "worst_sweep_score",
    ]
    stats: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [
            feat[k]
            for feat in cluster_features.values()
            if feat.get(k) is not None
        ]
        stats[k] = sum(vals) / len(vals) if vals else None
    return stats


def _median_val(cluster_features: PolicyFeatures, key: str) -> float:
    """Return the median of *key* across cluster members; 0.0 if empty."""
    vals = [
        feat[key]
        for feat in cluster_features.values()
        if feat.get(key) is not None
    ]
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
