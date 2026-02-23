"""Rule-based labeling and summary generation for strategy clusters."""

from __future__ import annotations

from typing import Dict, Optional

from simulation.analysis.strategy_features import PolicyFeatures

LABELS = ("Exploitative", "Cooperative", "Robust", "Unstable")

SUMMARIES = {
    "Exploitative": "High individual reward but contributes to system collapse.",
    "Cooperative": "Sustains the shared pool while maintaining moderate returns.",
    "Robust": "Performs well under worst-case conditions with low variance.",
    "Unstable": "No dominant behavioural pattern; performance varies widely.",
}


def _cluster_stats(
    cluster_id: int,
    clusters: Dict[str, int],
    features: PolicyFeatures,
) -> Dict[str, Optional[float]]:
    """Aggregate feature means for a single cluster."""
    members = [p for p, c in clusters.items() if c == cluster_id]
    if not members:
        return {}
    keys = ["mean_return", "worst_case_return", "collapse_rate",
            "mean_final_pool", "robustness_score"]
    stats: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [features[p][k] for p in members if features[p].get(k) is not None]
        stats[k] = sum(vals) / len(vals) if vals else None
    return stats


def label_clusters(
    clusters: Dict[str, int],
    features: PolicyFeatures,
) -> Dict[int, str]:
    """Assign a deterministic label to each cluster based on aggregate features.

    Rules (applied in order, first match wins):
    1. **Exploitative** – high mean_return *and* high collapse_rate.
    2. **Cooperative** – high mean_final_pool *and* low collapse_rate.
    3. **Robust** – high worst_case_return *and* low collapse_rate.
    4. **Unstable** – everything else.
    """
    if not clusters:
        return {}

    unique_ids = sorted(set(clusters.values()))

    # Gather per-cluster stats.
    all_stats = {cid: _cluster_stats(cid, clusters, features) for cid in unique_ids}

    # Compute global medians for thresholding.
    all_returns = [s["mean_return"] for s in all_stats.values()
                   if s.get("mean_return") is not None]
    all_collapse = [s["collapse_rate"] for s in all_stats.values()
                    if s.get("collapse_rate") is not None]
    all_pool = [s["mean_final_pool"] for s in all_stats.values()
                if s.get("mean_final_pool") is not None]
    all_worst = [s["worst_case_return"] for s in all_stats.values()
                 if s.get("worst_case_return") is not None]

    med_return = _median(all_returns)
    med_collapse = _median(all_collapse)
    med_pool = _median(all_pool)
    med_worst = _median(all_worst)

    labels: Dict[int, str] = {}
    for cid in unique_ids:
        s = all_stats[cid]
        mr = s.get("mean_return")
        cr = s.get("collapse_rate")
        mp = s.get("mean_final_pool")
        wc = s.get("worst_case_return")

        if (mr is not None and mr >= med_return
                and cr is not None and cr > med_collapse):
            labels[cid] = "Exploitative"
        elif (mp is not None and mp >= med_pool
              and cr is not None and cr <= med_collapse):
            labels[cid] = "Cooperative"
        elif (wc is not None and wc >= med_worst
              and cr is not None and cr <= med_collapse):
            labels[cid] = "Robust"
        else:
            labels[cid] = "Unstable"

    return labels


def cluster_summaries(labels: Dict[int, str]) -> Dict[int, str]:
    """Return a short text summary for each cluster based on its label."""
    return {cid: SUMMARIES[label] for cid, label in labels.items()}


def _median(values: list[float]) -> float:
    """Return the median; 0.0 for empty lists."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
