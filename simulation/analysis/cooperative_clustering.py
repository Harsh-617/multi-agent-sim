"""K-means strategy clustering for cooperative agents.

Feature vector (6 features, from Part 10):
  effort_utilization, idle_rate, dominant_type_fraction,
  final_specialization_score, role_stability, mean_reward_per_step

Expected emergent labels (NOT hardcoded — emerge from clustering):
  Dedicated Specialist, Adaptive Generalist, Free Rider,
  Opportunist, Overcontributor

Mirrors simulation/analysis/strategy_clustering.py — adapted feature vector.
Does NOT modify strategy_clustering.py (ADR-013).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SEED = 0
MAX_K = 5  # 5 emergent strategy types for cooperative

# Feature keys in canonical order (must match what callers supply)
COOPERATIVE_FEATURE_KEYS = [
    "effort_utilization",
    "idle_rate",
    "dominant_type_fraction",
    "final_specialization_score",
    "role_stability",
    "mean_reward_per_step",
]

# Cooperative strategy label map (cluster centroid interpretation rules)
# Labels are determined by centroid position — they are NOT hardcoded to clusters.
_LABEL_RULES: list[tuple[str, dict[str, float]]] = [
    # (label, {feature: threshold_direction})
    # Thresholds are soft — interpreted at label-time from centroid values.
    ("Dedicated Specialist", {"dominant_type_fraction": 0.7, "role_stability": 0.6}),
    ("Adaptive Generalist", {"dominant_type_fraction": 0.0, "idle_rate": 0.0}),
    ("Free Rider", {"idle_rate": 0.4}),
    ("Overcontributor", {"effort_utilization": 0.7, "idle_rate": 0.0}),
    ("Opportunist", {}),  # default / fallback
]


def cluster_cooperative_agents(
    features: dict[str, dict[str, Any]],
    n_clusters: int | None = None,
    seed: int = SEED,
) -> dict[str, int]:
    """Assign each agent a cluster_id via k-means (numpy, fixed seed).

    ``k = min(MAX_K, n_agents)`` unless overridden by n_clusters.
    None feature values are replaced with 0.

    Parameters
    ----------
    features:
        Dict mapping agent_id → dict with the 6 cooperative feature keys.
    n_clusters:
        Override k. Defaults to min(MAX_K, n_agents).
    seed:
        Random seed for determinism.

    Returns
    -------
    dict[str, int]
        Mapping agent_id → cluster_id.
    """
    names = sorted(features.keys())
    if not names:
        return {}

    # Build feature matrix (rows = agents, cols = features)
    matrix = np.array(
        [
            [float(features[n].get(k) or 0.0) for k in COOPERATIVE_FEATURE_KEYS]
            for n in names
        ],
        dtype=np.float64,
    )

    n = len(names)
    k = min(n_clusters if n_clusters is not None else MAX_K, n)

    # Normalise columns to [0, 1]
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    normed = (matrix - col_min) / col_range

    # K-means with deterministic init
    rng_np = np.random.RandomState(seed)
    indices = rng_np.choice(n, size=k, replace=False)
    centroids = normed[indices].copy()

    for _ in range(200):
        dists = np.linalg.norm(normed[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.empty_like(centroids)
        for c in range(k):
            cluster_members = normed[labels == c]
            if len(cluster_members) == 0:
                new_centroids[c] = centroids[c]
            else:
                new_centroids[c] = cluster_members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return {name: int(labels[i]) for i, name in enumerate(names)}


def label_cooperative_clusters(
    clusters: dict[str, int],
    features: dict[str, dict[str, Any]],
) -> dict[int, str]:
    """Assign strategy labels to cooperative cluster IDs based on centroid analysis.

    Labels emerge from feature distribution — they are not hardcoded to IDs.

    Parameters
    ----------
    clusters:
        Mapping agent_id → cluster_id (from cluster_cooperative_agents).
    features:
        Agent feature dicts (same input as cluster_cooperative_agents).

    Returns
    -------
    dict[int, str]
        Mapping cluster_id → label string.
    """
    cluster_ids = sorted(set(clusters.values()))
    if not cluster_ids:
        return {}

    # Compute per-cluster mean feature vectors
    cluster_means: dict[int, dict[str, float]] = {}
    for cid in cluster_ids:
        agent_ids = [aid for aid, c in clusters.items() if c == cid]
        for key in COOPERATIVE_FEATURE_KEYS:
            vals = [float(features[aid].get(key) or 0.0) for aid in agent_ids]
            cluster_means.setdefault(cid, {})[key] = float(np.mean(vals)) if vals else 0.0

    # Assign labels by dominant features (rule-based on centroids)
    label_map: dict[int, str] = {}
    used_labels: set[str] = set()

    for cid in cluster_ids:
        mean = cluster_means[cid]
        assigned = _assign_cooperative_label(mean, used_labels)
        label_map[cid] = assigned
        used_labels.add(assigned)

    return label_map


def _assign_cooperative_label(mean: dict[str, float], used: set[str]) -> str:
    """Assign the best-matching label to a cluster centroid.

    Priority order: Free Rider → Dedicated Specialist → Overcontributor
    → Adaptive Generalist → Opportunist.
    """
    idle = mean.get("idle_rate", 0.0)
    effort = mean.get("effort_utilization", 0.0)
    dom_frac = mean.get("dominant_type_fraction", 0.0)
    role_stab = mean.get("role_stability", 0.0)

    candidates: list[tuple[str, float]] = []

    if "Free Rider" not in used and idle >= 0.35:
        candidates.append(("Free Rider", idle))
    if "Dedicated Specialist" not in used and dom_frac >= 0.6 and role_stab >= 0.5:
        candidates.append(("Dedicated Specialist", dom_frac + role_stab))
    if "Overcontributor" not in used and effort >= 0.65 and idle <= 0.15:
        candidates.append(("Overcontributor", effort))
    if "Adaptive Generalist" not in used and dom_frac <= 0.55 and idle <= 0.25:
        candidates.append(("Adaptive Generalist", 1.0 - dom_frac))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # Fallback: pick first unused label from the complete list
    all_labels = [
        "Dedicated Specialist", "Adaptive Generalist",
        "Free Rider", "Opportunist", "Overcontributor",
    ]
    for label in all_labels:
        if label not in used:
            return label
    return "Opportunist"


def build_cooperative_feature_vector(
    agent_metrics: dict[str, Any],
) -> dict[str, float]:
    """Extract the 6-feature vector from an agent_metrics dict (from episode summary).

    This is a convenience helper for converting cooperative_collector output
    into the format expected by cluster_cooperative_agents().
    """
    return {
        "effort_utilization": float(agent_metrics.get("effort_utilization") or 0.0),
        "idle_rate": float(agent_metrics.get("idle_rate") or 0.0),
        "dominant_type_fraction": float(agent_metrics.get("dominant_type_fraction") or 0.0),
        "final_specialization_score": float(
            agent_metrics.get("final_specialization_score") or 0.0
        ),
        "role_stability": float(agent_metrics.get("role_stability") or 0.0),
        "mean_reward_per_step": float(agent_metrics.get("mean_reward_per_step") or 0.0),
    }
