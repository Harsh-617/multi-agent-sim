"""Deterministic k-means clustering for competitive strategies using numpy only."""

from __future__ import annotations

from typing import Dict

import numpy as np

from simulation.analysis.competitive_strategy_features import PolicyFeatures

SEED = 0
DEFAULT_K = 3


def cluster_competitive_strategies(
    features: PolicyFeatures,
    n_clusters: int = DEFAULT_K,
) -> Dict[str, int]:
    """Assign each policy a cluster_id via k-means (numpy, fixed seed).

    ``k = min(n_clusters, n_policies)``.  None feature values are replaced
    with 0.  Returns ``{policy_name: cluster_id}``.
    """
    names = sorted(features.keys())
    if not names:
        return {}

    feature_keys = [
        "mean_reward",
        "robustness_score",
        "winner_rate",
        "worst_case_reward",
        "reward_variance",
        "best_sweep_score",
        "worst_sweep_score",
    ]

    # Build feature matrix, replacing None with 0.
    matrix = np.array(
        [[features[n].get(k) or 0.0 for k in feature_keys] for n in names],
        dtype=np.float64,
    )

    n = len(names)
    k = min(n_clusters, n)

    # Normalise columns to [0, 1] for balanced distance computation.
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    rng = col_max - col_min
    rng[rng == 0] = 1.0
    normed = (matrix - col_min) / rng

    # K-means with deterministic init (random state, seed=0).
    rng_np = np.random.RandomState(SEED)
    indices = rng_np.choice(n, size=k, replace=False)
    centroids = normed[indices].copy()

    for _ in range(100):
        dists = np.linalg.norm(
            normed[:, None, :] - centroids[None, :, :], axis=2
        )
        labels = dists.argmin(axis=1)
        new_centroids = np.empty_like(centroids)
        for c in range(k):
            members = normed[labels == c]
            if len(members) == 0:
                new_centroids[c] = centroids[c]
            else:
                new_centroids[c] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return {name: int(labels[i]) for i, name in enumerate(names)}
