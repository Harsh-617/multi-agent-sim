"""Unit tests for strategy feature extraction, clustering, and labeling."""

from __future__ import annotations

from simulation.analysis.strategy_features import extract_features
from simulation.analysis.strategy_clustering import cluster_policies
from simulation.analysis.strategy_labels import label_clusters, cluster_summaries, SUMMARIES


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _eval_report() -> dict:
    """Minimal eval report with three policies."""
    return {
        "kind": "eval",
        "results": [
            {
                "policy_name": "greedy",
                "available": True,
                "mean_total_reward": 90.0,
                "std_total_reward": 5.0,
                "mean_final_shared_pool": 10.0,
                "collapse_rate": 0.8,
                "mean_episode_length": 50,
                "n_episodes": 10,
            },
            {
                "policy_name": "cooperative",
                "available": True,
                "mean_total_reward": 50.0,
                "std_total_reward": 2.0,
                "mean_final_shared_pool": 80.0,
                "collapse_rate": 0.05,
                "mean_episode_length": 100,
                "n_episodes": 10,
            },
            {
                "policy_name": "random",
                "available": True,
                "mean_total_reward": 30.0,
                "std_total_reward": 15.0,
                "mean_final_shared_pool": 40.0,
                "collapse_rate": 0.5,
                "mean_episode_length": 60,
                "n_episodes": 10,
            },
        ],
    }


def _robust_report() -> dict:
    """Minimal robustness report with two policies."""
    return {
        "kind": "robust",
        "per_policy_robustness": {
            "ppo_best": {
                "policy_name": "ppo_best",
                "overall_mean_reward": 70.0,
                "worst_case_mean_reward": 55.0,
                "robustness_score": 65.5,
                "collapse_rate_overall": 0.1,
                "n_sweeps_evaluated": 4,
            },
            "random": {
                "policy_name": "random",
                "overall_mean_reward": 25.0,
                "worst_case_mean_reward": 10.0,
                "robustness_score": 20.5,
                "collapse_rate_overall": 0.6,
                "n_sweeps_evaluated": 4,
            },
        },
        "per_sweep_results": {
            "sweep_a": {
                "ppo_best": {"mean_final_shared_pool": 60.0},
                "random": {"mean_final_shared_pool": 30.0},
            },
            "sweep_b": {
                "ppo_best": {"mean_final_shared_pool": 80.0},
                "random": {"mean_final_shared_pool": 20.0},
            },
        },
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_eval_basic_fields(self):
        feats = extract_features(_eval_report())
        assert set(feats.keys()) == {"greedy", "cooperative", "random"}
        assert feats["greedy"]["mean_return"] == 90.0
        assert feats["greedy"]["collapse_rate"] == 0.8
        assert feats["greedy"]["mean_final_pool"] == 10.0
        # Eval reports have no worst-case or robustness
        assert feats["greedy"]["worst_case_return"] is None
        assert feats["greedy"]["robustness_score"] is None

    def test_robust_basic_fields(self):
        feats = extract_features(_robust_report())
        assert set(feats.keys()) == {"ppo_best", "random"}
        assert feats["ppo_best"]["mean_return"] == 70.0
        assert feats["ppo_best"]["worst_case_return"] == 55.0
        assert feats["ppo_best"]["robustness_score"] == 65.5
        assert feats["ppo_best"]["collapse_rate"] == 0.1

    def test_robust_pool_from_sweeps(self):
        feats = extract_features(_robust_report())
        # (60 + 80) / 2 = 70
        assert feats["ppo_best"]["mean_final_pool"] == 70.0
        # (30 + 20) / 2 = 25
        assert feats["random"]["mean_final_pool"] == 25.0

    def test_unavailable_policies_skipped(self):
        report = _eval_report()
        report["results"][0]["available"] = False
        feats = extract_features(report)
        assert "greedy" not in feats

    def test_empty_report(self):
        assert extract_features({"results": []}) == {}

    def test_missing_fields_become_none(self):
        report = {
            "kind": "eval",
            "results": [{"policy_name": "bare", "available": True}],
        }
        feats = extract_features(report)
        assert feats["bare"]["mean_return"] is None
        assert feats["bare"]["collapse_rate"] is None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class TestClustering:
    def test_deterministic(self):
        """Running clustering twice produces identical results."""
        feats = extract_features(_eval_report())
        c1 = cluster_policies(feats)
        c2 = cluster_policies(feats)
        assert c1 == c2

    def test_cluster_ids_within_range(self):
        feats = extract_features(_eval_report())
        clusters = cluster_policies(feats)
        k = min(4, len(feats))
        for cid in clusters.values():
            assert 0 <= cid < k

    def test_all_policies_assigned(self):
        feats = extract_features(_eval_report())
        clusters = cluster_policies(feats)
        assert set(clusters.keys()) == set(feats.keys())

    def test_empty_features(self):
        assert cluster_policies({}) == {}

    def test_single_policy(self):
        feats = {"only": {"mean_return": 1.0, "worst_case_return": None,
                          "collapse_rate": 0.0, "mean_final_pool": 5.0,
                          "robustness_score": None}}
        clusters = cluster_policies(feats)
        assert clusters == {"only": 0}


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

class TestLabeling:
    def test_labels_are_valid(self):
        feats = extract_features(_eval_report())
        clusters = cluster_policies(feats)
        labels = label_clusters(clusters, feats)
        for label in labels.values():
            assert label in ("Exploitative", "Cooperative", "Robust", "Unstable")

    def test_deterministic(self):
        feats = extract_features(_eval_report())
        clusters = cluster_policies(feats)
        l1 = label_clusters(clusters, feats)
        l2 = label_clusters(clusters, feats)
        assert l1 == l2

    def test_summaries_match_labels(self):
        feats = extract_features(_eval_report())
        clusters = cluster_policies(feats)
        labels = label_clusters(clusters, feats)
        summaries = cluster_summaries(labels)
        for cid, label in labels.items():
            assert cid in summaries
            assert summaries[cid] == SUMMARIES[label]

    def test_empty_clusters(self):
        assert label_clusters({}, {}) == {}
