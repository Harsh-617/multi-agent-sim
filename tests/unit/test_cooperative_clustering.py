"""Unit tests for cooperative agent strategy clustering."""

from __future__ import annotations

from simulation.analysis.cooperative_clustering import (
    COOPERATIVE_FEATURE_KEYS,
    MAX_K,
    cluster_cooperative_agents,
    label_cooperative_clusters,
    build_cooperative_feature_vector,
    _assign_cooperative_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(n: int, seed: int = 0) -> dict[str, dict[str, float]]:
    """Generate n synthetic agent feature dicts with varied values."""
    import random

    rng = random.Random(seed)
    features: dict[str, dict[str, float]] = {}
    for i in range(n):
        name = f"agent_{i:03d}"
        eu = rng.random()
        ir = rng.random() * 0.5
        features[name] = {
            "effort_utilization": eu,
            "idle_rate": ir,
            "dominant_type_fraction": rng.random(),
            "final_specialization_score": rng.random(),
            "role_stability": rng.random(),
            "mean_reward_per_step": rng.uniform(0.0, 1.0),
        }
    return features


def _all_label_types() -> set[str]:
    return {
        "Dedicated Specialist",
        "Adaptive Generalist",
        "Free Rider",
        "Opportunist",
        "Overcontributor",
    }


# ---------------------------------------------------------------------------
# cluster_cooperative_agents
# ---------------------------------------------------------------------------


class TestClusterCooperativeAgents:
    def test_runs_without_error_on_valid_feature_matrix(self):
        features = _make_features(10)
        result = cluster_cooperative_agents(features)
        assert isinstance(result, dict)
        assert len(result) == 10

    def test_empty_features_returns_empty(self):
        assert cluster_cooperative_agents({}) == {}

    def test_single_agent_returns_cluster_zero(self):
        features = {"agent_0": {k: 0.5 for k in COOPERATIVE_FEATURE_KEYS}}
        result = cluster_cooperative_agents(features)
        assert result == {"agent_0": 0}

    def test_all_agents_are_assigned_a_cluster(self):
        features = _make_features(12)
        result = cluster_cooperative_agents(features)
        assert set(result.keys()) == set(features.keys())

    def test_cluster_ids_are_non_negative_integers(self):
        features = _make_features(8)
        result = cluster_cooperative_agents(features)
        for cid in result.values():
            assert isinstance(cid, int)
            assert cid >= 0

    def test_cluster_id_within_k_range(self):
        features = _make_features(20)
        result = cluster_cooperative_agents(features)
        k = min(MAX_K, len(features))
        for cid in result.values():
            assert cid < k

    def test_n_clusters_override(self):
        features = _make_features(10)
        result = cluster_cooperative_agents(features, n_clusters=2)
        unique_clusters = set(result.values())
        assert len(unique_clusters) <= 2

    def test_deterministic_given_same_seed(self):
        features = _make_features(15)
        r1 = cluster_cooperative_agents(features, seed=0)
        r2 = cluster_cooperative_agents(features, seed=0)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """With enough agents, different seeds can produce different assignments."""
        features = _make_features(20, seed=42)
        r1 = cluster_cooperative_agents(features, seed=0)
        r2 = cluster_cooperative_agents(features, seed=99)
        # They may or may not differ; just verify both are valid
        assert len(r1) == len(r2) == 20

    def test_feature_vector_has_correct_dimensions(self):
        """The feature matrix has exactly 6 columns."""
        assert len(COOPERATIVE_FEATURE_KEYS) == 6

    def test_none_values_treated_as_zero(self):
        """Agents with None feature values should not crash clustering."""
        features = {
            "agent_0": {k: None for k in COOPERATIVE_FEATURE_KEYS},
            "agent_1": {k: 0.5 for k in COOPERATIVE_FEATURE_KEYS},
            "agent_2": {k: 1.0 for k in COOPERATIVE_FEATURE_KEYS},
        }
        result = cluster_cooperative_agents(features)
        assert len(result) == 3

    def test_k_capped_at_max_k(self):
        """k = min(MAX_K, n_agents) — never exceeds MAX_K."""
        features = _make_features(100)
        result = cluster_cooperative_agents(features)
        unique_clusters = set(result.values())
        assert len(unique_clusters) <= MAX_K


# ---------------------------------------------------------------------------
# label_cooperative_clusters
# ---------------------------------------------------------------------------


class TestLabelCooperativeClusters:
    def test_labels_assigned_to_all_clusters(self):
        features = _make_features(10)
        clusters = cluster_cooperative_agents(features)
        labels = label_cooperative_clusters(clusters, features)

        cluster_ids = set(clusters.values())
        assert set(labels.keys()) == cluster_ids

    def test_labels_are_valid_strategy_names(self):
        valid_labels = _all_label_types()
        features = _make_features(15)
        clusters = cluster_cooperative_agents(features)
        labels = label_cooperative_clusters(clusters, features)

        for cid, label in labels.items():
            assert label in valid_labels, f"Unknown label: {label!r}"

    def test_no_duplicate_labels(self):
        """Each cluster gets a unique label (when enough agents)."""
        features = _make_features(25)
        clusters = cluster_cooperative_agents(features)
        labels = label_cooperative_clusters(clusters, features)

        label_values = list(labels.values())
        assert len(label_values) == len(set(label_values)), (
            f"Duplicate labels found: {label_values}"
        )

    def test_empty_clusters_returns_empty(self):
        result = label_cooperative_clusters({}, {})
        assert result == {}

    def test_single_cluster_gets_label(self):
        features = {"agent_0": {k: 0.5 for k in COOPERATIVE_FEATURE_KEYS}}
        clusters = {"agent_0": 0}
        labels = label_cooperative_clusters(clusters, features)
        assert 0 in labels
        assert labels[0] in _all_label_types()

    def test_deterministic(self):
        """Same inputs → same label output."""
        features = _make_features(12)
        clusters = cluster_cooperative_agents(features, seed=0)
        l1 = label_cooperative_clusters(clusters, features)
        l2 = label_cooperative_clusters(clusters, features)
        assert l1 == l2


# ---------------------------------------------------------------------------
# _assign_cooperative_label (rule-based labeling)
# ---------------------------------------------------------------------------


class TestAssignCooperativeLabel:
    def test_free_rider_when_high_idle(self):
        """High idle_rate triggers Free Rider label."""
        mean = {
            "idle_rate": 0.5,
            "effort_utilization": 0.2,
            "dominant_type_fraction": 0.3,
            "role_stability": 0.3,
        }
        label = _assign_cooperative_label(mean, used=set())
        assert label == "Free Rider"

    def test_dedicated_specialist_when_high_dom_frac_and_role_stability(self):
        mean = {
            "idle_rate": 0.05,
            "effort_utilization": 0.6,
            "dominant_type_fraction": 0.8,
            "role_stability": 0.7,
        }
        label = _assign_cooperative_label(mean, used=set())
        # Low idle means not a Free Rider
        # High dom_frac + role_stability → Dedicated Specialist
        assert label == "Dedicated Specialist"

    def test_overcontributor_when_high_effort_and_low_idle(self):
        mean = {
            "idle_rate": 0.05,
            "effort_utilization": 0.8,
            "dominant_type_fraction": 0.4,
            "role_stability": 0.4,
        }
        label = _assign_cooperative_label(mean, used=set())
        assert label == "Overcontributor"

    def test_adaptive_generalist_when_low_dom_frac_and_low_idle(self):
        mean = {
            "idle_rate": 0.1,
            "effort_utilization": 0.5,
            "dominant_type_fraction": 0.3,
            "role_stability": 0.4,
        }
        label = _assign_cooperative_label(mean, used=set())
        assert label == "Adaptive Generalist"

    def test_used_labels_are_skipped(self):
        """Labels already in 'used' set are not assigned again."""
        mean = {
            "idle_rate": 0.5,
            "effort_utilization": 0.2,
            "dominant_type_fraction": 0.3,
            "role_stability": 0.3,
        }
        # Free Rider would normally be chosen, but it's already used
        label = _assign_cooperative_label(mean, used={"Free Rider"})
        assert label != "Free Rider"

    def test_fallback_to_opportunist(self):
        """When all primary candidates exhausted, falls back."""
        mean = {
            "idle_rate": 0.0,
            "effort_utilization": 0.5,
            "dominant_type_fraction": 0.6,
            "role_stability": 0.3,
        }
        # Eliminate all clear candidates
        used = {"Free Rider", "Dedicated Specialist", "Overcontributor", "Adaptive Generalist"}
        label = _assign_cooperative_label(mean, used=used)
        assert label == "Opportunist"


# ---------------------------------------------------------------------------
# build_cooperative_feature_vector
# ---------------------------------------------------------------------------


class TestBuildCooperativeFeatureVector:
    def test_returns_dict_with_all_six_keys(self):
        agent_metrics = {
            "effort_utilization": 0.7,
            "idle_rate": 0.1,
            "dominant_type_fraction": 0.6,
            "final_specialization_score": 0.5,
            "role_stability": 0.8,
            "mean_reward_per_step": 0.3,
        }
        result = build_cooperative_feature_vector(agent_metrics)
        assert set(result.keys()) == set(COOPERATIVE_FEATURE_KEYS)

    def test_converts_values_to_float(self):
        agent_metrics = {k: 1 for k in COOPERATIVE_FEATURE_KEYS}  # int values
        result = build_cooperative_feature_vector(agent_metrics)
        for v in result.values():
            assert isinstance(v, float)

    def test_none_values_become_zero(self):
        agent_metrics = {k: None for k in COOPERATIVE_FEATURE_KEYS}
        result = build_cooperative_feature_vector(agent_metrics)
        for v in result.values():
            assert v == 0.0

    def test_missing_keys_become_zero(self):
        result = build_cooperative_feature_vector({})
        assert len(result) == 6
        for v in result.values():
            assert v == 0.0

    def test_passes_through_correct_values(self):
        agent_metrics = {
            "effort_utilization": 0.42,
            "idle_rate": 0.15,
            "dominant_type_fraction": 0.73,
            "final_specialization_score": 0.61,
            "role_stability": 0.88,
            "mean_reward_per_step": 0.34,
        }
        result = build_cooperative_feature_vector(agent_metrics)
        import pytest
        assert result["effort_utilization"] == pytest.approx(0.42)
        assert result["idle_rate"] == pytest.approx(0.15)
        assert result["dominant_type_fraction"] == pytest.approx(0.73)

    def test_output_is_compatible_with_cluster_input(self):
        """feature vector output can be fed directly to cluster_cooperative_agents."""
        agent_metrics_list = [
            {"effort_utilization": 0.7, "idle_rate": 0.1, "dominant_type_fraction": 0.8,
             "final_specialization_score": 0.6, "role_stability": 0.75, "mean_reward_per_step": 0.5},
            {"effort_utilization": 0.3, "idle_rate": 0.5, "dominant_type_fraction": 0.2,
             "final_specialization_score": 0.2, "role_stability": 0.3, "mean_reward_per_step": 0.1},
        ]
        features = {
            f"agent_{i}": build_cooperative_feature_vector(m)
            for i, m in enumerate(agent_metrics_list)
        }
        result = cluster_cooperative_agents(features)
        assert len(result) == 2
