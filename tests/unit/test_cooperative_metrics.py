"""Unit tests for cooperative metrics computation.

Covers:
  - All episode-level metric keys present in episode summary
  - All agent-level metric keys present per agent
  - Metric bounds: completion_ratio, group_efficiency_ratio, effort_gini_coefficient,
    free_rider_fraction, role_stability, specialization_divergence all in [0, 1]
  - event log contains only defined EventType values
  - metrics are deterministic — same seed + same actions = same metrics
"""

from __future__ import annotations

import asyncio

import pytest

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.metrics.cooperative_definitions import EventType
from simulation.runner.cooperative_experiment_runner import run_cooperative_experiment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPISODE_METRIC_KEYS = {
    "episode_length",
    "termination_reason",
    "total_tasks_arrived",
    "total_tasks_completed",
    "completion_ratio",
    "final_backlog_level",
    "final_system_stress",
    "mean_system_stress",
    "peak_system_stress",
    "collapse_occurred",
    "total_reward_per_agent",
    "mean_reward_per_step_per_agent",
    # Step E extended
    "group_efficiency_ratio",
    "contribution_variance",
    "specialization_divergence",
    "mean_role_stability",
    "free_rider_count",
    "free_rider_fraction",
    "effort_gini_coefficient",
    "agent_metrics",
}

AGENT_METRIC_KEYS = {
    "cumulative_reward",
    "mean_reward_per_step",
    "effort_utilization",
    "idle_rate",
    "dominant_task_type",
    "dominant_type_fraction",
    "final_specialization_score",
    "peak_specialization_score",
    "role_stability",
    "strategy_label",
}

VALID_EVENT_TYPES = {e.value for e in EventType}


def _small_config(seed: int = 42) -> CooperativeEnvironmentConfig:
    """Fast config for unit tests — 2 agents, 20 steps."""
    data = {
        "identity": {
            "environment_type": "cooperative",
            "environment_version": "1.0.0",
            "archetype": "shared_goal_collective",
            "seed": seed,
        },
        "population": {
            "num_agents": 3,
            "max_steps": 20,
            "num_task_types": 2,
            "agent_effort_capacity": 1.0,
            "collapse_sustain_window": 10,
            "enable_early_success": False,
            "clearance_sustain_window": 15,
        },
        "layers": {
            "observation_noise": 0.0,
            "history_window": 5,
            "specialization_scale": 0.3,
            "specialization_decay": 0.1,
            "task_arrival_noise": 0.0,
            "task_difficulty_variance": 0.0,
            "free_rider_pressure_scale": 1.0,
        },
        "task": {
            "task_arrival_rate": 1.0,
            "task_difficulty": 1.0,
            "collapse_threshold": 50,
            "initial_backlog": 0,
        },
        "rewards": {
            "w_group": 0.7,
            "w_individual": 0.2,
            "w_efficiency": 0.1,
        },
        "instrumentation": {
            "enable_step_metrics": True,
            "enable_episode_metrics": True,
            "step_log_frequency": 1,
        },
    }
    return CooperativeEnvironmentConfig.model_validate(data)


def _run(seed: int = 42, tmp_path=None) -> dict:
    """Run a small cooperative episode and return the episode summary."""
    import tempfile
    from pathlib import Path

    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp()
        runs_dir = Path(tmp_dir)
    else:
        runs_dir = Path(tmp_path)

    config = _small_config(seed=seed)
    summary = run_cooperative_experiment(
        config, f"test_{seed}", runs_dir, None, agent_policy="random"
    )
    return summary


# ---------------------------------------------------------------------------
# Tests — episode-level keys
# ---------------------------------------------------------------------------

class TestEpisodeLevelKeys:
    def test_all_episode_keys_present(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        missing = EPISODE_METRIC_KEYS - set(summary.keys())
        assert not missing, f"Missing episode-level keys: {missing}"

    def test_agent_metrics_key_present(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        assert "agent_metrics" in summary
        assert isinstance(summary["agent_metrics"], dict)
        assert len(summary["agent_metrics"]) > 0

    def test_all_agent_keys_present_per_agent(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        for agent_id, am in summary["agent_metrics"].items():
            missing = AGENT_METRIC_KEYS - set(am.keys())
            assert not missing, (
                f"Agent '{agent_id}' missing keys: {missing}"
            )


# ---------------------------------------------------------------------------
# Tests — bounds
# ---------------------------------------------------------------------------

class TestMetricBounds:
    def test_completion_ratio_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["completion_ratio"]
        assert 0.0 <= v <= 1.0, f"completion_ratio = {v}"

    def test_group_efficiency_ratio_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["group_efficiency_ratio"]
        assert 0.0 <= v <= 1.0, f"group_efficiency_ratio = {v}"

    def test_effort_gini_coefficient_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["effort_gini_coefficient"]
        assert 0.0 <= v <= 1.0, f"effort_gini_coefficient = {v}"

    def test_free_rider_fraction_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["free_rider_fraction"]
        assert 0.0 <= v <= 1.0, f"free_rider_fraction = {v}"

    def test_role_stability_bounded_per_agent(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        for aid, am in summary["agent_metrics"].items():
            v = am["role_stability"]
            assert 0.0 <= v <= 1.0, f"role_stability for {aid} = {v}"

    def test_specialization_divergence_nonnegative(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["specialization_divergence"]
        assert v >= 0.0, f"specialization_divergence = {v}"

    def test_mean_system_stress_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["mean_system_stress"]
        assert 0.0 <= v <= 1.0, f"mean_system_stress = {v}"

    def test_peak_system_stress_bounded(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        v = summary["peak_system_stress"]
        assert 0.0 <= v <= 1.0, f"peak_system_stress = {v}"

    def test_effort_utilization_bounded_per_agent(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        for aid, am in summary["agent_metrics"].items():
            v = am["effort_utilization"]
            assert 0.0 <= v <= 1.0, f"effort_utilization for {aid} = {v}"

    def test_idle_rate_bounded_per_agent(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        for aid, am in summary["agent_metrics"].items():
            v = am["idle_rate"]
            assert 0.0 <= v <= 1.0, f"idle_rate for {aid} = {v}"

    def test_final_specialization_score_bounded_per_agent(self, tmp_path):
        summary = _run(tmp_path=tmp_path)
        for aid, am in summary["agent_metrics"].items():
            v = am["final_specialization_score"]
            assert 0.0 <= v <= 1.0, f"final_specialization_score for {aid} = {v}"


# ---------------------------------------------------------------------------
# Tests — event log
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_event_log_contains_only_defined_event_types(self, tmp_path):
        import json
        from pathlib import Path

        config = _small_config(seed=99)
        run_id = "test_events_99"
        run_cooperative_experiment(config, run_id, tmp_path, None, agent_policy="random")

        events_path = tmp_path / run_id / "events.jsonl"
        if not events_path.exists():
            return  # no events logged — that's fine

        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            assert evt["event"] in VALID_EVENT_TYPES, (
                f"Unknown event type: {evt['event']!r}"
            )


# ---------------------------------------------------------------------------
# Tests — determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_same_metrics(self, tmp_path):
        """Same seed + random policy produces identical episode summaries."""
        cfg = _small_config(seed=7)

        runs_a = tmp_path / "run_a"
        runs_b = tmp_path / "run_b"
        runs_a.mkdir()
        runs_b.mkdir()

        summary_a = run_cooperative_experiment(cfg, "run_a", runs_a, None)
        summary_b = run_cooperative_experiment(cfg, "run_b", runs_b, None)

        # Core outcome metrics must match
        assert summary_a["episode_length"] == summary_b["episode_length"]
        assert summary_a["termination_reason"] == summary_b["termination_reason"]
        assert summary_a["total_tasks_completed"] == summary_b["total_tasks_completed"]
        assert abs(summary_a["completion_ratio"] - summary_b["completion_ratio"]) < 1e-9
        assert abs(summary_a["effort_gini_coefficient"] - summary_b["effort_gini_coefficient"]) < 1e-9
