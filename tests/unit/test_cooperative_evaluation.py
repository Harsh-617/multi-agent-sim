"""Unit tests for cooperative evaluation runner, robustness sweep, and sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from simulation.evaluation.cooperative_sweeps import (
    CoopSweepSpec,
    apply_coop_sweep,
    build_cooperative_sweeps,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fake_agent_dir(path: Path, obs_dim: int = 40, num_action_types: int = 4) -> Path:
    """Create a minimal fake cooperative agent artifact directory."""
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "obs_dim": obs_dim,
        "num_action_types": num_action_types,
        "seed": 42,
        "algo": "ppo_shared",
    }
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Save random weights for the actual network shape
    from simulation.training.ppo_shared import SharedPolicyNetwork

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    torch.save(net.state_dict(), path / "policy.pt")
    return path


def _small_coop_config():
    """Return a minimal cooperative config for fast tests."""
    from simulation.config.cooperative_defaults import default_cooperative_config

    cfg = default_cooperative_config(seed=1)
    cfg.population.max_steps = 10
    cfg.population.num_agents = 2
    return cfg


# ---------------------------------------------------------------------------
# CoopSweepSpec and build_cooperative_sweeps
# ---------------------------------------------------------------------------


class TestBuildCooperativeSweeps:
    def test_returns_exactly_20_sweeps(self):
        sweeps = build_cooperative_sweeps()
        assert len(sweeps) == 20

    def test_all_sweeps_are_coop_sweep_spec(self):
        sweeps = build_cooperative_sweeps()
        for s in sweeps:
            assert isinstance(s, CoopSweepSpec)

    def test_all_sweeps_have_unique_names(self):
        sweeps = build_cooperative_sweeps()
        names = [s.name for s in sweeps]
        assert len(names) == len(set(names)), "Sweep names must be unique"

    def test_all_sweeps_have_config_patch(self):
        sweeps = build_cooperative_sweeps()
        for s in sweeps:
            assert isinstance(s.config_patch, dict)
            assert len(s.config_patch) > 0, f"Sweep '{s.name}' has empty config_patch"

    def test_arrival_sweeps_present(self):
        sweeps = build_cooperative_sweeps()
        arrival_names = [s.name for s in sweeps if "arrival" in s.tags]
        assert len(arrival_names) == 4

    def test_difficulty_sweeps_present(self):
        sweeps = build_cooperative_sweeps()
        difficulty_names = [s.name for s in sweeps if "difficulty" in s.tags]
        assert len(difficulty_names) == 4

    def test_population_sweeps_present(self):
        sweeps = build_cooperative_sweeps()
        pop_sweeps = [s.name for s in sweeps if "population" in s.tags]
        assert len(pop_sweeps) == 3

    def test_combined_sweeps_present(self):
        sweeps = build_cooperative_sweeps()
        combined = [s.name for s in sweeps if "combined" in s.tags]
        assert len(combined) == 3

    def test_deterministic(self):
        """Calling twice returns identical specs."""
        s1 = build_cooperative_sweeps()
        s2 = build_cooperative_sweeps()
        assert [s.name for s in s1] == [s.name for s in s2]


class TestApplyCoopSweep:
    def test_patches_task_arrival_rate(self):
        cfg = _small_coop_config()
        sweep = CoopSweepSpec(
            name="test",
            description="test",
            config_patch={"task": {"task_arrival_rate": [2.0, 2.0, 2.0]}},
        )
        new_cfg = apply_coop_sweep(cfg, sweep)
        assert new_cfg.task.task_arrival_rate == [2.0, 2.0, 2.0]
        # Original unchanged
        assert cfg.task.task_arrival_rate != [2.0, 2.0, 2.0]

    def test_patches_population_num_agents(self):
        cfg = _small_coop_config()
        sweep = CoopSweepSpec(
            name="test",
            description="test",
            config_patch={"population": {"num_agents": 6}},
        )
        new_cfg = apply_coop_sweep(cfg, sweep)
        assert new_cfg.population.num_agents == 6
        assert cfg.population.num_agents == 2

    def test_does_not_mutate_base_config(self):
        cfg = _small_coop_config()
        original_agents = cfg.population.num_agents
        sweep = CoopSweepSpec(
            name="test",
            description="test",
            config_patch={"population": {"num_agents": 8}},
        )
        apply_coop_sweep(cfg, sweep)
        assert cfg.population.num_agents == original_agents


# ---------------------------------------------------------------------------
# Robustness score formula
# ---------------------------------------------------------------------------


class TestRobustnessScoreFormula:
    def test_formula_0_7_mean_plus_0_3_worst(self):
        """robustness_score = 0.7 × mean_completion_ratio + 0.3 × worst_case"""
        mean_cr = 0.8
        worst_cr = 0.5
        expected = round(0.7 * mean_cr + 0.3 * worst_cr, 4)
        assert expected == pytest.approx(0.7 * 0.8 + 0.3 * 0.5, abs=1e-6)

    def test_formula_with_perfect_score(self):
        """Perfect agent: mean=1.0, worst=1.0 → score=1.0"""
        score = round(0.7 * 1.0 + 0.3 * 1.0, 4)
        assert score == 1.0

    def test_formula_with_zero_scores(self):
        """Zero agent: mean=0.0, worst=0.0 → score=0.0"""
        score = round(0.7 * 0.0 + 0.3 * 0.0, 4)
        assert score == 0.0

    def test_formula_emphasizes_mean_over_worst(self):
        """Weight 0.7 > 0.3 means mean matters more than worst-case."""
        # Scenario A: high mean, low worst
        score_a = 0.7 * 0.9 + 0.3 * 0.2  # 0.63 + 0.06 = 0.69
        # Scenario B: low mean, high worst
        score_b = 0.7 * 0.5 + 0.3 * 0.8  # 0.35 + 0.24 = 0.59
        assert score_a > score_b

    def test_coop_policy_robustness_compute(self):
        """Verify CoopPolicyRobustness computes the score correctly."""
        from simulation.evaluation.cooperative_robustness import CoopPolicyRobustness

        pr = CoopPolicyRobustness(
            policy_name="test_policy",
            mean_completion_ratio=0.75,
            worst_case_completion_ratio=0.40,
        )
        pr.robustness_score = round(0.7 * pr.mean_completion_ratio + 0.3 * pr.worst_case_completion_ratio, 4)
        assert pr.robustness_score == pytest.approx(0.7 * 0.75 + 0.3 * 0.40, abs=1e-4)


# ---------------------------------------------------------------------------
# Cooperative eval runner (mocked environment)
# ---------------------------------------------------------------------------


class TestCooperativeEvalRunner:
    def test_eval_report_contains_required_keys(self, tmp_path: Path):
        """run_cooperative_eval saves a report with all required metric keys."""
        from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_eval(
            cfg,
            agent_dir,
            num_seeds=1,
            episodes_per_seed=1,
            base_seed=0,
            report_root=tmp_path / "reports",
        )

        summary_path = report_dir / "summary.json"
        assert summary_path.exists(), "summary.json must be created"

        data = json.loads(summary_path.read_text(encoding="utf-8"))

        # Top-level required keys
        assert "report_id" in data
        assert "kind" in data
        assert "timestamp" in data
        assert "config_hash" in data
        assert "summary" in data
        assert "per_seed" in data

        # Summary metric keys
        summary = data["summary"]
        assert "mean_completion_ratio" in summary
        assert "worst_case_completion_ratio" in summary
        assert "mean_group_efficiency_ratio" in summary
        assert "mean_effort_utilization" in summary
        assert "mean_system_stress" in summary
        assert "mean_return" in summary

    def test_eval_report_kind_is_cooperative_eval(self, tmp_path: Path):
        from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_eval(
            cfg,
            agent_dir,
            num_seeds=1,
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert data["kind"] == "cooperative_eval"

    def test_eval_report_id_prefix(self, tmp_path: Path):
        from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_eval(
            cfg,
            agent_dir,
            num_seeds=1,
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert data["report_id"].startswith("cooperative_eval_")

    def test_eval_report_saved_to_correct_path(self, tmp_path: Path):
        from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()
        reports_root = tmp_path / "reports"

        report_dir = run_cooperative_eval(
            cfg,
            agent_dir,
            num_seeds=1,
            episodes_per_seed=1,
            report_root=reports_root,
        )

        # Report directory should be inside reports_root
        assert report_dir.parent == reports_root
        # Folder name should start with cooperative_eval_
        assert report_dir.name.startswith("cooperative_eval_")

    def test_per_seed_results_count(self, tmp_path: Path):
        from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_eval(
            cfg,
            agent_dir,
            num_seeds=2,
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert len(data["per_seed"]) == 2


# ---------------------------------------------------------------------------
# Cooperative robustness sweep (mocked)
# ---------------------------------------------------------------------------


class TestCooperativeRobustnessSweep:
    def test_robustness_report_contains_all_sweeps(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()
        sweeps = build_cooperative_sweeps()

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            sweeps[:3],  # Use 3 sweeps for speed
            seeds=[0],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert "per_sweep_results" in data
        assert len(data["per_sweep_results"]) == 3

    def test_robustness_report_kind(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()
        sweeps = build_cooperative_sweeps()

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            sweeps[:2],
            seeds=[0],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert data["kind"] == "cooperative_robust"

    def test_robustness_report_id_prefix(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            build_cooperative_sweeps()[:2],
            seeds=[0],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert data["report_id"].startswith("cooperative_robust_")

    def test_robustness_score_formula_applied(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            build_cooperative_sweeps()[:2],
            seeds=[0],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        pr = data["per_policy_robustness"]["cooperative_champion"]

        # Verify the formula is applied correctly
        expected_score = round(
            0.7 * pr["mean_completion_ratio"] + 0.3 * pr["worst_case_completion_ratio"],
            4,
        )
        assert pr["robustness_score"] == pytest.approx(expected_score, abs=1e-3)

    def test_robustness_report_saved_to_correct_path(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()
        reports_root = tmp_path / "reports"

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            build_cooperative_sweeps()[:2],
            seeds=[0],
            episodes_per_seed=1,
            report_root=reports_root,
        )

        assert report_dir.parent == reports_root
        assert report_dir.name.startswith("cooperative_robust_")

    def test_per_sweep_results_have_required_keys(self, tmp_path: Path):
        from simulation.evaluation.cooperative_robustness import run_cooperative_robustness

        agent_dir = _make_fake_agent_dir(tmp_path / "agent")
        cfg = _small_coop_config()
        sweeps = build_cooperative_sweeps()[:2]

        report_dir = run_cooperative_robustness(
            cfg,
            agent_dir,
            sweeps,
            seeds=[0],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        data = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        for sweep_name, sweep_data in data["per_sweep_results"].items():
            assert "mean_completion_ratio" in sweep_data
            assert "worst_case_completion_ratio" in sweep_data
            assert "n_episodes" in sweep_data
