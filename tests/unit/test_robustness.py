"""Unit tests for robustness sweeps and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.config.schema import (
    AgentConfig,
    EnvironmentIdentity,
    LayerConfig,
    MixedEnvironmentConfig,
    PopulationConfig,
    RewardWeights,
)
from simulation.evaluation.policy_set import PolicySpec
from simulation.evaluation.reporting import write_robustness_report
from simulation.evaluation.robustness import evaluate_robustness
from simulation.evaluation.sweeps import SweepSpec, apply_sweep, build_default_sweeps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_config(seed: int = 42) -> MixedEnvironmentConfig:
    """Small fast config for testing."""
    return MixedEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=3,
            max_steps=10,
            initial_shared_pool=100.0,
            initial_agent_resources=20.0,
            collapse_threshold=5.0,
        ),
        layers=LayerConfig(
            information_asymmetry=0.3,
            temporal_memory_depth=5,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.1,
        ),
        rewards=RewardWeights(
            individual_weight=1.0,
            group_weight=0.5,
            relational_weight=0.3,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(observation_memory_steps=3),
    )


_BASELINE_SPEC = PolicySpec(
    name="random", agent_policy="random", source="baseline"
)


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------


class TestSweeps:
    def test_apply_sweep_does_not_mutate_base_config(self) -> None:
        """apply_sweep must return a new config; base stays unchanged."""
        base = _quick_config()
        original_ia = base.layers.information_asymmetry
        original_pool = base.population.initial_shared_pool

        sweep = SweepSpec(
            name="test_sweep",
            description="bump info asymmetry",
            config_patch={"layers": {"information_asymmetry": 0.9}},
        )
        swept = apply_sweep(base, sweep)

        # Base config unchanged
        assert base.layers.information_asymmetry == original_ia
        assert base.population.initial_shared_pool == original_pool

        # Swept config changed
        assert swept.layers.information_asymmetry == 0.9
        # Unrelated fields preserved
        assert swept.population.initial_shared_pool == original_pool

    def test_build_default_sweeps_non_empty(self) -> None:
        sweeps = build_default_sweeps()
        assert len(sweeps) >= 8
        names = [s.name for s in sweeps]
        assert len(names) == len(set(names)), "Sweep names must be unique"


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------


class TestRobustnessEvaluation:
    def test_robustness_output_contains_required_keys(self) -> None:
        """Output structure has metadata, per_sweep_results, per_policy_robustness."""
        config = _quick_config()
        sweeps = [
            SweepSpec(
                name="mild",
                description="low uncertainty",
                config_patch={"layers": {"uncertainty_intensity": 0.0}},
            ),
        ]
        result = evaluate_robustness(
            config,
            [_BASELINE_SPEC],
            sweeps,
            seeds=[1],
            episodes_per_seed=1,
            max_steps_override=5,
        )

        d = result.to_dict()
        assert "metadata" in d
        assert "per_sweep_results" in d
        assert "per_policy_robustness" in d

        # Metadata keys
        meta = d["metadata"]
        assert "sweeps" in meta
        assert "seeds" in meta
        assert "episodes_per_seed" in meta

        # Per-sweep structure
        assert "mild" in d["per_sweep_results"]
        assert "random" in d["per_sweep_results"]["mild"]

        # Per-policy robustness
        pr = d["per_policy_robustness"]["random"]
        for key in (
            "overall_mean_reward",
            "worst_case_mean_reward",
            "robustness_score",
            "collapse_rate_overall",
        ):
            assert key in pr

    def test_robustness_determinism_same_seed(self) -> None:
        """Same config + same seeds => identical robustness scores."""
        config = _quick_config()
        sweeps = [
            SweepSpec(
                name="baseline_sweep",
                description="no change",
                config_patch={"layers": {"uncertainty_intensity": 0.0}},
            ),
        ]
        kwargs = dict(
            seeds=[42],
            episodes_per_seed=1,
            max_steps_override=5,
        )

        r1 = evaluate_robustness(config, [_BASELINE_SPEC], sweeps, **kwargs)
        r2 = evaluate_robustness(config, [_BASELINE_SPEC], sweeps, **kwargs)

        pr1 = r1.per_policy_robustness["random"]
        pr2 = r2.per_policy_robustness["random"]
        assert pr1.robustness_score == pr2.robustness_score
        assert pr1.overall_mean_reward == pr2.overall_mean_reward
        assert pr1.worst_case_mean_reward == pr2.worst_case_mean_reward


# ---------------------------------------------------------------------------
# Robustness reporting
# ---------------------------------------------------------------------------


class TestRobustnessReporting:
    def test_reporting_writes_md_and_json(self, tmp_path: Path) -> None:
        """write_robustness_report produces report.json and report.md."""
        config = _quick_config()
        sweeps = [
            SweepSpec(
                name="mild",
                description="low uncertainty",
                config_patch={"layers": {"uncertainty_intensity": 0.0}},
            ),
        ]
        result = evaluate_robustness(
            config,
            [_BASELINE_SPEC],
            sweeps,
            seeds=[1],
            episodes_per_seed=1,
            max_steps_override=5,
        )

        config_dict = json.loads(config.model_dump_json())
        report_dir = write_robustness_report(
            result,
            config_dict=config_dict,
            report_root=tmp_path / "reports",
        )

        assert report_dir.exists()
        assert (report_dir / "report.json").exists()
        assert (report_dir / "report.md").exists()

        # Validate JSON
        data = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        assert data["report_id"].startswith("robust_")
        assert "per_sweep_results" in data
        assert "per_policy_robustness" in data

        # Markdown has the table
        md = (report_dir / "report.md").read_text(encoding="utf-8")
        assert "Robustness Report" in md
        assert "| Policy |" in md
        assert "random" in md
