"""Unit tests for the evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.config.schema import (
    MixedEnvironmentConfig,
    EnvironmentIdentity,
    PopulationConfig,
    LayerConfig,
    RewardWeights,
    AgentConfig,
)
from simulation.evaluation.policy_set import (
    BASELINE_POLICIES,
    PolicySpec,
    resolve_policy_set,
)
from simulation.evaluation.evaluator import evaluate_policies
from simulation.evaluation.reporting import config_hash, write_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_config(seed: int = 42, **pop_overrides) -> MixedEnvironmentConfig:
    """Small fast config for testing."""
    pop_kwargs = dict(
        num_agents=3,
        max_steps=10,
        initial_shared_pool=100.0,
        initial_agent_resources=20.0,
        collapse_threshold=5.0,
    )
    pop_kwargs.update(pop_overrides)
    return MixedEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(**pop_kwargs),
        layers=LayerConfig(
            information_asymmetry=0.0,
            temporal_memory_depth=5,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.0,
        ),
        rewards=RewardWeights(
            individual_weight=1.0,
            group_weight=0.5,
            relational_weight=0.3,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(observation_memory_steps=3),
    )


# ---------------------------------------------------------------------------
# PolicySet resolver
# ---------------------------------------------------------------------------


class TestPolicySetResolver:
    def test_baselines_present(self, tmp_path: Path) -> None:
        """All four baseline policies are always available."""
        league_root = tmp_path / "league"
        league_root.mkdir()
        specs = resolve_policy_set(
            league_root=league_root,
            ratings_path=league_root / "ratings.json",
            ppo_dir=tmp_path / "ppo_shared",
            top_k=3,
        )
        baseline_specs = [s for s in specs if s.source == "baseline"]
        assert len(baseline_specs) == len(BASELINE_POLICIES)
        for s in baseline_specs:
            assert s.available is True
            assert s.agent_policy in BASELINE_POLICIES

    def test_skips_missing_ppo_shared(self, tmp_path: Path) -> None:
        """ppo_shared is marked unavailable when artifacts are missing."""
        league_root = tmp_path / "league"
        league_root.mkdir()
        specs = resolve_policy_set(
            league_root=league_root,
            ratings_path=league_root / "ratings.json",
            ppo_dir=tmp_path / "ppo_shared_missing",
            top_k=3,
        )
        ppo_specs = [s for s in specs if s.source == "ppo"]
        assert len(ppo_specs) == 1
        assert ppo_specs[0].available is False
        assert "Missing artifacts" in (ppo_specs[0].skip_reason or "")

    def test_skips_champion_when_no_league(self, tmp_path: Path) -> None:
        """Champion and league members are skipped when league is empty."""
        league_root = tmp_path / "league"
        league_root.mkdir()
        specs = resolve_policy_set(
            league_root=league_root,
            ratings_path=league_root / "ratings.json",
            ppo_dir=tmp_path / "ppo_missing",
            top_k=3,
        )
        champ_specs = [s for s in specs if s.source == "champion"]
        assert len(champ_specs) == 1
        assert champ_specs[0].available is False


# ---------------------------------------------------------------------------
# Evaluator determinism
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_determinism_same_seed(self) -> None:
        """Same seed + same policy => identical results."""
        config = _quick_config()
        spec = PolicySpec(
            name="random",
            agent_policy="random",
            source="baseline",
        )

        run_a = evaluate_policies(config, [spec], seeds=[42], episodes_per_seed=1)
        run_b = evaluate_policies(config, [spec], seeds=[42], episodes_per_seed=1)

        assert run_a[0].mean_total_reward == run_b[0].mean_total_reward
        assert run_a[0].mean_final_shared_pool == run_b[0].mean_final_shared_pool
        assert run_a[0].mean_episode_length == run_b[0].mean_episode_length

    def test_unavailable_policy_not_run(self) -> None:
        """Unavailable policies appear in results but have zero episodes."""
        config = _quick_config()
        spec = PolicySpec(
            name="fake",
            agent_policy="fake",
            source="ppo",
            available=False,
            skip_reason="testing",
        )
        results = evaluate_policies(config, [spec], seeds=[1], episodes_per_seed=1)
        assert len(results) == 1
        assert results[0].spec.available is False
        assert len(results[0].episodes) == 0

    def test_max_steps_override(self) -> None:
        """max_steps_override caps episode length."""
        config = _quick_config(max_steps=50)
        spec = PolicySpec(
            name="random", agent_policy="random", source="baseline"
        )
        results = evaluate_policies(
            config, [spec], seeds=[1], episodes_per_seed=1, max_steps_override=5
        )
        assert results[0].mean_episode_length <= 5


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    def test_config_hash_stable(self) -> None:
        d = {"a": 1, "b": [2, 3]}
        assert config_hash(d) == config_hash(d)
        assert len(config_hash(d)) == 12

    def test_writes_json_and_md(self, tmp_path: Path) -> None:
        """write_report produces both report.json and report.md."""
        config = _quick_config()
        spec = PolicySpec(
            name="random", agent_policy="random", source="baseline"
        )
        results = evaluate_policies(config, [spec], seeds=[1], episodes_per_seed=1)

        config_dict = json.loads(config.model_dump_json())
        report_dir = write_report(
            results,
            config_dict=config_dict,
            seeds=[1],
            episodes_per_seed=1,
            report_root=tmp_path / "reports",
        )

        assert report_dir.exists()
        assert (report_dir / "report.json").exists()
        assert (report_dir / "report.md").exists()

        # Validate JSON structure
        data = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        assert "report_id" in data
        assert "config_hash" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["policy_name"] == "random"

        # Markdown contains a table
        md = (report_dir / "report.md").read_text(encoding="utf-8")
        assert "| Policy |" in md
        assert "random" in md
