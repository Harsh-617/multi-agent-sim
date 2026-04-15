"""Unit tests for simulation/transfer/transfer_runner.py.

All tests use temporary directories and synthetic policy weights so no
pre-trained artifacts are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from simulation.transfer.transfer_runner import (
    _adapt_obs,
    _flatten_obs_for_archetype,
    _mismatch_strategy,
    _probe_target_obs_dim,
    run_transfer_experiment,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic league members
# ---------------------------------------------------------------------------

def _make_policy_pt(agent_dir: Path, obs_dim: int, num_action_types: int = 4) -> None:
    """Write a randomly-initialised SharedPolicyNetwork to *agent_dir/policy.pt*."""
    from simulation.training.ppo_shared import SharedPolicyNetwork

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    torch.save(net.state_dict(), agent_dir / "policy.pt")


def _make_metadata(
    agent_dir: Path,
    obs_dim: int,
    num_action_types: int = 4,
    extra: dict | None = None,
) -> None:
    meta = {
        "obs_dim": obs_dim,
        "num_action_types": num_action_types,
        "member_id": agent_dir.name,
        **(extra or {}),
    }
    (agent_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def _make_league_member(
    league_dir: Path,
    member_id: str,
    obs_dim: int,
    num_action_types: int = 4,
    extra_meta: dict | None = None,
) -> Path:
    member_dir = league_dir / member_id
    member_dir.mkdir(parents=True)
    _make_policy_pt(member_dir, obs_dim, num_action_types)
    _make_metadata(member_dir, obs_dim, num_action_types, extra_meta)
    return member_dir


# ---------------------------------------------------------------------------
# Fixtures: isolated storage paths
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all transfer_runner storage references to tmp_path."""
    import simulation.transfer.transfer_runner as tr

    mixed_league = tmp_path / "agents/league"
    comp_league = tmp_path / "agents/competitive_league"
    coop_league = tmp_path / "agents/cooperative/league"
    configs_dir = tmp_path / "configs"
    reports_dir = tmp_path / "reports"

    for d in (mixed_league, comp_league, coop_league, configs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tr, "_MIXED_LEAGUE_ROOT", mixed_league)
    monkeypatch.setattr(tr, "_COMPETITIVE_LEAGUE_ROOT", comp_league)
    monkeypatch.setattr(tr, "_COOPERATIVE_LEAGUE_ROOT", coop_league)
    monkeypatch.setattr(tr, "_CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(tr, "_REPORTS_ROOT", reports_dir)

    yield {
        "mixed_league": mixed_league,
        "comp_league": comp_league,
        "coop_league": coop_league,
        "configs_dir": configs_dir,
        "reports_dir": reports_dir,
    }


# ---------------------------------------------------------------------------
# _adapt_obs — truncation and padding
# ---------------------------------------------------------------------------

class TestAdaptObs:
    def test_truncation_when_target_larger_than_source(self):
        # target obs has 10 features, source expects 6 → truncate to 6
        target_flat = np.arange(10, dtype=np.float32)
        adapted = _adapt_obs(target_flat, source_obs_dim=6)
        assert adapted.shape == (6,)
        np.testing.assert_array_equal(adapted, target_flat[:6])

    def test_zero_padding_when_target_smaller_than_source(self):
        # target obs has 4 features, source expects 8 → zero-pad to 8
        target_flat = np.ones(4, dtype=np.float32)
        adapted = _adapt_obs(target_flat, source_obs_dim=8)
        assert adapted.shape == (8,)
        np.testing.assert_array_equal(adapted[:4], np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(adapted[4:], np.zeros(4, dtype=np.float32))

    def test_passthrough_when_dims_match(self):
        target_flat = np.linspace(0, 1, 12, dtype=np.float32)
        adapted = _adapt_obs(target_flat, source_obs_dim=12)
        assert adapted.shape == (12,)
        np.testing.assert_array_equal(adapted, target_flat)

    def test_truncation_result_dtype_is_float32(self):
        arr = np.ones(20, dtype=np.float64)
        # _adapt_obs doesn't re-cast but slices preserve dtype of input
        # tests that result is still usable as float32 by torch
        adapted = _adapt_obs(arr.astype(np.float32), source_obs_dim=5)
        assert adapted.dtype == np.float32

    def test_zero_pad_preserves_original_values(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        adapted = _adapt_obs(original, source_obs_dim=10)
        np.testing.assert_array_equal(adapted[:3], original)


# ---------------------------------------------------------------------------
# _mismatch_strategy
# ---------------------------------------------------------------------------

class TestMismatchStrategy:
    def test_truncate_when_target_larger(self):
        assert _mismatch_strategy(source_obs_dim=33, target_obs_dim=45) == "truncate"

    def test_pad_when_target_smaller(self):
        assert _mismatch_strategy(source_obs_dim=33, target_obs_dim=20) == "pad"

    def test_none_when_equal(self):
        assert _mismatch_strategy(source_obs_dim=33, target_obs_dim=33) == "none"


# ---------------------------------------------------------------------------
# _probe_target_obs_dim
# ---------------------------------------------------------------------------

class TestProbeTargetObsDim:
    def test_cooperative_obs_dim(self):
        from simulation.config.cooperative_defaults import default_cooperative_config
        from simulation.envs.cooperative.env import CooperativeEnvironment

        config = default_cooperative_config()
        env = CooperativeEnvironment(config)
        dim = _probe_target_obs_dim(env, "cooperative")
        expected = 8 + 4 * config.population.num_task_types + config.layers.history_window * (config.population.num_task_types + 1)
        assert dim == expected

    def test_mixed_obs_dim_positive(self):
        from simulation.config.defaults import default_config
        from simulation.envs.mixed.env import MixedEnvironment

        env = MixedEnvironment(default_config())
        dim = _probe_target_obs_dim(env, "mixed")
        assert dim > 0

    def test_competitive_obs_dim_positive(self):
        from simulation.config.competitive_defaults import default_competitive_config
        from simulation.envs.competitive.env import CompetitiveEnvironment

        env = CompetitiveEnvironment(default_competitive_config())
        dim = _probe_target_obs_dim(env, "competitive")
        assert dim > 0


# ---------------------------------------------------------------------------
# run_transfer_experiment — end-to-end with synthetic weights
# ---------------------------------------------------------------------------

def _setup_mixed_member(dirs: dict, obs_dim: int = 33) -> str:
    member_id = "league_000001"
    _make_league_member(dirs["mixed_league"], member_id, obs_dim=obs_dim)
    return member_id


def _setup_competitive_member(dirs: dict, obs_dim: int = 20) -> str:
    member_id = "league_000001"
    _make_league_member(dirs["comp_league"], member_id, obs_dim=obs_dim)
    return member_id


def _setup_cooperative_member(dirs: dict, obs_dim: int = 35) -> str:
    member_id = "league_000001"
    _make_league_member(
        dirs["coop_league"],
        member_id,
        obs_dim=obs_dim,
        num_action_types=4,  # 3 task types + IDLE
    )
    return member_id


class TestTransferRunnerEndToEnd:
    def test_mixed_to_cooperative_completes(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs, obs_dim=33)

        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=2,
            seed=42,
        )

        assert result is not None
        assert result["source_archetype"] == "mixed"
        assert result["target_archetype"] == "cooperative"

    def test_cooperative_to_mixed_completes(self, _patch_storage):
        dirs = _patch_storage
        # Get real coop obs_dim
        from simulation.config.cooperative_defaults import default_cooperative_config
        from simulation.envs.cooperative.env import CooperativeEnvironment
        cfg = default_cooperative_config()
        env = CooperativeEnvironment(cfg)
        coop_obs_dim = env.obs_dim()

        member_id = _setup_cooperative_member(dirs, obs_dim=coop_obs_dim)

        result = run_transfer_experiment(
            source_archetype="cooperative",
            source_member_id=member_id,
            target_archetype="mixed",
            target_config_id="default",
            episodes=2,
            seed=7,
        )

        assert result is not None
        assert result["source_archetype"] == "cooperative"
        assert result["target_archetype"] == "mixed"

    def test_competitive_to_mixed_completes(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_competitive_member(dirs, obs_dim=20)

        result = run_transfer_experiment(
            source_archetype="competitive",
            source_member_id=member_id,
            target_archetype="mixed",
            target_config_id="default",
            episodes=2,
            seed=1,
        )

        assert result is not None
        assert result["source_archetype"] == "competitive"
        assert result["target_archetype"] == "mixed"

    def test_result_contains_all_required_fields(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)

        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )

        required = [
            "report_type", "report_id",
            "source_archetype", "source_member_id", "source_obs_dim",
            "source_strategy_label", "source_elo",
            "target_archetype", "target_config_hash", "target_obs_dim",
            "obs_mismatch_strategy",
            "episodes", "seed",
            "transferred_results", "baseline_results",
            "transferred_mean", "baseline_mean",
            "vs_baseline_delta", "vs_baseline_pct",
        ]
        for field in required:
            assert field in result, f"Missing field: {field!r}"

    def test_report_type_is_transfer(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )
        assert result["report_type"] == "transfer"

    def test_obs_mismatch_strategy_truncate(self, _patch_storage):
        # source obs_dim=5 (tiny), cooperative target obs_dim >> 5 → truncate
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs, obs_dim=5)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )
        assert result["obs_mismatch_strategy"] == "truncate"
        assert result["target_obs_dim"] > result["source_obs_dim"]

    def test_obs_mismatch_strategy_pad(self, _patch_storage):
        # source obs_dim=9999 (huge), any target obs_dim << 9999 → pad
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs, obs_dim=9999)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )
        assert result["obs_mismatch_strategy"] == "pad"
        assert result["target_obs_dim"] < result["source_obs_dim"]

    def test_report_saved_to_correct_storage_path(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )
        report_id = result["report_id"]
        report_path = dirs["reports_dir"] / report_id / "summary.json"
        assert report_path.exists(), f"Expected report at {report_path}"

        saved = json.loads(report_path.read_text())
        assert saved["report_id"] == report_id
        assert saved["report_type"] == "transfer"

    def test_saved_summary_matches_returned_dict(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=1,
            seed=0,
        )
        saved = json.loads(
            (dirs["reports_dir"] / result["report_id"] / "summary.json").read_text()
        )
        assert saved["transferred_mean"] == result["transferred_mean"]
        assert saved["baseline_mean"] == result["baseline_mean"]

    def test_primary_metric_mixed_target_is_cooperation_rate(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_cooperative_member(dirs, obs_dim=35)
        result = run_transfer_experiment(
            source_archetype="cooperative",
            source_member_id=member_id,
            target_archetype="mixed",
            target_config_id="default",
            episodes=2,
            seed=0,
        )
        for ep in result["transferred_results"]:
            assert "cooperation_rate" in ep
            assert 0.0 <= ep["cooperation_rate"] <= 1.0

    def test_primary_metric_cooperative_target_is_completion_ratio(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=2,
            seed=0,
        )
        for ep in result["transferred_results"]:
            assert "completion_ratio" in ep
            assert 0.0 <= ep["completion_ratio"] <= 1.0

    def test_primary_metric_competitive_target_is_normalized_rank(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="competitive",
            target_config_id="default",
            episodes=2,
            seed=0,
        )
        for ep in result["transferred_results"]:
            assert "normalized_rank" in ep
            assert 0.0 <= ep["normalized_rank"] <= 1.0

    def test_episode_counts_match_request(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        for n in (1, 3):
            result = run_transfer_experiment(
                source_archetype="mixed",
                source_member_id=member_id,
                target_archetype="cooperative",
                target_config_id="default",
                episodes=n,
                seed=0,
            )
            assert len(result["transferred_results"]) == n
            assert len(result["baseline_results"]) == n
            assert result["episodes"] == n

    def test_transferred_mean_in_valid_range_for_cooperative_target(self, _patch_storage):
        dirs = _patch_storage
        member_id = _setup_mixed_member(dirs)
        result = run_transfer_experiment(
            source_archetype="mixed",
            source_member_id=member_id,
            target_archetype="cooperative",
            target_config_id="default",
            episodes=2,
            seed=0,
        )
        assert 0.0 <= result["transferred_mean"] <= 1.0
        assert 0.0 <= result["baseline_mean"] <= 1.0
