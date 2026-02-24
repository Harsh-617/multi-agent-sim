"""Unit tests for simulation.pipeline.pipeline_run.

Expensive stages (training, rating computation, robustness evaluation) are
monkeypatched to fast stubs so tests run without a GPU or long wall-clock time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from simulation.config.defaults import default_config
from simulation.evaluation.robustness import RobustnessResult
from simulation.evaluation.policy_set import PolicySpec


# ---------------------------------------------------------------------------
# Shared stub factories
# ---------------------------------------------------------------------------


def _make_stub_train(ppo_dir: Path):
    """Return a stub for ``simulation.training.ppo_shared.train``.

    Creates the minimal PPO artifacts expected by downstream stages.
    """

    def _stub(env_config, ppo_cfg=None):
        # Derive the agent dir the same way _save_artifacts does
        from simulation.training.ppo_shared import PPOConfig
        if ppo_cfg is None:
            ppo_cfg = PPOConfig()
        out_dir = Path(ppo_cfg.save_dir) / ppo_cfg.agent_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "policy.pt").write_bytes(b"STUB_WEIGHTS")
        (out_dir / "metadata.json").write_text(
            json.dumps({"algo": "ppo_shared", "obs_dim": 33, "training_steps": ppo_cfg.total_timesteps}),
            encoding="utf-8",
        )
        return out_dir

    return _stub


def _make_stub_compute_ratings(member_ids: list[str]):
    """Return a stub for ``compute_ratings`` that gives each member 1000."""

    def _stub(registry, num_matches=10, seed=42):
        return {mid: 1000.0 for mid in member_ids}

    return _stub


def _make_fake_league_members(league_dir: Path, count: int = 2) -> list[str]:
    """Create *count* fake league member directories; return their ids."""
    ids: list[str] = []
    for i in range(1, count + 1):
        mid = f"league_{i:06d}"
        d = league_dir / mid
        d.mkdir(parents=True, exist_ok=True)
        (d / "policy.pt").write_bytes(b"STUB_WEIGHTS")
        (d / "metadata.json").write_text(
            json.dumps({
                "member_id": mid,
                "algo": "ppo_shared",
                "obs_dim": 33,
                "parent_id": None,
                "created_at": f"2025-01-0{i}T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        ids.append(mid)
    return ids


def _make_stub_robustness():
    """Return a stub for ``evaluate_robustness`` with minimal valid output."""
    spec = PolicySpec(name="random", agent_policy="random", source="baseline")

    def _stub(base_config, policy_specs, sweeps, *, seeds, episodes_per_seed=1, max_steps_override=None):
        from simulation.evaluation.robustness import PolicyRobustness

        result = RobustnessResult()
        result.metadata = {
            "sweeps": [s.name for s in sweeps],
            "sweep_count": len(sweeps),
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "max_steps_override": max_steps_override,
            "policy_count": len(policy_specs),
        }
        for ps in policy_specs:
            if ps.available:
                result.per_sweep_results.setdefault("stub_sweep", {})[ps.name] = {
                    "available": True,
                    "mean_total_reward": 1.0,
                    "n_episodes": 1,
                }
                pr = PolicyRobustness(
                    policy_name=ps.name,
                    overall_mean_reward=1.0,
                    worst_case_mean_reward=0.8,
                    robustness_score=0.94,
                    collapse_rate_overall=0.0,
                    n_sweeps_evaluated=1,
                )
                result.per_policy_robustness[ps.name] = pr
        return result

    return _stub


def _make_stub_write_report(report_root: Path):
    """Return a stub for ``write_robustness_report`` that creates a fake dir."""

    def _stub(robustness_result, *, config_dict, report_root=report_root):
        report_dir = report_root / "robust_stub_20250101T000000Z"
        report_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "report_id": report_dir.name,
            "per_sweep_results": {},
            "per_policy_robustness": {},
            "metadata": {},
        }
        (report_dir / "report.json").write_text(json.dumps(data), encoding="utf-8")
        (report_dir / "report.md").write_text("# Stub report\n", encoding="utf-8")
        return report_dir

    return _stub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_dirs(tmp_path: Path):
    """Return a dict of isolated Path objects for test runs."""
    return {
        "agents_dir": tmp_path / "agents",
        "ppo_agent_dir": tmp_path / "agents" / "ppo_shared",
        "league_dir": tmp_path / "agents" / "league",
        "pipelines_dir": tmp_path / "pipelines",
        "configs_dir": tmp_path / "configs",
        "reports_dir": tmp_path / "reports",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunPipelineSummarySchema:
    """summary.json must contain the required top-level keys."""

    def test_summary_has_required_keys(
        self,
        tmp_path: Path,
        isolated_dirs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dirs = isolated_dirs
        league_dir = dirs["league_dir"]
        league_dir.mkdir(parents=True, exist_ok=True)

        # Create fake league members so champion selection works
        member_ids = _make_fake_league_members(league_dir, count=2)

        # Patch expensive functions
        import simulation.pipeline.pipeline_run as pr_mod

        monkeypatch.setattr(pr_mod, "train", _make_stub_train(dirs["ppo_agent_dir"]))
        monkeypatch.setattr(
            pr_mod,
            "compute_ratings",
            _make_stub_compute_ratings(member_ids),
        )
        monkeypatch.setattr(pr_mod, "evaluate_robustness", _make_stub_robustness())
        monkeypatch.setattr(
            pr_mod,
            "write_robustness_report",
            _make_stub_write_report(dirs["reports_dir"]),
        )

        from simulation.pipeline.pipeline_run import run_pipeline

        out_dir = run_pipeline(
            config_id="default",
            seed=42,
            seeds=1,
            episodes_per_seed=1,
            total_timesteps=256,
            snapshot_every_timesteps=128,
            max_league_members=5,
            num_matches=1,
            limit_sweeps=1,
            ppo_agent_dir=dirs["ppo_agent_dir"],
            pipelines_dir=dirs["pipelines_dir"],
            configs_dir=dirs["configs_dir"],
            reports_dir=dirs["reports_dir"],
        )

        assert out_dir.is_dir()
        summary_file = out_dir / "summary.json"
        assert summary_file.exists()

        data = json.loads(summary_file.read_text(encoding="utf-8"))

        # Required top-level keys
        for key in (
            "pipeline_id",
            "timestamp",
            "config_id",
            "config_hash",
            "seed",
            "training",
            "ratings",
            "robustness",
            "report_id",
            "summary_path",
        ):
            assert key in data, f"Missing key: {key}"

        # Nested keys
        assert "ppo_agent_dir" in data["training"]
        assert "total_timesteps" in data["training"]
        assert "champion_id" in data["ratings"]
        assert "report_id" in data["robustness"]
        assert "n_sweeps" in data["robustness"]

    def test_pipeline_id_has_expected_prefix(
        self,
        isolated_dirs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dirs = isolated_dirs
        league_dir = dirs["league_dir"]
        league_dir.mkdir(parents=True, exist_ok=True)
        member_ids = _make_fake_league_members(league_dir, count=1)

        import simulation.pipeline.pipeline_run as pr_mod

        monkeypatch.setattr(pr_mod, "train", _make_stub_train(dirs["ppo_agent_dir"]))
        monkeypatch.setattr(pr_mod, "compute_ratings", _make_stub_compute_ratings(member_ids))
        monkeypatch.setattr(pr_mod, "evaluate_robustness", _make_stub_robustness())
        monkeypatch.setattr(pr_mod, "write_robustness_report", _make_stub_write_report(dirs["reports_dir"]))

        from simulation.pipeline.pipeline_run import run_pipeline

        out_dir = run_pipeline(
            seed=42,
            seeds=1,
            episodes_per_seed=1,
            total_timesteps=256,
            snapshot_every_timesteps=128,
            max_league_members=5,
            num_matches=1,
            limit_sweeps=1,
            ppo_agent_dir=dirs["ppo_agent_dir"],
            pipelines_dir=dirs["pipelines_dir"],
            configs_dir=dirs["configs_dir"],
            reports_dir=dirs["reports_dir"],
        )

        data = json.loads((out_dir / "summary.json").read_text())
        assert data["pipeline_id"].startswith("pipeline_")


class TestRunPipelineDeterminism:
    """Same seed => same summary key set (keys stable, not necessarily values)."""

    def test_summary_keys_stable_across_calls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _run(run_idx: int) -> set:
            dirs = {
                "ppo_agent_dir": tmp_path / f"run{run_idx}" / "agents" / "ppo_shared",
                "league_dir": tmp_path / f"run{run_idx}" / "agents" / "league",
                "pipelines_dir": tmp_path / f"run{run_idx}" / "pipelines",
                "configs_dir": tmp_path / f"run{run_idx}" / "configs",
                "reports_dir": tmp_path / f"run{run_idx}" / "reports",
            }
            league_dir = dirs["league_dir"]
            league_dir.mkdir(parents=True, exist_ok=True)
            member_ids = _make_fake_league_members(league_dir, count=1)

            import simulation.pipeline.pipeline_run as pr_mod

            monkeypatch.setattr(pr_mod, "train", _make_stub_train(dirs["ppo_agent_dir"]))
            monkeypatch.setattr(pr_mod, "compute_ratings", _make_stub_compute_ratings(member_ids))
            monkeypatch.setattr(pr_mod, "evaluate_robustness", _make_stub_robustness())
            monkeypatch.setattr(pr_mod, "write_robustness_report", _make_stub_write_report(dirs["reports_dir"]))

            from simulation.pipeline.pipeline_run import run_pipeline

            out_dir = run_pipeline(
                seed=42,
                seeds=1,
                episodes_per_seed=1,
                total_timesteps=256,
                snapshot_every_timesteps=128,
                max_league_members=5,
                num_matches=1,
                limit_sweeps=1,
                ppo_agent_dir=dirs["ppo_agent_dir"],
                pipelines_dir=dirs["pipelines_dir"],
                configs_dir=dirs["configs_dir"],
                reports_dir=dirs["reports_dir"],
            )
            return set(json.loads((out_dir / "summary.json").read_text()).keys())

        keys1 = _run(1)
        keys2 = _run(2)
        assert keys1 == keys2


class TestRunPipelineProgressCallback:
    """Progress callback receives stage updates in expected order."""

    def test_callback_called_with_expected_stages(
        self,
        isolated_dirs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dirs = isolated_dirs
        league_dir = dirs["league_dir"]
        league_dir.mkdir(parents=True, exist_ok=True)
        member_ids = _make_fake_league_members(league_dir, count=1)

        import simulation.pipeline.pipeline_run as pr_mod

        monkeypatch.setattr(pr_mod, "train", _make_stub_train(dirs["ppo_agent_dir"]))
        monkeypatch.setattr(pr_mod, "compute_ratings", _make_stub_compute_ratings(member_ids))
        monkeypatch.setattr(pr_mod, "evaluate_robustness", _make_stub_robustness())
        monkeypatch.setattr(pr_mod, "write_robustness_report", _make_stub_write_report(dirs["reports_dir"]))

        from simulation.pipeline.pipeline_run import run_pipeline

        stages_seen: list[str] = []

        def _cb(stage: str, detail: str = "") -> None:
            stages_seen.append(stage)

        run_pipeline(
            seed=42,
            seeds=1,
            episodes_per_seed=1,
            total_timesteps=256,
            snapshot_every_timesteps=128,
            max_league_members=5,
            num_matches=1,
            limit_sweeps=1,
            progress_callback=_cb,
            ppo_agent_dir=dirs["ppo_agent_dir"],
            pipelines_dir=dirs["pipelines_dir"],
            configs_dir=dirs["configs_dir"],
            reports_dir=dirs["reports_dir"],
        )

        assert "loading_config" in stages_seen
        assert "training" in stages_seen
        assert "ratings" in stages_seen
        assert "robustness" in stages_seen
        assert "saving" in stages_seen
        assert "done" in stages_seen


class TestRunPipelineNoLeagueMembers:
    """Pipeline must complete even when no league members exist after training."""

    def test_pipeline_completes_with_no_members(
        self,
        isolated_dirs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dirs = isolated_dirs
        # league_dir left empty – no members

        import simulation.pipeline.pipeline_run as pr_mod

        # train stub that does NOT create league members
        def _empty_train(env_config, ppo_cfg=None):
            from simulation.training.ppo_shared import PPOConfig
            if ppo_cfg is None:
                ppo_cfg = PPOConfig()
            out_dir = Path(ppo_cfg.save_dir) / ppo_cfg.agent_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "policy.pt").write_bytes(b"STUB")
            (out_dir / "metadata.json").write_text("{}", encoding="utf-8")
            return out_dir

        monkeypatch.setattr(pr_mod, "train", _empty_train)
        monkeypatch.setattr(pr_mod, "compute_ratings", lambda *a, **kw: {})
        monkeypatch.setattr(pr_mod, "evaluate_robustness", _make_stub_robustness())
        monkeypatch.setattr(pr_mod, "write_robustness_report", _make_stub_write_report(dirs["reports_dir"]))

        from simulation.pipeline.pipeline_run import run_pipeline

        out_dir = run_pipeline(
            seed=42,
            seeds=1,
            episodes_per_seed=1,
            total_timesteps=256,
            snapshot_every_timesteps=128,
            max_league_members=5,
            num_matches=1,
            limit_sweeps=1,
            ppo_agent_dir=dirs["ppo_agent_dir"],
            pipelines_dir=dirs["pipelines_dir"],
            configs_dir=dirs["configs_dir"],
            reports_dir=dirs["reports_dir"],
        )

        data = json.loads((out_dir / "summary.json").read_text())
        # champion_id is None when no members exist
        assert data["ratings"]["champion_id"] is None
        assert data["ratings"]["num_members_rated"] == 0
