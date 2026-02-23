"""Tests for RunRegistry — scanning stored run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.registry.run_registry import RunRegistry


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def _create_run(
    runs_dir: Path,
    run_id: str,
    *,
    seed: int = 42,
    agent_policy: str = "random",
    termination_reason: str = "max_steps",
    episode_length: int = 100,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> Path:
    """Create a minimal run directory with config + summary."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()

    config = {
        "written_at": timestamp,
        "_agent_policy": agent_policy,
        "identity": {"seed": seed, "environment_type": "mixed", "environment_version": "0.1.0"},
        "population": {"num_agents": 3, "max_steps": 100},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    summary = {
        "written_at": timestamp,
        "episode_length": episode_length,
        "termination_reason": termination_reason,
        "final_shared_pool": 85.0,
        "total_reward_per_agent": {"agent_0": 10.0, "agent_1": 12.0},
    }
    (run_dir / "episode_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    return run_dir


class TestListRuns:

    def test_empty_dir(self, runs_dir: Path):
        reg = RunRegistry(runs_dir)
        assert reg.list_runs() == []

    def test_nonexistent_dir(self, tmp_path: Path):
        reg = RunRegistry(tmp_path / "nope")
        assert reg.list_runs() == []

    def test_lists_runs_with_metadata(self, runs_dir: Path):
        _create_run(runs_dir, "run_a", agent_policy="random", timestamp="2026-01-01T00:00:00+00:00")
        _create_run(runs_dir, "run_b", agent_policy="tit_for_tat", timestamp="2026-01-02T00:00:00+00:00")

        reg = RunRegistry(runs_dir)
        result = reg.list_runs()

        assert len(result) == 2
        # Most recent first
        assert result[0]["run_id"] == "run_b"
        assert result[1]["run_id"] == "run_a"
        assert result[0]["agent_policy"] == "tit_for_tat"
        assert result[1]["agent_policy"] == "random"
        assert result[0]["termination_reason"] == "max_steps"
        assert result[0]["episode_length"] == 100

    def test_skips_dirs_without_config(self, runs_dir: Path):
        (runs_dir / "incomplete_run").mkdir()
        _create_run(runs_dir, "good_run")

        reg = RunRegistry(runs_dir)
        result = reg.list_runs()
        assert len(result) == 1
        assert result[0]["run_id"] == "good_run"


class TestGetRun:

    def test_returns_none_for_missing(self, runs_dir: Path):
        reg = RunRegistry(runs_dir)
        assert reg.get_run("nonexistent") is None

    def test_returns_full_detail(self, runs_dir: Path):
        _create_run(runs_dir, "detail_run", episode_length=50, termination_reason="system_collapse")

        reg = RunRegistry(runs_dir)
        result = reg.get_run("detail_run")

        assert result is not None
        assert result["run_id"] == "detail_run"
        assert result["termination_reason"] == "system_collapse"
        assert result["episode_length"] == 50
        assert result["episode_summary"] is not None
        assert result["episode_summary"]["final_shared_pool"] == 85.0
        assert "written_at" not in result["episode_summary"]  # stripped

    def test_handles_missing_summary(self, runs_dir: Path):
        run_dir = runs_dir / "no_summary"
        run_dir.mkdir()
        config = {"written_at": "2026-01-01T00:00:00+00:00", "identity": {"seed": 1}}
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        reg = RunRegistry(runs_dir)
        result = reg.get_run("no_summary")
        assert result is not None
        assert result["episode_summary"] is None
