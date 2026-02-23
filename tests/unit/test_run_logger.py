"""Tests for RunLogger — verifies file creation and content structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.runner.run_logger import RunLogger


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


class TestRunLoggerWritesFiles:
    """RunLogger creates the expected artifact files."""

    def test_write_config(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_001")
        logger.write_config({"identity": {"seed": 42}, "population": {"num_agents": 5}})

        config_path = logger.run_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "written_at" in data
        assert data["identity"]["seed"] == 42
        assert data["population"]["num_agents"] == 5

    def test_log_step_metrics(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_002")
        records = [
            {"step": 0, "agent_id": "a0", "reward": 0.1},
            {"step": 0, "agent_id": "a1", "reward": 0.2},
        ]
        logger.log_step_metrics(records)
        logger.log_step_metrics([{"step": 1, "agent_id": "a0", "reward": 0.15}])

        metrics_path = logger.run_dir / "metrics.jsonl"
        assert metrics_path.exists()
        lines = metrics_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["agent_id"] == "a0"
        assert json.loads(lines[2])["step"] == 1

    def test_log_events(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_003")
        logger.log_events([
            {"event": "collapse", "step": 10, "shared_pool": 1.0},
        ])
        logger.log_events([
            {"event": "agent_deactivated", "step": 8, "agent_id": "a2"},
        ])

        events_path = logger.run_dir / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "collapse"
        assert json.loads(lines[1])["agent_id"] == "a2"

    def test_write_episode_summary(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_004")
        logger.write_episode_summary({
            "episode_length": 100,
            "termination_reason": "max_steps",
            "final_shared_pool": 42.5,
            "total_reward_per_agent": {"a0": 5.0, "a1": 3.2},
        })

        summary_path = logger.run_dir / "episode_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "written_at" in data
        assert data["episode_length"] == 100
        assert data["total_reward_per_agent"]["a0"] == 5.0

    def test_empty_records_skip_write(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_005")
        logger.log_step_metrics([])
        logger.log_events([])
        logger.write_episode_summary({})

        assert not (logger.run_dir / "metrics.jsonl").exists()
        assert not (logger.run_dir / "events.jsonl").exists()
        assert not (logger.run_dir / "episode_summary.json").exists()

    def test_run_dir_created(self, run_dir: Path):
        logger = RunLogger(run_dir, "test_run_006")
        assert logger.run_dir.exists()
        assert logger.run_dir.is_dir()
