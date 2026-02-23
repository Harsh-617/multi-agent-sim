"""Integration tests for run history, replay, and benchmark endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation.config.defaults import default_config


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect storage directories to a temp folder for test isolation."""
    configs_dir = tmp_path / "configs"
    runs_dir = tmp_path / "runs"
    configs_dir.mkdir()
    runs_dir.mkdir()

    import backend.api.routes_config as rc
    import backend.api.routes_experiment as re_mod
    import backend.api.routes_history as rh_mod

    monkeypatch.setattr(rc, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(rh_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(rh_mod, "CONFIGS_DIR", configs_dir)

    from backend.runner.run_manager import manager
    manager.reset_state()

    yield

    manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    return tmp_path / "configs"


def _small_config_json() -> dict:
    cfg = default_config(seed=7)
    data = json.loads(cfg.model_dump_json())
    data["population"]["num_agents"] = 3
    data["population"]["max_steps"] = 15
    return data


def _create_run_artifacts(runs_dir: Path, run_id: str) -> None:
    """Run experiment directly to produce artifacts."""
    from backend.runner.run_manager import RunManager
    from backend.runner.experiment_runner import run_experiment
    from simulation.config.schema import MixedEnvironmentConfig

    cfg_data = _small_config_json()
    config = MixedEnvironmentConfig.model_validate(cfg_data)
    mgr = RunManager()
    asyncio.run(run_experiment(config, run_id, runs_dir, mgr))


# ------------------------------------------------------------------
# Run History
# ------------------------------------------------------------------

class TestRunHistory:

    def test_list_runs_empty(self, client: TestClient):
        resp = client.get("/api/runs/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_runs_after_experiment(self, client: TestClient, runs_dir: Path):
        _create_run_artifacts(runs_dir, "hist_test_1")

        resp = client.get("/api/runs/history")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["run_id"] == "hist_test_1"
        assert items[0]["termination_reason"] is not None
        assert items[0]["episode_length"] is not None

    def test_get_run_detail(self, client: TestClient, runs_dir: Path):
        _create_run_artifacts(runs_dir, "detail_test")

        resp = client.get("/api/runs/detail_test/detail")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "detail_test"
        assert data["episode_summary"] is not None
        assert "total_reward_per_agent" in data["episode_summary"]

    def test_get_run_detail_missing(self, client: TestClient):
        resp = client.get("/api/runs/nonexistent/detail")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Replay (SSE)
# ------------------------------------------------------------------

class TestReplay:

    def test_replay_streams_metrics(self, client: TestClient, runs_dir: Path):
        _create_run_artifacts(runs_dir, "replay_test")

        resp = client.get("/api/runs/replay_test/replay")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) >= 2  # at least some steps + done
        step_events = [e for e in events if e["type"] == "step"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(step_events) > 0
        assert len(done_events) == 1

        # Verify step event structure
        first = step_events[0]
        assert first["run_id"] == "replay_test"
        assert "t" in first
        assert "metrics" in first
        assert len(first["metrics"]) > 0
        assert "agent_id" in first["metrics"][0]

        # Verify done event
        done = done_events[0]
        assert "termination_reason" in done
        assert "episode_summary" in done

    def test_replay_missing_run(self, client: TestClient):
        resp = client.get("/api/runs/nonexistent/replay")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------

class TestBenchmark:

    def test_benchmark_compares_policies(
        self, client: TestClient, configs_dir: Path, runs_dir: Path
    ):
        # Create a config via API
        resp = client.post("/api/configs", json=_small_config_json())
        assert resp.status_code == 201
        config_id = resp.json()["config_id"]

        # Run benchmark with two policies
        resp = client.post("/api/benchmark", json={
            "config_id": config_id,
            "agent_policies": ["random", "always_cooperate"],
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["config_id"] == config_id
        assert len(data["results"]) == 2

        for result in data["results"]:
            assert result["agent_policy"] in ["random", "always_cooperate"]
            assert result["mean_reward"] is not None
            assert result["final_shared_pool"] is not None
            assert result["termination_reason"] is not None
            assert result["episode_length"] is not None

    def test_benchmark_missing_config(self, client: TestClient):
        resp = client.post("/api/benchmark", json={
            "config_id": "nonexistent",
            "agent_policies": ["random"],
        })
        assert resp.status_code == 404
