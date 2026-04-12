"""Integration tests for Cooperative archetype API routes and artifact production."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation.config.cooperative_defaults import default_cooperative_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect storage directories to a temp folder for test isolation."""
    configs_dir = tmp_path / "configs"
    runs_dir = tmp_path / "runs"
    configs_dir.mkdir()
    runs_dir.mkdir()

    import backend.api.routes_config as rc
    import backend.api.routes_experiment as re_mod

    monkeypatch.setattr(rc, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "RUNS_DIR", runs_dir)

    from backend.runner.run_manager import manager
    manager.reset_state()

    yield

    manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def _cooperative_config_dict(seed: int = 7) -> dict:
    """Return a small, fast cooperative config dict."""
    cfg = default_cooperative_config(seed=seed)
    data = json.loads(cfg.model_dump_json())
    # Make it faster for tests
    data["population"]["num_agents"] = 2
    data["population"]["max_steps"] = 10
    return data


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

class TestCooperativeConfigEndpoints:
    def test_post_cooperative_config_returns_201(self, client: TestClient):
        body = _cooperative_config_dict()
        resp = client.post("/api/configs", json=body)
        assert resp.status_code == 201
        config_id = resp.json()["config_id"]
        assert isinstance(config_id, str) and len(config_id) > 0

    def test_get_cooperative_config_returns_correct_data(self, client: TestClient):
        body = _cooperative_config_dict(seed=99)
        resp = client.post("/api/configs", json=body)
        config_id = resp.json()["config_id"]

        resp2 = client.get(f"/api/configs/{config_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["identity"]["environment_type"] == "cooperative"
        assert data["identity"]["seed"] == 99

    def test_list_configs_includes_cooperative(self, client: TestClient):
        body = _cooperative_config_dict()
        client.post("/api/configs", json=body)
        client.post("/api/configs", json=body)

        resp = client.get("/api/configs")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        # Verify num_agents and max_steps are readable from cooperative config
        assert all(item["num_agents"] == 2 for item in items)
        assert all(item["max_steps"] == 10 for item in items)


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------

class TestCooperativeRunControl:
    def _create_config(self, client: TestClient) -> str:
        resp = client.post("/api/configs", json=_cooperative_config_dict())
        assert resp.status_code == 201
        return resp.json()["config_id"]

    def test_start_cooperative_run_returns_run_id(self, client: TestClient):
        config_id = self._create_config(client)
        resp = client.post("/api/runs/start", json={"config_id": config_id})
        assert resp.status_code == 200
        assert "run_id" in resp.json()


# ---------------------------------------------------------------------------
# Artifact production
# ---------------------------------------------------------------------------

class TestCooperativeRunArtifacts:
    """Run a cooperative experiment directly to verify artifact production."""

    def test_run_completes_and_episode_summary_is_saved_to_storage(
        self, tmp_path: Path
    ):
        from simulation.runner.cooperative_experiment_runner import (
            run_cooperative_experiment,
        )
        from simulation.config.cooperative_schema import CooperativeEnvironmentConfig

        data = _cooperative_config_dict(seed=42)
        config = CooperativeEnvironmentConfig.model_validate(data)
        runs_dir = tmp_path / "coop_runs"
        runs_dir.mkdir()

        summary = run_cooperative_experiment(
            config, "coop_test_run", runs_dir, None, agent_policy="random"
        )

        run_dir = runs_dir / "coop_test_run"

        # Config snapshot
        assert (run_dir / "config.json").exists()
        config_data = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        assert config_data["identity"]["environment_type"] == "cooperative"
        assert "written_at" in config_data

        # Metrics
        assert (run_dir / "metrics.jsonl").exists()
        lines = (
            (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
        )
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert "agent_id" in first
        assert "reward" in first

        # Episode summary
        assert (run_dir / "episode_summary.json").exists()
        ep_summary = json.loads(
            (run_dir / "episode_summary.json").read_text(encoding="utf-8")
        )
        assert "episode_length" in ep_summary
        assert "termination_reason" in ep_summary
        assert "total_reward_per_agent" in ep_summary

        # Summary returned from the function also has expected keys
        assert "episode_length" in summary
        assert "termination_reason" in summary
        assert "total_reward_per_agent" in summary
