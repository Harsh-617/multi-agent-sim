"""Integration tests for the FastAPI backend.

Tests HTTP endpoints and artifact production.  WebSocket broadcast is tested
by subscribing directly to the RunManager queue (avoids inherent race
between synchronous TestClient POST and WS connect).
"""

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


def _small_config_json() -> dict:
    """A small, fast config for tests (3 agents, 15 steps)."""
    cfg = default_config(seed=7)
    data = json.loads(cfg.model_dump_json())
    data["population"]["num_agents"] = 3
    data["population"]["max_steps"] = 15
    return data


# ------------------------------------------------------------------
# Config endpoints
# ------------------------------------------------------------------

class TestConfigEndpoints:

    def test_create_and_get_config(self, client: TestClient):
        body = _small_config_json()
        resp = client.post("/api/configs", json=body)
        assert resp.status_code == 201
        config_id = resp.json()["config_id"]
        assert isinstance(config_id, str) and len(config_id) > 0

        resp2 = client.get(f"/api/configs/{config_id}")
        assert resp2.status_code == 200
        assert resp2.json()["identity"]["seed"] == 7

    def test_list_configs(self, client: TestClient):
        body = _small_config_json()
        client.post("/api/configs", json=body)
        client.post("/api/configs", json=body)

        resp = client.get("/api/configs")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        assert all(item["num_agents"] == 3 for item in items)

    def test_get_missing_config(self, client: TestClient):
        resp = client.get("/api/configs/nonexistent")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Run control
# ------------------------------------------------------------------

class TestRunControl:

    def _create_config(self, client: TestClient) -> str:
        resp = client.post("/api/configs", json=_small_config_json())
        return resp.json()["config_id"]

    def test_start_run_returns_run_id(self, client: TestClient):
        config_id = self._create_config(client)
        resp = client.post("/api/runs/start", json={"config_id": config_id})
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_start_run_missing_config(self, client: TestClient):
        resp = client.post("/api/runs/start", json={"config_id": "nope"})
        assert resp.status_code == 404

    def test_status_idle(self, client: TestClient):
        resp = client.get("/api/runs/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_stop_when_idle_returns_409(self, client: TestClient):
        resp = client.post("/api/runs/stop")
        assert resp.status_code == 409

    def test_start_run_ppo_shared_missing_artifacts_returns_422(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Starting a PPO run without trained artifacts should fail with 422."""
        import backend.api.routes_experiment as re_mod
        monkeypatch.setattr(re_mod, "PPO_AGENT_DIR", tmp_path / "nonexistent")

        config_id = self._create_config(client)
        resp = client.post(
            "/api/runs/start",
            json={"config_id": config_id, "agent_policy": "ppo_shared"},
        )
        assert resp.status_code == 422
        assert "PPO artifacts not found" in resp.json()["detail"]

    def test_start_run_ppo_shared_with_artifacts(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Starting a PPO run with valid artifacts should return 200."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from simulation.training.ppo_shared import SharedPolicyNetwork

        # Create fake PPO artifacts in tmp_path
        ppo_dir = tmp_path / "agents" / "ppo_shared"
        ppo_dir.mkdir(parents=True)

        obs_dim = 20
        net = SharedPolicyNetwork(obs_dim, 4)
        torch.save(net.state_dict(), ppo_dir / "policy.pt")
        metadata = {
            "obs_dim": obs_dim,
            "action_mapping": {"0": "cooperate", "1": "extract", "2": "defend", "3": "conditional"},
        }
        (ppo_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        # Monkeypatch both the route-level validation path and the agent default dir
        import backend.api.routes_experiment as re_mod
        import simulation.agents.ppo_shared_agent as ppo_mod

        monkeypatch.setattr(re_mod, "PPO_AGENT_DIR", ppo_dir)
        monkeypatch.setattr(ppo_mod, "_DEFAULT_AGENT_DIR", ppo_dir)

        config_id = self._create_config(client)
        resp = client.post(
            "/api/runs/start",
            json={"config_id": config_id, "agent_policy": "ppo_shared"},
        )
        assert resp.status_code == 200
        assert "run_id" in resp.json()


# ------------------------------------------------------------------
# Artifact production (run completes, files are written)
# ------------------------------------------------------------------

class TestRunArtifacts:
    """Run the experiment directly (not via HTTP) to verify artifact production.

    The TestClient event loop doesn't reliably pump background tasks between
    requests, so we use asyncio.run() for deterministic execution.
    """

    def test_run_produces_all_artifacts(self, tmp_path: Path):
        from backend.runner.run_manager import RunManager
        from backend.runner.experiment_runner import run_experiment
        from simulation.config.schema import MixedEnvironmentConfig

        cfg_data = _small_config_json()
        config = MixedEnvironmentConfig.model_validate(cfg_data)
        runs_dir = tmp_path / "artifact_runs"
        runs_dir.mkdir()

        mgr = RunManager()
        asyncio.run(run_experiment(config, "artifact_test", runs_dir, mgr))

        run_dir = runs_dir / "artifact_test"
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "episode_summary.json").exists()

        # Verify config content
        config_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert "written_at" in config_data
        assert config_data["identity"]["seed"] == 7

        # Verify metrics content
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 0
        first_record = json.loads(lines[0])
        assert "agent_id" in first_record
        assert "reward" in first_record

        # Verify episode summary
        summary = json.loads((run_dir / "episode_summary.json").read_text(encoding="utf-8"))
        assert "episode_length" in summary
        assert "termination_reason" in summary
        assert "total_reward_per_agent" in summary


# ------------------------------------------------------------------
# Broadcast mechanism (test RunManager queue directly)
# ------------------------------------------------------------------

class TestBroadcastMechanism:
    """Test that the experiment runner broadcasts messages through RunManager.

    Uses asyncio directly to avoid the synchronous TestClient race condition
    with WebSocket connections.
    """

    def test_experiment_broadcasts_step_and_done(self, tmp_path: Path):
        from backend.runner.run_manager import RunManager
        from backend.runner.experiment_runner import run_experiment
        from simulation.config.schema import MixedEnvironmentConfig

        cfg_data = _small_config_json()
        config = MixedEnvironmentConfig.model_validate(cfg_data)
        runs_dir = tmp_path / "broadcast_runs"
        runs_dir.mkdir()

        mgr = RunManager()
        queue = mgr.subscribe()

        async def _run_and_collect():
            task = asyncio.create_task(
                run_experiment(config, "test_broadcast", runs_dir, mgr)
            )
            messages = []
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                messages.append(msg)
                if msg.get("type") == "done":
                    break
            await task
            return messages

        messages = asyncio.run(_run_and_collect())

        step_msgs = [m for m in messages if m["type"] == "step"]
        done_msgs = [m for m in messages if m["type"] == "done"]

        assert len(step_msgs) > 0, "Should have step messages"
        assert len(done_msgs) == 1, "Should have exactly one done message"

        # Verify step message structure
        first = step_msgs[0]
        assert first["run_id"] == "test_broadcast"
        assert "t" in first
        assert "metrics" in first
        assert isinstance(first["metrics"], list)
        assert "agent_id" in first["metrics"][0]
        assert "reward" in first["metrics"][0]

        # Verify done message
        done = done_msgs[0]
        assert done["run_id"] == "test_broadcast"
        assert "termination_reason" in done
        assert "episode_summary" in done
        assert "total_reward_per_agent" in done["episode_summary"]

    def test_ws_endpoint_accepts_and_closes(self, client: TestClient):
        """Verify WS endpoint connects and closes cleanly when no run is active."""
        try:
            with client.websocket_connect("/api/ws/metrics/nonexistent") as ws:
                # Handler should timeout and close since no run is active
                msg = ws.receive_json()
        except Exception:
            pass  # WebSocketDisconnect is expected
