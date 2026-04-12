"""Integration tests for cooperative archetype API routes.

Covers:
  - GET /api/cooperative/runs  — returns list
  - GET /api/cooperative/runs/{run_id}  — returns correct schema
  - GET /api/cooperative/runs/{run_id}/summary  — returns all required fields
  - GET /api/cooperative/runs/{run_id}/replay  — streams valid JSONL (SSE)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.runner.cooperative_experiment_runner import run_cooperative_experiment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect RUNS_DIR so test artifacts are isolated."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    import backend.api.routes_cooperative as rc

    monkeypatch.setattr(rc, "RUNS_DIR", runs_dir)

    yield runs_dir


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def _small_config(seed: int = 7) -> CooperativeEnvironmentConfig:
    cfg = default_cooperative_config(seed=seed)
    data = json.loads(cfg.model_dump_json())
    data["population"]["num_agents"] = 2
    data["population"]["max_steps"] = 10
    return CooperativeEnvironmentConfig.model_validate(data)


def _create_run(
    tmp_path: Path,
    seed: int = 7,
    run_id: str | None = None,
) -> tuple[str, dict]:
    """Create a real cooperative run in tmp_path and return (run_id, summary)."""
    import backend.api.routes_cooperative as rc

    runs_dir = rc.RUNS_DIR
    rid = run_id or f"coop_test_{seed}"
    config = _small_config(seed=seed)
    summary = run_cooperative_experiment(
        config, rid, runs_dir, None, agent_policy="random"
    )
    return rid, summary


# ---------------------------------------------------------------------------
# GET /api/cooperative/runs
# ---------------------------------------------------------------------------

class TestListCooperativeRuns:
    def test_empty_list_when_no_runs(self, client: TestClient):
        resp = client.get("/api/cooperative/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_cooperative_run(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=1)
        resp = client.get("/api/cooperative/runs")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["run_id"] == run_id

    def test_returns_multiple_runs(self, client: TestClient, tmp_path: Path):
        _create_run(tmp_path, seed=1, run_id="coop_a")
        _create_run(tmp_path, seed=2, run_id="coop_b")
        resp = client.get("/api/cooperative/runs")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2

    def test_list_items_have_expected_fields(self, client: TestClient, tmp_path: Path):
        _create_run(tmp_path, seed=3)
        resp = client.get("/api/cooperative/runs")
        item = resp.json()[0]
        for key in ("run_id", "num_agents", "max_steps", "agent_policy", "termination_reason"):
            assert key in item, f"Missing field: {key}"


# ---------------------------------------------------------------------------
# GET /api/cooperative/runs/{run_id}
# ---------------------------------------------------------------------------

class TestGetCooperativeRun:
    def test_returns_404_for_missing_run(self, client: TestClient):
        resp = client.get("/api/cooperative/runs/nonexistent123")
        assert resp.status_code == 404

    def test_returns_run_detail(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=10)
        resp = client.get(f"/api/cooperative/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id

    def test_run_detail_contains_episode_summary(self, client: TestClient, tmp_path: Path):
        run_id, original_summary = _create_run(tmp_path, seed=11)
        resp = client.get(f"/api/cooperative/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_summary" in data
        ep = data["episode_summary"]
        assert ep is not None
        assert "completion_ratio" in ep
        assert "termination_reason" in ep

    def test_run_detail_schema_has_num_agents(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=12)
        resp = client.get(f"/api/cooperative/runs/{run_id}")
        data = resp.json()
        assert data.get("num_agents") == 2  # from _small_config

    def test_invalid_run_id_returns_400(self, client: TestClient):
        resp = client.get("/api/cooperative/runs/../../etc/passwd")
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# GET /api/cooperative/runs/{run_id}/summary
# ---------------------------------------------------------------------------

class TestGetCooperativeRunSummary:
    REQUIRED_FIELDS = {
        "episode_length",
        "termination_reason",
        "completion_ratio",
        "total_tasks_arrived",
        "total_tasks_completed",
        "final_backlog_level",
        "final_system_stress",
        "mean_system_stress",
        "collapse_occurred",
        "total_reward_per_agent",
        # Step E extended
        "group_efficiency_ratio",
        "free_rider_count",
        "free_rider_fraction",
        "effort_gini_coefficient",
        "agent_metrics",
    }

    def test_returns_all_required_fields(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=20)
        resp = client.get(f"/api/cooperative/runs/{run_id}/summary")
        assert resp.status_code == 200
        data = resp.json()
        missing = self.REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Missing summary fields: {missing}"

    def test_returns_404_for_missing_run(self, client: TestClient):
        resp = client.get("/api/cooperative/runs/doesnotexist/summary")
        assert resp.status_code == 404

    def test_completion_ratio_bounded(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=21)
        resp = client.get(f"/api/cooperative/runs/{run_id}/summary")
        data = resp.json()
        v = data["completion_ratio"]
        assert 0.0 <= v <= 1.0

    def test_agent_metrics_keys_present(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=22)
        resp = client.get(f"/api/cooperative/runs/{run_id}/summary")
        data = resp.json()
        agent_metrics = data.get("agent_metrics", {})
        assert len(agent_metrics) > 0
        for aid, am in agent_metrics.items():
            for key in ("effort_utilization", "idle_rate", "role_stability"):
                assert key in am, f"Agent {aid} missing key: {key}"


# ---------------------------------------------------------------------------
# GET /api/cooperative/runs/{run_id}/replay (SSE)
# ---------------------------------------------------------------------------

class TestCooperativeReplay:
    def test_returns_404_for_missing_run(self, client: TestClient):
        resp = client.get("/api/cooperative/runs/nope123/replay")
        assert resp.status_code == 404

    def test_streams_valid_jsonl(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=30)
        resp = client.get(
            f"/api/cooperative/runs/{run_id}/replay",
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200

        messages = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                payload = line[6:]
                messages.append(json.loads(payload))

        assert len(messages) >= 1, "Expected at least the 'done' message"

        # Every message must have a valid type
        for msg in messages:
            assert msg.get("type") in ("step", "done"), f"Bad msg type: {msg}"

    def test_replay_contains_done_message(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=31)
        resp = client.get(f"/api/cooperative/runs/{run_id}/replay")
        assert resp.status_code == 200

        last_msg = None
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                last_msg = json.loads(line[6:])

        assert last_msg is not None
        assert last_msg.get("type") == "done"

    def test_replay_step_messages_have_metrics(self, client: TestClient, tmp_path: Path):
        run_id, _ = _create_run(tmp_path, seed=32)
        resp = client.get(f"/api/cooperative/runs/{run_id}/replay")

        step_messages = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                msg = json.loads(line[6:])
                if msg.get("type") == "step":
                    step_messages.append(msg)

        assert len(step_messages) > 0, "Expected step messages in replay"
        first = step_messages[0]
        assert "metrics" in first
        assert isinstance(first["metrics"], list)
        assert len(first["metrics"]) > 0

        # Each metric record must have cooperative-specific keys
        record = first["metrics"][0]
        for key in ("step", "agent_id", "reward", "backlog_level", "system_stress"):
            assert key in record, f"Missing key in metric record: {key}"
