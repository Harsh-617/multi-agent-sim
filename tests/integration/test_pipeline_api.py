"""Integration tests for the pipeline API endpoints.

The expensive ``run_pipeline`` function is monkeypatched to a fast stub so
tests run without GPU or long wall-clock time.  We verify the full
request/response contract for both endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_run_pipeline(pipelines_dir: Path):
    """Return a fast stub for ``run_pipeline`` that writes a real summary.json."""

    def _stub(
        config_id: str = "default",
        *,
        seed: int = 42,
        seeds: int = 3,
        episodes_per_seed: int = 2,
        max_steps=None,
        total_timesteps: int = 50_000,
        snapshot_every_timesteps: int = 10_000,
        max_league_members: int = 50,
        num_matches: int = 10,
        limit_sweeps=None,
        progress_callback=None,
        ppo_agent_dir=None,
        pipelines_dir=pipelines_dir,
        configs_dir=None,
        reports_dir=None,
    ) -> Path:
        from datetime import datetime, timezone
        import hashlib

        if progress_callback:
            for stage in ("loading_config", "training", "ratings", "robustness", "saving"):
                progress_callback(stage, "")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        h = hashlib.sha256(f"{config_id}:{seed}".encode()).hexdigest()[:8]
        pid = f"pipeline_{ts}_{h}"
        report_id = f"robust_stub_{ts}"

        out_dir = pipelines_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "pipeline_id": pid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_id": config_id,
            "config_hash": "abc123",
            "seed": seed,
            "training": {
                "total_timesteps": total_timesteps,
                "ppo_agent_dir": "storage/agents/ppo_shared",
                "snapshot_every_timesteps": snapshot_every_timesteps,
                "max_league_members": max_league_members,
            },
            "ratings": {
                "champion_id": "league_000001",
                "champion_rating": 1050.0,
                "num_members_rated": 1,
                "num_matches": num_matches,
            },
            "robustness": {
                "report_id": report_id,
                "report_dir": "storage/reports/" + report_id,
                "n_sweeps": 1,
                "n_policies": 4,
            },
            "report_id": report_id,
            "summary_path": str(out_dir / "summary.json"),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if progress_callback:
            progress_callback("done", pid)

        return out_dir

    return _stub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch pipeline_manager to isolated state and stub run_pipeline."""
    from backend.pipeline.pipeline_manager import pipeline_manager

    pipeline_manager.reset_state()

    # Patch run_pipeline inside routes module to avoid heavy computation
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    import backend.api.routes_pipeline as rp

    monkeypatch.setattr(rp, "run_pipeline", _make_stub_run_pipeline(pipelines_dir))

    yield

    pipeline_manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: POST /api/pipeline/run
# ---------------------------------------------------------------------------


class TestStartPipeline:
    def test_returns_pipeline_id(self, client: TestClient) -> None:
        resp = client.post("/api/pipeline/run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_id" in data
        assert isinstance(data["pipeline_id"], str)
        assert len(data["pipeline_id"]) > 0

    def test_conflict_when_running(self, client: TestClient) -> None:
        """Second request while one is active must return 409."""
        from backend.pipeline.pipeline_manager import pipeline_manager

        pipeline_manager.running = True
        pipeline_manager.pipeline_id = "already_running"

        resp = client.post("/api/pipeline/run", json={})
        assert resp.status_code == 409

    def test_default_params_accepted(self, client: TestClient) -> None:
        """Empty body (all defaults) must be accepted."""
        resp = client.post("/api/pipeline/run", json={})
        assert resp.status_code == 200

    def test_custom_params_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/api/pipeline/run",
            json={
                "config_id": "default",
                "seed": 7,
                "seeds": 2,
                "episodes_per_seed": 1,
                "total_timesteps": 1024,
                "snapshot_every_timesteps": 512,
                "max_league_members": 10,
                "num_matches": 3,
                "limit_sweeps": 2,
            },
        )
        assert resp.status_code == 200
        assert "pipeline_id" in resp.json()


# ---------------------------------------------------------------------------
# Tests: GET /api/pipeline/{pipeline_id}/status
# ---------------------------------------------------------------------------


class TestPipelineStatus:
    def test_status_unknown_id_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/pipeline/nonexistent_id/status")
        assert resp.status_code == 404

    def test_status_after_start_returns_pipeline_id(
        self, client: TestClient
    ) -> None:
        start = client.post("/api/pipeline/run", json={"total_timesteps": 256})
        assert start.status_code == 200
        pipeline_id = start.json()["pipeline_id"]

        status = client.get(f"/api/pipeline/{pipeline_id}/status")
        assert status.status_code == 200
        data = status.json()
        assert data["pipeline_id"] == pipeline_id
        assert "running" in data
        assert "stage" in data

    def test_status_contains_report_id_when_done(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After pipeline completes, status includes report_id and summary_path.

        We patch ``_run_pipeline_task`` to an immediate coroutine (no
        run_in_executor) so the task finishes during the endpoint's
        ``await asyncio.sleep(0)`` yield — no polling required.
        """
        import backend.api.routes_pipeline as rp

        report_id_stub = "robust_stub_00000000T000000Z"
        summary_dir = tmp_path / "pipelines" / "pipeline_test_abc"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / "summary.json"
        summary_file.write_text(
            json.dumps(
                {
                    "pipeline_id": "pipeline_test_abc",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "config_id": "default",
                    "config_hash": "abc123",
                    "seed": 42,
                    "training": {"total_timesteps": 256},
                    "ratings": {"champion_id": "league_000001"},
                    "robustness": {"report_id": report_id_stub},
                    "report_id": report_id_stub,
                    "summary_path": str(summary_file),
                }
            ),
            encoding="utf-8",
        )

        # Immediate coroutine: no awaits, completes during asyncio.sleep(0) yield
        async def _immediate_task(pm, pipeline_id, kwargs):
            pm.running = True
            pm.report_id = report_id_stub
            pm.summary_path = str(summary_file)
            pm.stage = "done"
            pm.running = False

        monkeypatch.setattr(rp, "_run_pipeline_task", _immediate_task)

        start = client.post("/api/pipeline/run", json={"total_timesteps": 256})
        assert start.status_code == 200
        pipeline_id = start.json()["pipeline_id"]

        # By the time POST returns, the immediate task has already completed
        resp = client.get(f"/api/pipeline/{pipeline_id}/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data.get("stage") == "done", f"Unexpected stage: {data}"
        assert "report_id" in data
        assert data["report_id"].startswith("robust_")
        assert "summary_path" in data
        assert Path(data["summary_path"]).exists()

    def test_status_running_flag_reflects_state(
        self, client: TestClient
    ) -> None:
        from backend.pipeline.pipeline_manager import pipeline_manager

        # Manually set to a known state (simulating mid-run)
        pipeline_manager.pipeline_id = "test_mid_run"
        pipeline_manager.running = True
        pipeline_manager.stage = "training"

        resp = client.get("/api/pipeline/test_mid_run/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert data["stage"] == "training"
