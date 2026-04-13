"""Integration tests for cooperative pipeline automation endpoints.

The expensive ``run_cooperative_pipeline`` function is monkeypatched to a fast
stub so tests run quickly without training or environment execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub for run_cooperative_pipeline
# ---------------------------------------------------------------------------


def _make_stub_run_coop_pipeline(pipelines_dir: Path):
    """Return a fast stub for ``run_cooperative_pipeline``."""

    def _stub(
        config_id: str = "default",
        *,
        seed: int = 42,
        seeds: int = 1,
        episodes_per_seed: int = 1,
        max_steps=None,
        total_timesteps: int = 50_000,
        snapshot_every_timesteps: int = 10_000,
        max_league_members: int = 50,
        num_matches: int = 10,
        limit_sweeps=None,
        progress_callback=None,
    ) -> Path:
        from datetime import datetime, timezone
        import hashlib

        if progress_callback:
            for stage in ("loading_config", "training", "ratings", "evaluating", "robustness", "saving"):
                progress_callback(stage, "")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        h = hashlib.sha256(f"{config_id}:{seed}".encode()).hexdigest()[:8]
        pid = f"cooperative_pipeline_{ts}_{h}"
        report_id = f"cooperative_eval_stub_{ts}"

        out_dir = pipelines_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "pipeline_id": pid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_id": config_id,
            "config_hash": h,
            "seed": seed,
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
def _reset_pipeline_manager():
    """Reset the cooperative pipeline manager before and after each test."""
    import backend.api.routes_cooperative_pipeline as rcp

    rcp.coop_pipeline_manager.reset_state()
    yield
    rcp.coop_pipeline_manager.reset_state()


@pytest.fixture(autouse=True)
def _isolate_pipelines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect pipelines directory and stub out heavy pipeline execution."""
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir()

    import backend.api.routes_cooperative_pipeline as rcp

    monkeypatch.setattr(rcp, "_PIPELINES_DIR", pipelines_dir)
    monkeypatch.setattr(
        rcp, "run_cooperative_pipeline", _make_stub_run_coop_pipeline(pipelines_dir)
    )
    return pipelines_dir


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def pipelines_dir(tmp_path: Path) -> Path:
    return tmp_path / "pipelines"


# ---------------------------------------------------------------------------
# POST /api/cooperative/pipeline/run
# ---------------------------------------------------------------------------


class TestCoopPipelineRun:
    def test_returns_pipeline_id(self, client: TestClient):
        resp = client.post(
            "/api/cooperative/pipeline/run",
            json={"seed": 42, "seeds": 1, "episodes_per_seed": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_id" in data
        assert isinstance(data["pipeline_id"], str)
        assert len(data["pipeline_id"]) > 0

    def test_second_call_returns_409_while_running(self, client: TestClient):
        import backend.api.routes_cooperative_pipeline as rcp

        rcp.coop_pipeline_manager.running = True
        rcp.coop_pipeline_manager.pipeline_id = "already_running"

        resp = client.post(
            "/api/cooperative/pipeline/run",
            json={"seed": 42, "seeds": 1, "episodes_per_seed": 1},
        )
        assert resp.status_code == 409

    def test_default_params_accepted(self, client: TestClient):
        """POST without any body uses defaults."""
        resp = client.post("/api/cooperative/pipeline/run", json={})
        assert resp.status_code == 200
        assert "pipeline_id" in resp.json()

    def test_custom_params_accepted(self, client: TestClient):
        resp = client.post(
            "/api/cooperative/pipeline/run",
            json={
                "config_id": "default",
                "seed": 7,
                "seeds": 1,
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

    def test_pipeline_id_is_unique_per_call(self, client: TestClient):
        import backend.api.routes_cooperative_pipeline as rcp

        resp1 = client.post("/api/cooperative/pipeline/run", json={})
        id1 = resp1.json()["pipeline_id"]

        rcp.coop_pipeline_manager.reset_state()

        resp2 = client.post("/api/cooperative/pipeline/run", json={})
        id2 = resp2.json()["pipeline_id"]

        assert id1 != id2


# ---------------------------------------------------------------------------
# GET /api/cooperative/pipeline/status/{pipeline_id}
# ---------------------------------------------------------------------------


class TestCoopPipelineStatus:
    def test_unknown_id_returns_404(self, client: TestClient):
        resp = client.get("/api/cooperative/pipeline/status/nonexistent_id")
        assert resp.status_code == 404

    def test_known_id_returns_status(self, client: TestClient):
        import backend.api.routes_cooperative_pipeline as rcp

        pid = "testpipeline01"
        rcp.coop_pipeline_manager.pipeline_id = pid
        rcp.coop_pipeline_manager.stage = "training"
        rcp.coop_pipeline_manager.running = True

        resp = client.get(f"/api/cooperative/pipeline/status/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_id"] == pid
        assert data["stage"] == "training"
        assert data["running"] is True

    def test_status_includes_error_when_present(self, client: TestClient):
        import backend.api.routes_cooperative_pipeline as rcp

        pid = "failed_pipeline"
        rcp.coop_pipeline_manager.pipeline_id = pid
        rcp.coop_pipeline_manager.stage = "error"
        rcp.coop_pipeline_manager.error = "Training failed: out of memory"
        rcp.coop_pipeline_manager.running = False

        resp = client.get(f"/api/cooperative/pipeline/status/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage"] == "error"
        assert "error" in data
        assert data["error"] == "Training failed: out of memory"

    def test_status_includes_report_id_when_done(self, client: TestClient):
        import backend.api.routes_cooperative_pipeline as rcp

        pid = "done_pipeline"
        rcp.coop_pipeline_manager.pipeline_id = pid
        rcp.coop_pipeline_manager.stage = "done"
        rcp.coop_pipeline_manager.running = False
        rcp.coop_pipeline_manager.report_id = "cooperative_eval_abc123_20260101T000000Z"

        resp = client.get(f"/api/cooperative/pipeline/status/{pid}")
        data = resp.json()
        assert data["report_id"] == "cooperative_eval_abc123_20260101T000000Z"

    def test_after_start_id_is_tracked(self, client: TestClient):
        resp = client.post("/api/cooperative/pipeline/run", json={})
        pid = resp.json()["pipeline_id"]

        status_resp = client.get(f"/api/cooperative/pipeline/status/{pid}")
        assert status_resp.status_code == 200
        assert status_resp.json()["pipeline_id"] == pid

    def test_status_done_after_immediate_task(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify report_id is populated after stub completes."""
        import backend.api.routes_cooperative_pipeline as rcp

        report_id_stub = "cooperative_eval_stub_20260101T000000Z"

        async def _immediate_task(pm, pipeline_id, kwargs):
            pm.running = True
            pm.report_id = report_id_stub
            pm.stage = "done"
            pm.running = False

        monkeypatch.setattr(rcp, "_run_coop_pipeline_task", _immediate_task)

        start = client.post("/api/cooperative/pipeline/run", json={})
        assert start.status_code == 200
        pid = start.json()["pipeline_id"]

        resp = client.get(f"/api/cooperative/pipeline/status/{pid}")
        data = resp.json()
        assert data.get("stage") == "done"
        assert data.get("report_id", "").startswith("cooperative_eval_")


# ---------------------------------------------------------------------------
# GET /api/cooperative/pipeline/runs
# ---------------------------------------------------------------------------


class TestCoopPipelineRuns:
    def test_empty_pipelines_dir_returns_empty_list(self, client: TestClient):
        resp = client.get("/api/cooperative/pipeline/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_pipeline_summaries(
        self, client: TestClient, pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import backend.api.routes_cooperative_pipeline as rcp

        monkeypatch.setattr(rcp, "_PIPELINES_DIR", pipelines_dir)

        run_dir = pipelines_dir / "cooperative_pipeline_20260401T000000Z_abc123"
        run_dir.mkdir(parents=True)
        summary = {
            "pipeline_id": "cooperative_pipeline_20260401T000000Z_abc123",
            "timestamp": "2026-04-01T00:00:00+00:00",
            "config_id": "default",
            "config_hash": "abc123",
            "report_id": "cooperative_eval_abc123_20260401T000000Z",
            "archetype": "cooperative",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

        resp = client.get("/api/cooperative/pipeline/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["pipeline_id"] == "cooperative_pipeline_20260401T000000Z_abc123"

    def test_non_cooperative_dirs_ignored(
        self, client: TestClient, pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import backend.api.routes_cooperative_pipeline as rcp

        monkeypatch.setattr(rcp, "_PIPELINES_DIR", pipelines_dir)

        other_dir = pipelines_dir / "mixed_pipeline_20260401T000000Z_xyz"
        other_dir.mkdir()
        (other_dir / "summary.json").write_text(
            json.dumps({"pipeline_id": "mixed_pipeline", "timestamp": ""})
        )

        resp = client.get("/api/cooperative/pipeline/runs")
        assert resp.json() == []

    def test_runs_sorted_newest_first(
        self, client: TestClient, pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import backend.api.routes_cooperative_pipeline as rcp

        monkeypatch.setattr(rcp, "_PIPELINES_DIR", pipelines_dir)

        for ts, suffix in [
            ("2026-01-01T00:00:00+00:00", "first"),
            ("2026-03-01T00:00:00+00:00", "second"),
            ("2026-02-01T00:00:00+00:00", "third"),
        ]:
            run_dir = pipelines_dir / f"cooperative_pipeline_{suffix}"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps({"pipeline_id": suffix, "timestamp": ts})
            )

        resp = client.get("/api/cooperative/pipeline/runs")
        data = resp.json()
        timestamps = [d["timestamp"] for d in data]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_dirs_without_summary_json_skipped(
        self, client: TestClient, pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import backend.api.routes_cooperative_pipeline as rcp

        monkeypatch.setattr(rcp, "_PIPELINES_DIR", pipelines_dir)

        (pipelines_dir / "cooperative_pipeline_nosummary").mkdir()

        resp = client.get("/api/cooperative/pipeline/runs")
        assert resp.json() == []
