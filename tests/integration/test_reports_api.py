"""Integration tests for the reports browsing API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect reports root to a temp folder and seed a fake report."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    import backend.api.routes_reports as rr

    monkeypatch.setattr(rr, "_REPORTS_ROOT", reports_dir)

    # Create a fake eval report
    eval_dir = reports_dir / "eval_abc123_20260101T000000Z"
    eval_dir.mkdir()
    (eval_dir / "report.json").write_text(
        json.dumps(
            {
                "report_id": "eval_abc123_20260101T000000Z",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "config_hash": "abc123",
                "seeds": [1],
                "episodes_per_seed": 1,
                "config": {},
                "results": [
                    {
                        "policy_name": "random",
                        "source": "baseline",
                        "available": True,
                        "mean_total_reward": 10.0,
                        "std_total_reward": 1.0,
                        "mean_final_shared_pool": 50.0,
                        "collapse_rate": 0.0,
                        "mean_episode_length": 100,
                        "n_episodes": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Create a fake robust report
    robust_dir = reports_dir / "robust_def456_20260102T000000Z"
    robust_dir.mkdir()
    (robust_dir / "report.json").write_text(
        json.dumps(
            {
                "report_id": "robust_def456_20260102T000000Z",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "config_hash": "def456",
                "config": {},
                "metadata": {"seeds": [1], "episodes_per_seed": 1, "sweep_count": 1},
                "per_sweep_results": {},
                "per_policy_robustness": {},
            }
        ),
        encoding="utf-8",
    )

    yield


@pytest.fixture()
def client():
    from backend.main import app

    return TestClient(app)


class TestListReports:
    def test_returns_both_reports(self, client: TestClient):
        resp = client.get("/api/reports")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = {r["report_id"] for r in data}
        assert "eval_abc123_20260101T000000Z" in ids
        assert "robust_def456_20260102T000000Z" in ids

    def test_sorted_newest_first(self, client: TestClient):
        resp = client.get("/api/reports")
        data = resp.json()
        assert data[0]["report_id"] == "robust_def456_20260102T000000Z"
        assert data[1]["report_id"] == "eval_abc123_20260101T000000Z"

    def test_kind_classification(self, client: TestClient):
        resp = client.get("/api/reports")
        data = resp.json()
        kinds = {r["report_id"]: r["kind"] for r in data}
        assert kinds["eval_abc123_20260101T000000Z"] == "eval"
        assert kinds["robust_def456_20260102T000000Z"] == "robust"


class TestGetReport:
    def test_returns_eval_report(self, client: TestClient):
        resp = client.get("/api/reports/eval_abc123_20260101T000000Z")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == "eval_abc123_20260101T000000Z"
        assert data["kind"] == "eval"
        assert len(data["results"]) == 1
        assert data["results"][0]["policy_name"] == "random"

    def test_returns_robust_report(self, client: TestClient):
        resp = client.get("/api/reports/robust_def456_20260102T000000Z")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == "robust_def456_20260102T000000Z"
        assert data["kind"] == "robust"

    def test_404_for_missing(self, client: TestClient):
        resp = client.get("/api/reports/nonexistent_report")
        assert resp.status_code == 404

    def test_404_for_missing_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ):
        """A folder exists but report.json is missing."""
        import backend.api.routes_reports as rr

        reports_dir = rr._REPORTS_ROOT
        empty_dir = reports_dir / "eval_empty_20260103T000000Z"
        empty_dir.mkdir()

        resp = client.get("/api/reports/eval_empty_20260103T000000Z")
        assert resp.status_code == 404


class TestGetStrategies:
    def test_returns_strategy_analysis_for_eval(self, client: TestClient):
        resp = client.get(
            "/api/reports/eval_abc123_20260101T000000Z/strategies"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert "clusters" in data
        assert "labels" in data
        assert "summaries" in data
        # The single policy should appear in features and clusters.
        assert "random" in data["features"]
        assert "random" in data["clusters"]

    def test_returns_strategy_analysis_for_robust(self, client: TestClient):
        """Robust report with no policies still returns valid empty-ish structure."""
        resp = client.get(
            "/api/reports/robust_def456_20260102T000000Z/strategies"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["features"] == {}
        assert data["clusters"] == {}

    def test_404_for_missing_report(self, client: TestClient):
        resp = client.get("/api/reports/nonexistent/strategies")
        assert resp.status_code == 404
