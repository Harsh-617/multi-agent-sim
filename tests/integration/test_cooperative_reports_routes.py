"""Integration tests for cooperative evaluation and robustness report endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_report(reports_dir: Path, report_id: str | None = None) -> str:
    """Create a fake cooperative eval report folder. Returns report_id."""
    ts = "20260101T000000Z"
    cfg_hash = "aabbcc112233"
    rid = report_id or f"cooperative_eval_{cfg_hash}_{ts}"
    d = reports_dir / rid
    d.mkdir(parents=True, exist_ok=True)

    summary = {
        "report_id": rid,
        "kind": "cooperative_eval",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "config_hash": cfg_hash,
        "policy_name": "cooperative_agent",
        "num_seeds": 1,
        "episodes_per_seed": 1,
        "config": {},
        "summary": {
            "mean_completion_ratio": 0.72,
            "worst_case_completion_ratio": 0.55,
            "mean_group_efficiency_ratio": 0.68,
            "mean_effort_utilization": 0.80,
            "mean_system_stress": 0.30,
            "free_rider_fraction": 0.0,
            "effort_gini_coefficient": 0.0,
            "mean_return": 1.5,
        },
        "per_seed": [
            {
                "seed": 1000,
                "mean_completion_ratio": 0.72,
                "mean_group_efficiency_ratio": 0.68,
                "mean_effort_utilization": 0.80,
                "mean_system_stress": 0.30,
                "mean_return": 1.5,
            }
        ],
    }
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return rid


def _make_robust_report(reports_dir: Path, report_id: str | None = None) -> str:
    """Create a fake cooperative robustness report folder. Returns report_id."""
    ts = "20260102T000000Z"
    cfg_hash = "ddeeff445566"
    rid = report_id or f"cooperative_robust_{cfg_hash}_{ts}"
    d = reports_dir / rid
    d.mkdir(parents=True, exist_ok=True)

    summary = {
        "report_id": rid,
        "kind": "cooperative_robust",
        "timestamp": "2026-01-02T00:00:00+00:00",
        "config_hash": cfg_hash,
        "config": {},
        "metadata": {
            "sweeps": ["arrival_low_arrival", "difficulty_easy"],
            "sweep_count": 2,
            "seeds": [0],
            "episodes_per_seed": 1,
            "policy_name": "cooperative_champion",
        },
        "per_sweep_results": {
            "arrival_low_arrival": {
                "sweep_name": "arrival_low_arrival",
                "mean_completion_ratio": 0.70,
                "worst_case_completion_ratio": 0.60,
                "n_episodes": 1,
                "policy": "cooperative_champion",
            },
            "difficulty_easy": {
                "sweep_name": "difficulty_easy",
                "mean_completion_ratio": 0.80,
                "worst_case_completion_ratio": 0.75,
                "n_episodes": 1,
                "policy": "cooperative_champion",
            },
        },
        "per_policy_robustness": {
            "cooperative_champion": {
                "policy_name": "cooperative_champion",
                "mean_completion_ratio": 0.75,
                "worst_case_completion_ratio": 0.60,
                "robustness_score": 0.705,
                "n_sweeps_evaluated": 2,
            }
        },
    }
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return rid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect reports root to a temp folder."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    import backend.api.routes_cooperative_reports as rcr

    monkeypatch.setattr(rcr, "_REPORTS_ROOT", reports_dir)
    yield reports_dir


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/cooperative/reports
# ---------------------------------------------------------------------------


class TestListCoopReports:
    def test_empty_dir_returns_empty_list(self, client: TestClient):
        resp = client.get("/api/cooperative/reports")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_eval_reports(self, client: TestClient, reports_dir: Path):
        _make_eval_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["kind"] == "cooperative_eval"

    def test_returns_robust_reports(self, client: TestClient, reports_dir: Path):
        _make_robust_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["kind"] == "cooperative_robust"

    def test_returns_both_kinds(self, client: TestClient, reports_dir: Path):
        _make_eval_report(reports_dir)
        _make_robust_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        data = resp.json()
        assert len(data) == 2
        kinds = {d["kind"] for d in data}
        assert "cooperative_eval" in kinds
        assert "cooperative_robust" in kinds

    def test_ignores_non_cooperative_dirs(self, client: TestClient, reports_dir: Path):
        # Create non-cooperative directories
        (reports_dir / "eval_abc123_20260101T000000Z").mkdir()
        (reports_dir / "robust_xyz_20260101T000000Z").mkdir()
        (reports_dir / "mixed_something").mkdir()

        resp = client.get("/api/cooperative/reports")
        assert resp.json() == []

    def test_report_list_items_have_required_keys(self, client: TestClient, reports_dir: Path):
        _make_eval_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        item = resp.json()[0]
        for key in ("report_id", "kind", "config_hash", "timestamp"):
            assert key in item

    def test_reports_sorted_newest_first(self, client: TestClient, reports_dir: Path):
        _make_eval_report(reports_dir, "cooperative_eval_aaa_20260101T000000Z")
        _make_robust_report(reports_dir, "cooperative_robust_bbb_20260103T000000Z")

        resp = client.get("/api/cooperative/reports")
        data = resp.json()
        timestamps = [d["timestamp"] for d in data if d["timestamp"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_dirs_without_summary_json_excluded(self, client: TestClient, reports_dir: Path):
        (reports_dir / "cooperative_eval_nosummary_20260101T000000Z").mkdir()
        resp = client.get("/api/cooperative/reports")
        assert resp.json() == []

    def test_eval_report_includes_mean_completion_ratio(
        self, client: TestClient, reports_dir: Path
    ):
        _make_eval_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        data = resp.json()
        assert data[0]["mean_completion_ratio"] == pytest.approx(0.72)

    def test_robust_report_includes_robustness_score(
        self, client: TestClient, reports_dir: Path
    ):
        _make_robust_report(reports_dir)
        resp = client.get("/api/cooperative/reports")
        data = resp.json()
        assert data[0]["robustness_score"] == pytest.approx(0.705)


# ---------------------------------------------------------------------------
# GET /api/cooperative/reports/{report_id}
# ---------------------------------------------------------------------------


class TestGetCoopReport:
    def test_returns_eval_report_detail(self, client: TestClient, reports_dir: Path):
        rid = _make_eval_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == rid
        assert data["kind"] == "cooperative_eval"

    def test_returns_robust_report_detail(self, client: TestClient, reports_dir: Path):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == rid
        assert data["kind"] == "cooperative_robust"

    def test_unknown_report_returns_404(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/cooperative_eval_nonexistent_20260101T000000Z"
        )
        assert resp.status_code == 404

    def test_invalid_report_id_returns_400(self, client: TestClient):
        # report_id with special characters → unsafe
        resp = client.get(
            "/api/cooperative/reports/cooperative_eval_../../etc/passwd"
        )
        assert resp.status_code in (400, 422, 404)

    def test_non_cooperative_report_id_returns_400(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/eval_abc123_20260101T000000Z"
        )
        assert resp.status_code == 400

    def test_eval_report_contains_summary_section(
        self, client: TestClient, reports_dir: Path
    ):
        rid = _make_eval_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}")
        data = resp.json()
        assert "summary" in data
        assert "mean_completion_ratio" in data["summary"]


# ---------------------------------------------------------------------------
# GET /api/cooperative/reports/{report_id}/robustness
# ---------------------------------------------------------------------------


class TestGetCoopReportRobustness:
    def test_returns_heatmap_data(self, client: TestClient, reports_dir: Path):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/robustness")
        assert resp.status_code == 200
        data = resp.json()
        assert "sweep_names" in data
        assert "policies" in data
        assert "heatmap" in data

    def test_sweep_names_match_report(self, client: TestClient, reports_dir: Path):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/robustness")
        data = resp.json()
        # Report has 2 sweeps
        assert set(data["sweep_names"]) == {"arrival_low_arrival", "difficulty_easy"}

    def test_heatmap_has_per_policy_values(self, client: TestClient, reports_dir: Path):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/robustness")
        data = resp.json()
        assert "cooperative_champion" in data["heatmap"]

    def test_eval_report_returns_empty_heatmap(
        self, client: TestClient, reports_dir: Path
    ):
        rid = _make_eval_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/robustness")
        assert resp.status_code == 200
        data = resp.json()
        # Eval reports have no per_sweep_results
        assert data["sweep_names"] == []
        assert data["policies"] == []

    def test_unknown_report_returns_404(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/cooperative_robust_notfound_20260101T000000Z/robustness"
        )
        assert resp.status_code == 404

    def test_non_cooperative_id_returns_400(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/eval_abc123_20260101T000000Z/robustness"
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/cooperative/reports/{report_id}/strategies
# ---------------------------------------------------------------------------


class TestGetCoopReportStrategies:
    def test_returns_strategy_data(self, client: TestClient, reports_dir: Path):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert "clusters" in data
        assert "labels" in data

    def test_eval_report_without_per_policy_returns_empty(
        self, client: TestClient, reports_dir: Path
    ):
        rid = _make_eval_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/strategies")
        assert resp.status_code == 200
        data = resp.json()
        # Eval report has no per_policy_robustness
        assert data["clusters"] == {}
        assert data["features"] == {}

    def test_robust_report_clusters_all_policies(
        self, client: TestClient, reports_dir: Path
    ):
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/strategies")
        data = resp.json()
        # cooperative_champion should be in clusters
        assert "cooperative_champion" in data["clusters"]
        assert "cooperative_champion" in data["features"]

    def test_labels_are_valid_strategy_names(
        self, client: TestClient, reports_dir: Path
    ):
        valid_labels = {
            "Dedicated Specialist", "Adaptive Generalist", "Free Rider",
            "Opportunist", "Overcontributor",
        }
        rid = _make_robust_report(reports_dir)
        resp = client.get(f"/api/cooperative/reports/{rid}/strategies")
        data = resp.json()
        for label in data["labels"].values():
            assert label in valid_labels, f"Unexpected label: {label!r}"

    def test_unknown_report_returns_404(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/cooperative_robust_notfound_20260101T000000Z/strategies"
        )
        assert resp.status_code == 404

    def test_non_cooperative_id_returns_400(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/reports/eval_abc123_20260101T000000Z/strategies"
        )
        assert resp.status_code == 400
