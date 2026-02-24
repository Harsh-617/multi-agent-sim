"""Integration tests for the GET /api/league/evolution endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all storage roots to tmp_path for test isolation."""
    configs_dir = tmp_path / "configs"
    league_dir = tmp_path / "league"
    reports_dir = tmp_path / "reports"
    configs_dir.mkdir()
    league_dir.mkdir()
    reports_dir.mkdir()

    import backend.api.routes_config as rc
    import backend.api.routes_experiment as re_mod
    import backend.api.routes_league as rl
    from simulation.league.registry import LeagueRegistry

    isolated_registry = LeagueRegistry(league_dir)
    monkeypatch.setattr(rl, "_registry", isolated_registry)
    monkeypatch.setattr(rl, "LEAGUE_ROOT", league_dir)
    monkeypatch.setattr(rl, "RATINGS_PATH", league_dir / "ratings.json")
    monkeypatch.setattr(rl, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(rl, "PPO_AGENT_DIR", tmp_path / "ppo_shared")
    monkeypatch.setattr(rl, "REPORTS_DIR", reports_dir)

    monkeypatch.setattr(rc, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(re_mod, "LEAGUE_ROOT", league_dir)

    from backend.runner.run_manager import manager

    manager.reset_state()
    yield
    manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def league_dir(tmp_path: Path) -> Path:
    return tmp_path / "league"


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fake_member(
    league_dir: Path,
    member_id: str,
    *,
    parent_id: str | None = None,
    created_at: str = "2025-01-01T00:00:00+00:00",
    notes: str | None = None,
) -> None:
    member_dir = league_dir / member_id
    member_dir.mkdir(parents=True, exist_ok=True)
    (member_dir / "policy.pt").write_bytes(b"FAKE_WEIGHTS")
    meta = {
        "member_id": member_id,
        "algo": "test",
        "obs_dim": 33,
        "parent_id": parent_id,
        "created_at": created_at,
        "notes": notes or f"Test member {member_id}",
    }
    (member_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_ratings(league_dir: Path, ratings: list[dict]) -> None:
    (league_dir / "ratings.json").write_text(json.dumps(ratings), encoding="utf-8")


def _write_robustness_report(reports_dir: Path, folder_name: str, rob_score: float) -> None:
    """Write a minimal robustness report with a league_champion entry."""
    report_dir = reports_dir / folder_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_id": folder_name,
        "per_policy_robustness": {
            "league_champion": {
                "policy_name": "league_champion",
                "overall_mean_reward": rob_score,
                "worst_case_mean_reward": rob_score,
                "robustness_score": rob_score,
                "collapse_rate_overall": 0.0,
                "n_sweeps_evaluated": 2,
            }
        },
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestLeagueEvolution:
    def test_league_evolution_empty_returns_empty_lists(self, client: TestClient):
        resp = client.get("/api/league/evolution")
        assert resp.status_code == 200
        assert resp.json() == {"members": [], "champion_history": []}

    def test_league_evolution_with_members_returns_enriched_nodes(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(
            league_dir, "league_000001", created_at="2025-01-01T00:00:00+00:00"
        )
        _make_fake_member(
            league_dir,
            "league_000002",
            parent_id="league_000001",
            created_at="2025-06-01T00:00:00+00:00",
        )
        _write_ratings(
            league_dir,
            [
                {"member_id": "league_000001", "rating": 950.0},
                {"member_id": "league_000002", "rating": 1050.0},
            ],
        )

        resp = client.get("/api/league/evolution")
        assert resp.status_code == 200
        data = resp.json()

        assert "members" in data
        assert "champion_history" in data
        assert len(data["members"]) == 2
        assert len(data["champion_history"]) == 2

        # Verify required keys on each member node.
        for m in data["members"]:
            assert "member_id" in m
            assert "parent_id" in m
            assert "created_at" in m
            assert "notes" in m
            assert "rating" in m
            assert "robustness_score" in m
            strat = m["strategy"]
            assert "cluster_id" in strat
            assert "label" in strat
            for key in (
                "mean_return",
                "worst_case_return",
                "collapse_rate",
                "mean_final_pool",
                "robustness_score",
            ):
                assert key in strat["features"]

        # Verify required keys on each champion_history node.
        for h in data["champion_history"]:
            assert "member_id" in h
            assert "created_at" in h
            assert "rating" in h
            assert "label" in h
            assert "cluster_id" in h
            assert "robustness_score" in h

        # Ratings propagated correctly.
        by_id = {m["member_id"]: m for m in data["members"]}
        assert by_id["league_000001"]["rating"] == 950.0
        assert by_id["league_000002"]["rating"] == 1050.0

        # Without a robustness report, robustness_score is null everywhere.
        for m in data["members"]:
            assert m["robustness_score"] is None

        # Parent info preserved.
        assert by_id["league_000002"]["parent_id"] == "league_000001"

        # Labels are strings.
        for m in data["members"]:
            assert isinstance(m["strategy"]["label"], str)
            assert len(m["strategy"]["label"]) > 0

        # Features are all null (no eval reports available).
        for m in data["members"]:
            for val in m["strategy"]["features"].values():
                assert val is None

    def test_league_evolution_is_sorted_and_deterministic(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(
            league_dir, "league_000003", created_at="2025-03-01T00:00:00+00:00"
        )
        _make_fake_member(
            league_dir, "league_000001", created_at="2025-01-01T00:00:00+00:00"
        )
        _make_fake_member(
            league_dir, "league_000002", created_at="2025-02-01T00:00:00+00:00"
        )

        resp1 = client.get("/api/league/evolution")
        resp2 = client.get("/api/league/evolution")
        assert resp1.status_code == 200
        assert resp1.json() == resp2.json()  # Deterministic across calls.

        data = resp1.json()

        # members sorted by member_id ascending.
        member_ids = [m["member_id"] for m in data["members"]]
        assert member_ids == sorted(member_ids)

        # champion_history sorted by created_at ascending.
        created_ats = [h["created_at"] for h in data["champion_history"]]
        assert created_ats == sorted(created_ats)

    def test_league_evolution_default_rating_when_no_ratings_file(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")

        resp = client.get("/api/league/evolution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["members"][0]["rating"] == 1000.0
        assert data["champion_history"][0]["rating"] == 1000.0

    def test_league_evolution_robustness_score_from_report(
        self, client: TestClient, league_dir: Path, reports_dir: Path
    ):
        _make_fake_member(
            league_dir, "league_000001", created_at="2025-01-01T00:00:00+00:00"
        )
        _make_fake_member(
            league_dir, "league_000002", created_at="2025-06-01T00:00:00+00:00"
        )
        _write_ratings(
            league_dir,
            [
                {"member_id": "league_000001", "rating": 950.0},
                {"member_id": "league_000002", "rating": 1100.0},
            ],
        )
        # league_000002 is the champion (highest rating).
        _write_robustness_report(
            reports_dir, "robust_abc123_20250601T120000Z", rob_score=1.23
        )

        resp = client.get("/api/league/evolution")
        assert resp.status_code == 200
        data = resp.json()

        by_id = {m["member_id"]: m for m in data["members"]}
        # Champion gets the robustness_score.
        assert by_id["league_000002"]["robustness_score"] == pytest.approx(1.23)
        # Non-champion gets null.
        assert by_id["league_000001"]["robustness_score"] is None

        # Same rule in champion_history.
        history_by_id = {h["member_id"]: h for h in data["champion_history"]}
        assert history_by_id["league_000002"]["robustness_score"] == pytest.approx(1.23)
        assert history_by_id["league_000001"]["robustness_score"] is None
