"""Integration tests for cooperative league endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation.league.cooperative_registry import CooperativeLeagueRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_coop_league(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect cooperative league storage to a temp folder."""
    league_dir = tmp_path / "coop_league"
    league_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    import backend.api.routes_cooperative_league as rcl

    isolated_registry = CooperativeLeagueRegistry(league_dir)
    monkeypatch.setattr(rcl, "_registry", isolated_registry)
    monkeypatch.setattr(rcl, "LEAGUE_ROOT", league_dir)
    monkeypatch.setattr(rcl, "RATINGS_PATH", league_dir / "ratings.json")
    monkeypatch.setattr(rcl, "REPORTS_ROOT", reports_dir)

    # Reset robustness manager state
    rcl.coop_robustness_manager.reset_state()
    yield
    rcl.coop_robustness_manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def league_dir(tmp_path: Path) -> Path:
    return tmp_path / "coop_league"


# ---------------------------------------------------------------------------
# Helper: create fake league member
# ---------------------------------------------------------------------------


def _make_fake_member(
    league_dir: Path,
    parent_id: str | None = None,
    obs_dim: int = 20,
    rating: float | None = None,
    notes: str | None = None,
) -> str:
    """Create a fake cooperative league member and return member_id."""
    import torch
    from simulation.training.ppo_shared import SharedPolicyNetwork

    reg = CooperativeLeagueRegistry(league_dir)
    source = league_dir / f"_src_{len(list(league_dir.iterdir()))}"
    source.mkdir(exist_ok=True)

    net = SharedPolicyNetwork(obs_dim, 4)
    torch.save(net.state_dict(), source / "policy.pt")

    meta = {"obs_dim": obs_dim, "num_action_types": 4, "seed": 1, "algo": "ppo_shared"}
    (source / "metadata.json").write_text(json.dumps(meta))

    member_id = reg.save_snapshot(source, parent_id=parent_id, notes=notes)

    # Optionally save ratings
    if rating is not None:
        ratings_path = league_dir / "ratings.json"
        existing = []
        if ratings_path.exists():
            existing = json.loads(ratings_path.read_text())
        existing.append({"member_id": member_id, "rating": rating})
        ratings_path.write_text(json.dumps(existing))

    return member_id


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/members
# ---------------------------------------------------------------------------


class TestGetCoopMembers:
    def test_empty_league_returns_empty_list(self, client: TestClient):
        resp = client.get("/api/cooperative/league/members")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_saved_members(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir, notes="first")
        _make_fake_member(league_dir, notes="second")

        resp = client.get("/api/cooperative/league/members")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = {m["member_id"] for m in data}
        assert "league_000001" in ids
        assert "league_000002" in ids

    def test_members_have_rating_field(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir, rating=1050.0)

        resp = client.get("/api/cooperative/league/members")
        data = resp.json()
        assert len(data) == 1
        assert "rating" in data[0]
        assert data[0]["rating"] == 1050.0

    def test_default_rating_when_no_ratings_file(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir)  # no rating kwarg = no ratings.json entry

        resp = client.get("/api/cooperative/league/members")
        data = resp.json()
        assert data[0]["rating"] == 1000.0  # _DEFAULT_RATING


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/members/{id}
# ---------------------------------------------------------------------------


class TestGetCoopMember:
    def test_returns_member_metadata(self, client: TestClient, league_dir: Path):
        mid = _make_fake_member(league_dir, notes="detail_test")

        resp = client.get(f"/api/cooperative/league/members/{mid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["member_id"] == mid
        assert data["notes"] == "detail_test"

    def test_unknown_member_returns_404(self, client: TestClient):
        resp = client.get("/api/cooperative/league/members/league_999999")
        assert resp.status_code == 404

    def test_member_includes_rating(self, client: TestClient, league_dir: Path):
        mid = _make_fake_member(league_dir, rating=1080.0)

        resp = client.get(f"/api/cooperative/league/members/{mid}")
        data = resp.json()
        assert data["rating"] == 1080.0


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/champion
# ---------------------------------------------------------------------------


class TestGetCoopChampion:
    def test_empty_league_returns_null_champion(self, client: TestClient):
        resp = client.get("/api/cooperative/league/champion")
        assert resp.status_code == 200
        assert resp.json()["member_id"] is None

    def test_champion_is_highest_rated(self, client: TestClient, league_dir: Path):
        mid1 = _make_fake_member(league_dir, rating=900.0)
        mid2 = _make_fake_member(league_dir, rating=1100.0)
        mid3 = _make_fake_member(league_dir, rating=950.0)

        resp = client.get("/api/cooperative/league/champion")
        assert resp.status_code == 200
        data = resp.json()
        assert data["member_id"] == mid2
        assert data["rating"] == 1100.0

    def test_single_member_is_champion(self, client: TestClient, league_dir: Path):
        mid = _make_fake_member(league_dir, rating=1000.0)

        resp = client.get("/api/cooperative/league/champion")
        data = resp.json()
        assert data["member_id"] == mid


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/lineage
# ---------------------------------------------------------------------------


class TestGetCoopLineage:
    def test_returns_members_key(self, client: TestClient):
        resp = client.get("/api/cooperative/league/lineage")
        assert resp.status_code == 200
        assert "members" in resp.json()

    def test_lineage_includes_parent_id(self, client: TestClient, league_dir: Path):
        mid1 = _make_fake_member(league_dir)
        mid2 = _make_fake_member(league_dir, parent_id=mid1)

        resp = client.get("/api/cooperative/league/lineage")
        data = resp.json()
        members = {m["member_id"]: m for m in data["members"]}

        assert members[mid1]["parent_id"] is None
        assert members[mid2]["parent_id"] == mid1

    def test_lineage_members_have_label(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir)

        resp = client.get("/api/cooperative/league/lineage")
        data = resp.json()
        for m in data["members"]:
            assert "label" in m

    def test_lineage_members_have_rating(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir, rating=1050.0)

        resp = client.get("/api/cooperative/league/lineage")
        data = resp.json()
        assert data["members"][0]["rating"] == 1050.0


# ---------------------------------------------------------------------------
# POST /api/cooperative/league/champion/robustness
# ---------------------------------------------------------------------------


class TestCoopChampionRobustness:
    def test_no_members_returns_error_quickly(self, client: TestClient):
        """Without league members, robustness task fails gracefully."""
        resp = client.post(
            "/api/cooperative/league/champion/robustness",
            json={"seeds": 1, "episodes_per_seed": 1, "limit_sweeps": 1},
        )
        # Should return 200 with a robustness_id (task runs in background)
        # OR 409 if already running
        assert resp.status_code in (200, 409)

    def test_returns_robustness_id(self, client: TestClient, league_dir: Path):
        import backend.api.routes_cooperative_league as rcl

        # Ensure manager is idle
        rcl.coop_robustness_manager.reset_state()

        _make_fake_member(league_dir, rating=1000.0)

        resp = client.post(
            "/api/cooperative/league/champion/robustness",
            json={"seeds": 1, "episodes_per_seed": 1, "limit_sweeps": 1, "seed": 0},
        )
        assert resp.status_code == 200
        assert "robustness_id" in resp.json()

    def test_duplicate_trigger_returns_409(self, client: TestClient, league_dir: Path):
        import backend.api.routes_cooperative_league as rcl

        _make_fake_member(league_dir, rating=1000.0)
        rcl.coop_robustness_manager.running = True

        resp = client.post(
            "/api/cooperative/league/champion/robustness",
            json={"seeds": 1, "episodes_per_seed": 1, "limit_sweeps": 1},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/champion/robustness/{id}/status
# ---------------------------------------------------------------------------


class TestCoopRobustnessStatus:
    def test_unknown_robustness_id_returns_404(self, client: TestClient):
        resp = client.get(
            "/api/cooperative/league/champion/robustness/nonexistent123/status"
        )
        assert resp.status_code == 404

    def test_known_id_returns_status(self, client: TestClient):
        import backend.api.routes_cooperative_league as rcl

        rid = "testabc123456"
        rcl.coop_robustness_manager.robustness_id = rid
        rcl.coop_robustness_manager.stage = "evaluating"

        resp = client.get(
            f"/api/cooperative/league/champion/robustness/{rid}/status"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["robustness_id"] == rid
        assert data["stage"] == "evaluating"


# ---------------------------------------------------------------------------
# GET /api/cooperative/league/evolution
# ---------------------------------------------------------------------------


class TestGetCoopEvolution:
    def test_empty_league_returns_empty_members(self, client: TestClient):
        resp = client.get("/api/cooperative/league/evolution")
        assert resp.status_code == 200
        data = resp.json()
        assert "members" in data
        assert "champion_history" in data
        assert data["members"] == []

    def test_evolution_includes_strategy(self, client: TestClient, league_dir: Path):
        _make_fake_member(league_dir, rating=1000.0)

        resp = client.get("/api/cooperative/league/evolution")
        data = resp.json()
        for m in data["members"]:
            assert "strategy" in m
            assert "label" in m["strategy"]
