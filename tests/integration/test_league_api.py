"""Integration tests for league rating endpoints and league_snapshot runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation.config.defaults import default_config


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect storage directories to a temp folder for test isolation."""
    configs_dir = tmp_path / "configs"
    runs_dir = tmp_path / "runs"
    league_dir = tmp_path / "league"
    configs_dir.mkdir()
    runs_dir.mkdir()
    league_dir.mkdir()

    import backend.api.routes_config as rc
    import backend.api.routes_experiment as re_mod
    import backend.api.routes_league as rl

    monkeypatch.setattr(rc, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(re_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(re_mod, "LEAGUE_ROOT", league_dir)

    # Patch league routes to use isolated paths
    from simulation.league.registry import LeagueRegistry

    isolated_registry = LeagueRegistry(league_dir)
    monkeypatch.setattr(rl, "_registry", isolated_registry)
    monkeypatch.setattr(rl, "LEAGUE_ROOT", league_dir)
    monkeypatch.setattr(rl, "RATINGS_PATH", league_dir / "ratings.json")
    monkeypatch.setattr(rl, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(rl, "PPO_AGENT_DIR", tmp_path / "ppo_shared")

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


def _small_config_json() -> dict:
    cfg = default_config(seed=7)
    data = json.loads(cfg.model_dump_json())
    data["population"]["num_agents"] = 3
    data["population"]["max_steps"] = 15
    return data


def _create_config(client: TestClient) -> str:
    resp = client.post("/api/configs", json=_small_config_json())
    return resp.json()["config_id"]


def _make_fake_member(
    league_dir: Path,
    member_id: str,
    *,
    parent_id: str | None = None,
    created_at: str = "2025-01-01T00:00:00+00:00",
    notes: str | None = None,
) -> None:
    """Create a fake league member directory with minimal artifacts."""
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


# ── Rating endpoint tests ───────────────────────────────────────────


class TestLeagueRatings:
    def test_get_ratings_empty(self, client: TestClient):
        resp = client.get("/api/league/ratings")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_ratings_after_save(self, client: TestClient, league_dir: Path):
        """If ratings.json exists, GET returns its contents."""
        ratings_data = [
            {"member_id": "league_000001", "rating": 1050.0},
            {"member_id": "league_000002", "rating": 950.0},
        ]
        (league_dir / "ratings.json").write_text(
            json.dumps(ratings_data), encoding="utf-8"
        )

        resp = client.get("/api/league/ratings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["member_id"] == "league_000001"

    def test_recompute_no_members(self, client: TestClient):
        """Recompute with no members returns empty list."""
        resp = client.post(
            "/api/league/ratings/recompute",
            json={"num_matches": 1, "seed": 42},
        )
        assert resp.status_code == 200
        assert resp.json() == []


# ── League snapshot run tests ────────────────────────────────────────


class TestLeagueSnapshotRun:
    def test_league_snapshot_missing_member_id_returns_422(self, client: TestClient):
        """Starting a run with league_snapshot but no member_id should fail."""
        config_id = _create_config(client)
        resp = client.post(
            "/api/runs/start",
            json={
                "config_id": config_id,
                "agent_policy": "league_snapshot",
            },
        )
        assert resp.status_code == 422
        assert "league_member_id" in resp.json()["detail"]

    def test_league_snapshot_unknown_member_returns_404(
        self, client: TestClient
    ):
        """Starting a run with a nonexistent league member should fail."""
        config_id = _create_config(client)
        resp = client.post(
            "/api/runs/start",
            json={
                "config_id": config_id,
                "agent_policy": "league_snapshot",
                "league_member_id": "league_999999",
            },
        )
        assert resp.status_code == 404
        assert "league_999999" in resp.json()["detail"]

    def test_league_snapshot_valid_member_starts_run(
        self,
        client: TestClient,
        league_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Starting a run with a valid league member should return 200.

        We monkeypatch create_agent to avoid torch dependency.
        """
        _make_fake_member(league_dir, "league_000001")
        config_id = _create_config(client)

        # Monkeypatch create_agent so we don't need torch
        from simulation.agents.random_agent import RandomAgent

        def _fake_create_agent(policy: str, **kwargs):
            return RandomAgent()

        import backend.runner.experiment_runner as runner_mod

        monkeypatch.setattr(runner_mod, "create_agent", _fake_create_agent)

        resp = client.post(
            "/api/runs/start",
            json={
                "config_id": config_id,
                "agent_policy": "league_snapshot",
                "league_member_id": "league_000001",
            },
        )
        assert resp.status_code == 200
        assert "run_id" in resp.json()


# ── Member endpoint tests ───────────────────────────────────────────


class TestLeagueMembers:
    def test_list_members_empty(self, client: TestClient):
        resp = client.get("/api/league/members")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_members_with_data(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")
        _make_fake_member(league_dir, "league_000002")

        resp = client.get("/api/league/members")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_member_not_found(self, client: TestClient):
        resp = client.get("/api/league/members/league_999999")
        assert resp.status_code == 404


# ── Lineage endpoint tests ────────────────────────────────────────


class TestLeagueLineage:
    def test_lineage_empty(self, client: TestClient):
        resp = client.get("/api/league/lineage")
        assert resp.status_code == 200
        assert resp.json() == {"members": []}

    def test_lineage_returns_members_with_rating(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")
        _make_fake_member(league_dir, "league_000002", parent_id="league_000001")

        # Write ratings
        ratings_data = [
            {"member_id": "league_000001", "rating": 1050.0},
            {"member_id": "league_000002", "rating": 980.0},
        ]
        (league_dir / "ratings.json").write_text(
            json.dumps(ratings_data), encoding="utf-8"
        )

        resp = client.get("/api/league/lineage")
        assert resp.status_code == 200
        data = resp.json()
        members = data["members"]
        assert len(members) == 2
        # Sorted by member_id
        assert members[0]["member_id"] == "league_000001"
        assert members[1]["member_id"] == "league_000002"
        # Each has rating field
        assert members[0]["rating"] == 1050.0
        assert members[1]["rating"] == 980.0
        # Parent info preserved
        assert members[1]["parent_id"] == "league_000001"

    def test_lineage_default_rating_when_no_ratings_file(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")

        resp = client.get("/api/league/lineage")
        assert resp.status_code == 200
        members = resp.json()["members"]
        assert len(members) == 1
        assert members[0]["rating"] == 1000.0


# ── Champion endpoint tests ───────────────────────────────────────


class TestLeagueChampion:
    def test_champion_no_members_returns_404(self, client: TestClient):
        resp = client.get("/api/league/champion")
        assert resp.status_code == 404

    def test_champion_returns_highest_rated(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")
        _make_fake_member(league_dir, "league_000002")

        ratings_data = [
            {"member_id": "league_000001", "rating": 950.0},
            {"member_id": "league_000002", "rating": 1100.0},
        ]
        (league_dir / "ratings.json").write_text(
            json.dumps(ratings_data), encoding="utf-8"
        )

        resp = client.get("/api/league/champion")
        assert resp.status_code == 200
        data = resp.json()
        assert data["member_id"] == "league_000002"
        assert data["rating"] == 1100.0

    def test_champion_tiebreak_newest(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(
            league_dir, "league_000001",
            created_at="2025-01-01T00:00:00+00:00",
        )
        _make_fake_member(
            league_dir, "league_000002",
            created_at="2025-06-01T00:00:00+00:00",
        )
        # No ratings file -> both default to 1000
        resp = client.get("/api/league/champion")
        assert resp.status_code == 200
        # tie-break: newest created_at wins
        assert resp.json()["member_id"] == "league_000002"


# ── Champion benchmark endpoint tests ─────────────────────────────


class TestChampionBenchmark:
    def test_benchmark_no_members_returns_404(self, client: TestClient):
        config_id = _create_config(client)
        resp = client.post(
            "/api/league/champion/benchmark",
            json={"config_id": config_id, "episodes": 1, "seed": 42},
        )
        assert resp.status_code == 404

    def test_benchmark_bad_config_returns_404(
        self, client: TestClient, league_dir: Path
    ):
        _make_fake_member(league_dir, "league_000001")
        resp = client.post(
            "/api/league/champion/benchmark",
            json={"config_id": "nonexistent", "episodes": 1, "seed": 42},
        )
        assert resp.status_code == 404

    def test_benchmark_runs_and_returns_results(
        self,
        client: TestClient,
        league_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _make_fake_member(league_dir, "league_000001")
        config_id = _create_config(client)

        # Monkeypatch create_agent so we don't need torch
        from simulation.agents.random_agent import RandomAgent

        def _fake_create_agent(policy: str, **kwargs):
            return RandomAgent()

        import backend.api.routes_league as rl

        monkeypatch.setattr(rl, "create_agent", _fake_create_agent)

        resp = client.post(
            "/api/league/champion/benchmark",
            json={"config_id": config_id, "episodes": 2, "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Champion info present
        assert data["champion"]["member_id"] == "league_000001"

        # Results for champion + 4 baselines (ppo_shared excluded since no artifacts)
        policies = [r["policy"] for r in data["results"]]
        assert "league_champion" in policies
        assert "random" in policies
        assert "always_cooperate" in policies
        assert "always_extract" in policies
        assert "tit_for_tat" in policies
        assert "ppo_shared" not in policies

        # Each result has required keys
        for r in data["results"]:
            assert "mean_total_reward" in r
            assert "mean_final_shared_pool" in r
            assert "collapse_rate" in r
            assert "mean_episode_length" in r

    def test_benchmark_includes_ppo_when_artifacts_exist(
        self,
        client: TestClient,
        league_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _make_fake_member(league_dir, "league_000001")
        config_id = _create_config(client)

        # Create fake PPO artifacts
        ppo_dir = tmp_path / "ppo_shared"
        ppo_dir.mkdir(parents=True, exist_ok=True)
        (ppo_dir / "policy.pt").write_bytes(b"FAKE")
        (ppo_dir / "metadata.json").write_text("{}", encoding="utf-8")

        from simulation.agents.random_agent import RandomAgent

        def _fake_create_agent(policy: str, **kwargs):
            return RandomAgent()

        import backend.api.routes_league as rl

        monkeypatch.setattr(rl, "create_agent", _fake_create_agent)

        resp = client.post(
            "/api/league/champion/benchmark",
            json={"config_id": config_id, "episodes": 1, "seed": 42},
        )
        assert resp.status_code == 200
        policies = [r["policy"] for r in resp.json()["results"]]
        assert "ppo_shared" in policies
