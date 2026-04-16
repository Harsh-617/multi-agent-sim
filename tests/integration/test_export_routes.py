"""Integration tests for GET /api/export/{archetype}/champion and /members/{id}.

Uses FastAPI TestClient with isolated tmp_path storage; fake members are
created with randomly-initialised torch weights so the real zip packing path
is exercised.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_member(
    league_dir: Path,
    member_id: str = "league_000001",
    obs_dim: int = 33,
    num_action_types: int = 4,
) -> Path:
    """Write policy.pt + metadata.json into league_dir/member_id/."""
    from simulation.training.ppo_shared import SharedPolicyNetwork

    member_dir = league_dir / member_id
    member_dir.mkdir(parents=True, exist_ok=True)

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    torch.save(net.state_dict(), member_dir / "policy.pt")

    meta = {
        "obs_dim": obs_dim,
        "num_action_types": num_action_types,
        "member_id": member_id,
        "config_hash": "deadbeef",
        "training_steps": 50000,
        "seed": 42,
    }
    (member_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return member_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect routes_export storage roots to tmp_path."""
    import backend.api.routes_export as re_mod

    mixed_league = tmp_path / "agents/league"
    comp_league = tmp_path / "agents/competitive_league"
    coop_league = tmp_path / "agents/cooperative/league"

    for d in (mixed_league, comp_league, coop_league):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(re_mod, "_MIXED_LEAGUE_ROOT", mixed_league)
    monkeypatch.setattr(re_mod, "_COMPETITIVE_LEAGUE_ROOT", comp_league)
    monkeypatch.setattr(re_mod, "_COOPERATIVE_LEAGUE_ROOT", coop_league)

    yield {
        "mixed_league": mixed_league,
        "comp_league": comp_league,
        "coop_league": coop_league,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Champion endpoints — 200 + application/zip
# ---------------------------------------------------------------------------

class TestChampionExport:
    def test_mixed_champion_returns_200_and_zip(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["mixed_league"], obs_dim=33, num_action_types=4)

        resp = client.get("/api/export/mixed/champion")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_competitive_champion_returns_200_and_zip(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["comp_league"], obs_dim=93, num_action_types=4)

        resp = client.get("/api/export/competitive/champion")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_cooperative_champion_returns_200_and_zip(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["coop_league"], obs_dim=40, num_action_types=4)

        resp = client.get("/api/export/cooperative/champion")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_champion_zip_contains_three_files(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["mixed_league"], obs_dim=33, num_action_types=4)

        resp = client.get("/api/export/mixed/champion")
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
        assert names == {"policy.py", "policy.pt", "README.md"}

    def test_champion_returns_404_when_no_members(self, client, _isolate_export):
        # No members created — league is empty
        resp = client.get("/api/export/mixed/champion")
        assert resp.status_code == 404

    def test_champion_returns_404_when_policy_pt_missing(
        self, client, _isolate_export
    ):
        dirs = _isolate_export
        # Create member dir with only metadata (no policy.pt)
        member_dir = dirs["mixed_league"] / "league_000001"
        member_dir.mkdir(parents=True, exist_ok=True)
        meta = {"obs_dim": 33, "num_action_types": 4, "member_id": "league_000001"}
        (member_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        resp = client.get("/api/export/mixed/champion")
        assert resp.status_code == 404

    def test_champion_filename_header_correct(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(
            dirs["mixed_league"], member_id="league_000001", obs_dim=33, num_action_types=4
        )

        resp = client.get("/api/export/mixed/champion")
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "policy_mixed_league_000001.zip" in cd


# ---------------------------------------------------------------------------
# Member endpoints
# ---------------------------------------------------------------------------

class TestMemberExport:
    def test_mixed_member_returns_200_and_zip(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["mixed_league"], member_id="league_000001")

        resp = client.get("/api/export/mixed/members/league_000001")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_nonexistent_member_returns_404(self, client, _isolate_export):
        resp = client.get("/api/export/mixed/members/league_999999")
        assert resp.status_code == 404

    def test_member_zip_contains_three_files(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["mixed_league"], member_id="league_000001")

        resp = client.get("/api/export/mixed/members/league_000001")
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
        assert names == {"policy.py", "policy.pt", "README.md"}

    def test_member_filename_header_correct(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["mixed_league"], member_id="league_000001")

        resp = client.get("/api/export/mixed/members/league_000001")
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "policy_mixed_league_000001.zip" in cd

    def test_cooperative_member_returns_200_and_zip(self, client, _isolate_export):
        dirs = _isolate_export
        _make_fake_member(dirs["coop_league"], member_id="league_000001", obs_dim=40, num_action_types=4)

        resp = client.get("/api/export/cooperative/members/league_000001")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"


# ---------------------------------------------------------------------------
# Invalid archetype — 422
# ---------------------------------------------------------------------------

class TestInvalidArchetype:
    def test_invalid_archetype_champion_returns_422(self, client, _isolate_export):
        resp = client.get("/api/export/invalid_archetype/champion")
        assert resp.status_code == 422

    def test_invalid_archetype_member_returns_422(self, client, _isolate_export):
        resp = client.get("/api/export/bad_type/members/league_000001")
        assert resp.status_code == 422
