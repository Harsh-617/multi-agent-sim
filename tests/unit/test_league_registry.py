"""Tests for simulation.league.registry.LeagueRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.league.registry import LeagueRegistry


# ── helpers ──────────────────────────────────────────────────────────


def _make_fake_artifacts(path: Path) -> Path:
    """Create a minimal fake agent artifact directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy.pt").write_bytes(b"FAKE_WEIGHTS")
    meta = {"algo": "ppo_shared", "obs_dim": 33, "seed": 42}
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


# ── tests ────────────────────────────────────────────────────────────


class TestSaveSnapshot:
    def test_creates_member_dir_with_copied_files(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)

        member_dir = reg.root / member_id
        assert member_dir.is_dir()
        assert (member_dir / "policy.pt").read_bytes() == b"FAKE_WEIGHTS"
        assert (member_dir / "metadata.json").exists()

    def test_assigns_monotonic_ids(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        id1 = reg.save_snapshot(source)
        id2 = reg.save_snapshot(source)
        id3 = reg.save_snapshot(source)

        assert id1 == "league_000001"
        assert id2 == "league_000002"
        assert id3 == "league_000003"

    def test_injects_metadata_fields(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(
            source, parent_id="league_000099", notes="test snapshot"
        )

        meta = reg.get_member_metadata(member_id)
        assert meta["member_id"] == member_id
        assert meta["parent_id"] == "league_000099"
        assert meta["notes"] == "test snapshot"
        assert "created_at" in meta
        # Original keys preserved
        assert meta["algo"] == "ppo_shared"
        assert meta["obs_dim"] == 33

    def test_rejects_missing_policy(self, tmp_path: Path):
        source = tmp_path / "bad_source"
        source.mkdir()
        (source / "metadata.json").write_text("{}")

        reg = LeagueRegistry(tmp_path / "league")
        with pytest.raises(FileNotFoundError, match="policy.pt"):
            reg.save_snapshot(source)

    def test_rejects_missing_metadata(self, tmp_path: Path):
        source = tmp_path / "bad_source"
        source.mkdir()
        (source / "policy.pt").write_bytes(b"x")

        reg = LeagueRegistry(tmp_path / "league")
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            reg.save_snapshot(source)


class TestListMembers:
    def test_empty_league(self, tmp_path: Path):
        reg = LeagueRegistry(tmp_path / "league")
        assert reg.list_members() == []

    def test_returns_saved_members(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        reg.save_snapshot(source, notes="first")
        reg.save_snapshot(source, notes="second")

        members = reg.list_members()
        assert len(members) == 2
        assert members[0]["member_id"] == "league_000001"
        assert members[1]["member_id"] == "league_000002"
        assert members[0]["notes"] == "first"

    def test_ignores_non_league_dirs(self, tmp_path: Path):
        reg = LeagueRegistry(tmp_path / "league")
        (reg.root / "random_dir").mkdir()
        (reg.root / ".gitkeep").touch()
        assert reg.list_members() == []


class TestLoadMember:
    def test_returns_path_for_existing_member(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)
        path = reg.load_member(member_id)

        assert path.is_dir()
        assert path.name == member_id

    def test_raises_for_unknown_member(self, tmp_path: Path):
        reg = LeagueRegistry(tmp_path / "league")
        with pytest.raises(KeyError, match="league_999999"):
            reg.load_member("league_999999")


class TestGetMemberMetadata:
    def test_returns_metadata_dict(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = LeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)
        meta = reg.get_member_metadata(member_id)

        assert isinstance(meta, dict)
        assert meta["member_id"] == member_id
        assert meta["seed"] == 42
