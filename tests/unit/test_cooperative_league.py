"""Unit tests for CooperativeLeagueRegistry and cooperative Elo ratings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.league.cooperative_registry import CooperativeLeagueRegistry
from simulation.league.cooperative_ratings import (
    K_FACTOR,
    START_RATING,
    elo_expected,
    elo_update,
    load_cooperative_ratings,
    save_cooperative_ratings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_artifacts(path: Path) -> Path:
    """Create a minimal fake cooperative agent artifact directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy.pt").write_bytes(b"FAKE_COOP_WEIGHTS")
    meta = {
        "algo": "ppo_shared",
        "obs_dim": 20,
        "num_action_types": 4,
        "seed": 7,
    }
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _make_registry_with_members(
    tmp_path: Path, count: int
) -> tuple[CooperativeLeagueRegistry, list[str]]:
    """Create a registry with *count* pre-seeded fake members."""
    source = _make_fake_artifacts(tmp_path / "source")
    reg = CooperativeLeagueRegistry(tmp_path / "league")
    ids = [reg.save_snapshot(source, notes=f"member_{i+1}") for i in range(count)]
    return reg, ids


# ---------------------------------------------------------------------------
# CooperativeLeagueRegistry – save_snapshot
# ---------------------------------------------------------------------------


class TestCoopSaveSnapshot:
    def test_creates_member_dir_with_copied_files(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)

        member_dir = reg.root / member_id
        assert member_dir.is_dir()
        assert (member_dir / "policy.pt").read_bytes() == b"FAKE_COOP_WEIGHTS"
        assert (member_dir / "metadata.json").exists()

    def test_assigns_monotonic_ids(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        id1 = reg.save_snapshot(source)
        id2 = reg.save_snapshot(source)
        id3 = reg.save_snapshot(source)

        assert id1 == "league_000001"
        assert id2 == "league_000002"
        assert id3 == "league_000003"

    def test_injects_metadata_fields(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(
            source, parent_id="league_000099", notes="coop_snap"
        )

        meta = reg.get_member_metadata(member_id)
        assert meta["member_id"] == member_id
        assert meta["parent_id"] == "league_000099"
        assert meta["notes"] == "coop_snap"
        assert "created_at" in meta
        # Original keys preserved
        assert meta["obs_dim"] == 20
        assert meta["seed"] == 7

    def test_rejects_missing_policy(self, tmp_path: Path):
        source = tmp_path / "bad_source"
        source.mkdir()
        (source / "metadata.json").write_text("{}")

        reg = CooperativeLeagueRegistry(tmp_path / "league")
        with pytest.raises(FileNotFoundError, match="policy.pt"):
            reg.save_snapshot(source)

    def test_rejects_missing_metadata(self, tmp_path: Path):
        source = tmp_path / "bad_source"
        source.mkdir()
        (source / "policy.pt").write_bytes(b"x")

        reg = CooperativeLeagueRegistry(tmp_path / "league")
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            reg.save_snapshot(source)


# ---------------------------------------------------------------------------
# CooperativeLeagueRegistry – list_members
# ---------------------------------------------------------------------------


class TestCoopListMembers:
    def test_empty_league(self, tmp_path: Path):
        reg = CooperativeLeagueRegistry(tmp_path / "league")
        assert reg.list_members() == []

    def test_returns_saved_members(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        reg.save_snapshot(source, notes="first")
        reg.save_snapshot(source, notes="second")

        members = reg.list_members()
        assert len(members) == 2
        assert members[0]["member_id"] == "league_000001"
        assert members[1]["member_id"] == "league_000002"
        assert members[0]["notes"] == "first"
        assert members[1]["notes"] == "second"

    def test_ignores_non_league_dirs(self, tmp_path: Path):
        reg = CooperativeLeagueRegistry(tmp_path / "league")
        (reg.root / "random_dir").mkdir()
        (reg.root / ".gitkeep").touch()
        assert reg.list_members() == []

    def test_returns_members_sorted_by_id(self, tmp_path: Path):
        _, ids = _make_registry_with_members(tmp_path, 5)
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        members = reg.list_members()
        returned_ids = [m["member_id"] for m in members]
        assert returned_ids == sorted(returned_ids)


# ---------------------------------------------------------------------------
# CooperativeLeagueRegistry – load_member
# ---------------------------------------------------------------------------


class TestCoopLoadMember:
    def test_returns_path_for_existing_member(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)
        path = reg.load_member(member_id)

        assert path.is_dir()
        assert path.name == member_id

    def test_raises_key_error_for_unknown_member(self, tmp_path: Path):
        reg = CooperativeLeagueRegistry(tmp_path / "league")
        with pytest.raises(KeyError, match="league_999999"):
            reg.load_member("league_999999")

    def test_returned_path_contains_policy(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)
        path = reg.load_member(member_id)

        assert (path / "policy.pt").exists()
        assert (path / "metadata.json").exists()


# ---------------------------------------------------------------------------
# CooperativeLeagueRegistry – get_member_metadata
# ---------------------------------------------------------------------------


class TestCoopGetMemberMetadata:
    def test_returns_metadata_dict(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        member_id = reg.save_snapshot(source)
        meta = reg.get_member_metadata(member_id)

        assert isinstance(meta, dict)
        assert meta["member_id"] == member_id
        assert meta["obs_dim"] == 20

    def test_raises_for_unknown_member(self, tmp_path: Path):
        reg = CooperativeLeagueRegistry(tmp_path / "league")
        with pytest.raises(KeyError):
            reg.get_member_metadata("league_000099")


# ---------------------------------------------------------------------------
# Champion selection (highest-rated member)
# ---------------------------------------------------------------------------


class TestCoopChampionSelection:
    def test_champion_is_highest_rated_member(self, tmp_path: Path):
        ratings = {
            "league_000001": 980.0,
            "league_000002": 1100.0,
            "league_000003": 1020.0,
        }
        # Champion is the member with maximum rating
        champion = max(ratings, key=lambda k: ratings[k])
        assert champion == "league_000002"

    def test_single_member_is_champion(self, tmp_path: Path):
        ratings = {"league_000001": 1000.0}
        champion = max(ratings, key=lambda k: ratings[k])
        assert champion == "league_000001"


# ---------------------------------------------------------------------------
# Lineage acyclicity
# ---------------------------------------------------------------------------


class TestCoopLineageAcyclicity:
    def test_lineage_is_acyclic(self, tmp_path: Path):
        """Members with parent_id pointers must form an acyclic DAG."""
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        id1 = reg.save_snapshot(source)
        id2 = reg.save_snapshot(source, parent_id=id1)
        id3 = reg.save_snapshot(source, parent_id=id2)

        members = reg.list_members()
        parent_map = {m["member_id"]: m.get("parent_id") for m in members}

        # Check acyclicity by following parent chain
        def has_cycle(start: str) -> bool:
            visited = set()
            current = start
            while current is not None:
                if current in visited:
                    return True
                visited.add(current)
                current = parent_map.get(current)
            return False

        for mid in [id1, id2, id3]:
            assert not has_cycle(mid), f"Cycle detected from {mid}"

    def test_root_member_has_no_parent(self, tmp_path: Path):
        source = _make_fake_artifacts(tmp_path / "source")
        reg = CooperativeLeagueRegistry(tmp_path / "league")

        id1 = reg.save_snapshot(source)
        meta = reg.get_member_metadata(id1)
        assert meta.get("parent_id") is None


# ---------------------------------------------------------------------------
# Cooperative Elo math
# ---------------------------------------------------------------------------


class TestCoopEloExpected:
    def test_equal_ratings_give_half(self):
        assert elo_expected(1000, 1000) == pytest.approx(0.5)

    def test_higher_rating_expects_more(self):
        assert elo_expected(1200, 1000) > 0.5

    def test_lower_rating_expects_less(self):
        assert elo_expected(1000, 1200) < 0.5

    def test_symmetric(self):
        ea = elo_expected(1200, 1000)
        eb = elo_expected(1000, 1200)
        assert ea + eb == pytest.approx(1.0)


class TestCoopEloUpdate:
    def test_equal_ratings_win(self):
        a, b = elo_update(1000, 1000, 1.0)
        assert a > 1000
        assert b < 1000
        assert a - 1000 == pytest.approx(1000 - b)

    def test_draw_between_equals_no_change(self):
        a, b = elo_update(1000, 1000, 0.5)
        assert a == pytest.approx(1000)
        assert b == pytest.approx(1000)

    def test_total_rating_preserved(self):
        a, b = elo_update(1100, 900, 1.0)
        assert a + b == pytest.approx(1100 + 900)

    def test_k_factor_scales_change(self):
        a1, _ = elo_update(1000, 1000, 1.0, k=16)
        a2, _ = elo_update(1000, 1000, 1.0, k=32)
        assert (a2 - 1000) == pytest.approx(2 * (a1 - 1000))

    def test_upset_gives_large_gain(self):
        # Weaker beats stronger → larger gain than equal-rating win
        a_upset, _ = elo_update(800, 1200, 1.0)
        a_equal, _ = elo_update(1000, 1000, 1.0)
        assert (a_upset - 800) > (a_equal - 1000)


# ---------------------------------------------------------------------------
# Cooperative ratings persistence
# ---------------------------------------------------------------------------


class TestCoopRatingsPersistence:
    def test_round_trip(self, tmp_path: Path):
        ratings = {"league_000001": 1050.0, "league_000002": 950.0}
        path = tmp_path / "coop_ratings.json"
        save_cooperative_ratings(path, ratings)
        loaded = load_cooperative_ratings(path)

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        # Sorted descending by rating
        assert loaded[0]["member_id"] == "league_000001"
        assert loaded[0]["rating"] == 1050.0
        assert loaded[1]["member_id"] == "league_000002"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        assert load_cooperative_ratings(tmp_path / "nope.json") == []

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "coop_ratings.json"
        save_cooperative_ratings(path, {"league_000001": 1000.0})
        assert path.exists()

    def test_ratings_are_rounded(self, tmp_path: Path):
        path = tmp_path / "ratings.json"
        save_cooperative_ratings(path, {"x": 1000.12345})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["rating"] == 1000.12


class TestCoopConstants:
    def test_start_rating(self):
        assert START_RATING == 1000.0

    def test_k_factor(self):
        assert K_FACTOR == 32.0
