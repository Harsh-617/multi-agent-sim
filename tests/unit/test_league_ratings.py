"""Tests for simulation.league.ratings — Elo math and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.league.ratings import (
    K_FACTOR,
    START_RATING,
    elo_expected,
    elo_update,
    load_ratings,
    save_ratings,
)


# ── Elo math ─────────────────────────────────────────────────────────


class TestEloExpected:
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

    def test_400_point_gap(self):
        # 400-point difference → ~0.909 expected score
        e = elo_expected(1400, 1000)
        assert e == pytest.approx(10 / 11, abs=0.001)


class TestEloUpdate:
    def test_equal_ratings_win(self):
        a, b = elo_update(1000, 1000, 1.0)
        # Winner gains, loser loses equally
        assert a > 1000
        assert b < 1000
        assert a - 1000 == pytest.approx(1000 - b)

    def test_equal_ratings_draw(self):
        a, b = elo_update(1000, 1000, 0.5)
        # Draw between equals: no change
        assert a == pytest.approx(1000)
        assert b == pytest.approx(1000)

    def test_upset_gives_large_gain(self):
        # Weaker player beats stronger player → big gain
        a, b = elo_update(800, 1200, 1.0)
        gain = a - 800
        # Compare to expected gain when equal
        a2, _ = elo_update(1000, 1000, 1.0)
        gain_equal = a2 - 1000
        assert gain > gain_equal

    def test_k_factor_scales_change(self):
        a1, b1 = elo_update(1000, 1000, 1.0, k=16)
        a2, b2 = elo_update(1000, 1000, 1.0, k=32)
        assert (a2 - 1000) == pytest.approx(2 * (a1 - 1000))

    def test_total_rating_preserved(self):
        a, b = elo_update(1100, 900, 1.0)
        assert a + b == pytest.approx(1100 + 900)


# ── Persistence ──────────────────────────────────────────────────────


class TestSaveLoadRatings:
    def test_round_trip(self, tmp_path: Path):
        ratings = {"league_000001": 1050.0, "league_000002": 950.0}
        path = tmp_path / "ratings.json"
        save_ratings(path, ratings)
        loaded = load_ratings(path)

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        # Sorted descending by rating
        assert loaded[0]["member_id"] == "league_000001"
        assert loaded[0]["rating"] == 1050.0
        assert loaded[1]["member_id"] == "league_000002"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        assert load_ratings(tmp_path / "nope.json") == []

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "ratings.json"
        save_ratings(path, {"a": 1000.0})
        assert path.exists()

    def test_ratings_are_rounded(self, tmp_path: Path):
        path = tmp_path / "ratings.json"
        save_ratings(path, {"x": 1000.12345})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["rating"] == 1000.12


class TestConstants:
    def test_start_rating(self):
        assert START_RATING == 1000.0

    def test_k_factor(self):
        assert K_FACTOR == 32.0
