"""Tests for simulation.league.sampling — OpponentSampler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.league.registry import LeagueRegistry
from simulation.league.sampling import (
    BASELINE_POLICIES,
    OpponentSampler,
    OpponentSpec,
    SamplingWeights,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_fake_artifacts(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy.pt").write_bytes(b"FAKE_WEIGHTS")
    meta = {"algo": "ppo_shared", "obs_dim": 33, "seed": 42}
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _populate_league(tmp_path: Path, n: int = 3) -> LeagueRegistry:
    source = _make_fake_artifacts(tmp_path / "source")
    reg = LeagueRegistry(tmp_path / "league")
    for i in range(n):
        reg.save_snapshot(source, notes=f"member_{i}")
    return reg


# ── determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_same_sequence(self, tmp_path: Path):
        reg = _populate_league(tmp_path, n=5)
        s1 = OpponentSampler(reg, seed=99)
        s2 = OpponentSampler(reg, seed=99)

        seq1 = [s1.sample_opponent_policy() for _ in range(20)]
        seq2 = [s2.sample_opponent_policy() for _ in range(20)]

        for a, b in zip(seq1, seq2):
            assert a.source == b.source
            assert a.policy == b.policy

    def test_different_seed_different_sequence(self, tmp_path: Path):
        reg = _populate_league(tmp_path, n=5)
        s1 = OpponentSampler(reg, seed=1)
        s2 = OpponentSampler(reg, seed=9999)

        seq1 = [s1.sample_opponent_policy().policy for _ in range(20)]
        seq2 = [s2.sample_opponent_policy().policy for _ in range(20)]

        # Extremely unlikely to be identical
        assert seq1 != seq2


# ── sampling behaviour ───────────────────────────────────────────────


class TestSamplingBehaviour:
    def test_baseline_only_when_no_league(self, tmp_path: Path):
        reg = LeagueRegistry(tmp_path / "empty_league")
        sampler = OpponentSampler(reg, seed=42)

        specs = [sampler.sample_opponent_policy() for _ in range(30)]
        for s in specs:
            assert s.source in ("baseline", "fixed")
            if s.source == "baseline":
                assert s.policy in BASELINE_POLICIES

    def test_league_members_appear_when_present(self, tmp_path: Path):
        reg = _populate_league(tmp_path, n=5)
        sampler = OpponentSampler(
            reg, seed=42, weights=SamplingWeights(league_weight=10.0, baseline_weight=0.0)
        )

        specs = [sampler.sample_opponent_policy() for _ in range(30)]
        league_specs = [s for s in specs if s.source == "league"]
        assert len(league_specs) > 0
        for s in league_specs:
            assert s.policy.startswith("league_")

    def test_include_fixed_adds_ppo_shared(self, tmp_path: Path):
        reg = LeagueRegistry(tmp_path / "empty_league")
        sampler = OpponentSampler(
            reg, seed=42, include_fixed=["ppo_shared"]
        )
        specs = [sampler.sample_opponent_policy() for _ in range(50)]
        fixed_specs = [s for s in specs if s.source == "fixed"]
        assert len(fixed_specs) > 0
        for s in fixed_specs:
            assert s.policy == "ppo_shared"


class TestListLeagueMembers:
    def test_lists_members_from_registry(self, tmp_path: Path):
        reg = _populate_league(tmp_path, n=3)
        sampler = OpponentSampler(reg, seed=0)
        members = sampler.list_league_members()
        assert len(members) == 3
        assert members[0]["member_id"] == "league_000001"


class TestOpponentSpec:
    def test_to_dict(self):
        spec = OpponentSpec(source="baseline", policy="random")
        d = spec.to_dict()
        assert d == {"type": "baseline", "policy": "random"}
