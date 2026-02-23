"""OpponentSampler — selects opponent policies from league, baselines, or fixed.

Deterministic sampling given a seed.  Supports weighting between recent/old
league members, baseline policies, and optionally a fixed trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.core.seeding import make_rng
from simulation.league.registry import LeagueRegistry

# Baseline policy names that don't require torch
BASELINE_POLICIES = ("random", "always_cooperate", "always_extract", "tit_for_tat")


@dataclass
class OpponentSpec:
    """Lightweight descriptor for an opponent policy."""

    source: str  # "baseline" | "league" | "fixed"
    policy: str  # e.g. "tit_for_tat", "league_000003", "ppo_shared"

    def to_dict(self) -> dict[str, str]:
        return {"type": self.source, "policy": self.policy}


@dataclass
class SamplingWeights:
    """Relative weights controlling the opponent mix.

    All weights are non-negative.  They are normalised internally so they
    don't need to sum to 1.
    """

    league_weight: float = 1.0
    baseline_weight: float = 1.0
    recent_vs_old: float = 0.7  # within league: P(recent) vs P(old)


class OpponentSampler:
    """Sample opponent specs from the available population.

    Parameters
    ----------
    registry : LeagueRegistry
        Provides league member listings.
    seed : int
        Root seed for deterministic sampling.
    weights : SamplingWeights | None
        Controls the mix of league vs baseline opponents.
    include_fixed : list[str] | None
        Extra fixed policy names (e.g. ``["ppo_shared"]``) to include
        in the sampling pool alongside baselines.
    """

    def __init__(
        self,
        registry: LeagueRegistry,
        seed: int = 42,
        weights: SamplingWeights | None = None,
        include_fixed: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._weights = weights or SamplingWeights()
        self._rng = make_rng(seed)
        self._include_fixed = include_fixed or []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_league_members(self) -> list[dict[str, Any]]:
        """Return metadata dicts from the registry, sorted by id."""
        return self._registry.list_members()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_opponent_policy(self) -> OpponentSpec:
        """Return an :class:`OpponentSpec` chosen according to the weights.

        Deterministic given the internal RNG state (which advances with each
        call).
        """
        w = self._weights
        members = self.list_league_members()
        has_league = len(members) > 0

        # Build candidate buckets with associated weights
        buckets: list[tuple[str, float]] = []  # (bucket_name, weight)
        if has_league:
            buckets.append(("league", w.league_weight))
        buckets.append(("baseline", w.baseline_weight))
        for fp in self._include_fixed:
            buckets.append(("fixed:" + fp, w.baseline_weight * 0.5))

        # Normalise
        names = [b[0] for b in buckets]
        raw_weights = np.array([b[1] for b in buckets], dtype=np.float64)
        total = raw_weights.sum()
        if total <= 0:
            # Fallback: uniform over baselines
            return self._sample_baseline()
        probs = raw_weights / total

        chosen_bucket = str(self._rng.choice(names, p=probs))

        if chosen_bucket == "league":
            return self._sample_league_member(members)
        if chosen_bucket.startswith("fixed:"):
            policy_name = chosen_bucket.split(":", 1)[1]
            return OpponentSpec(source="fixed", policy=policy_name)
        return self._sample_baseline()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_baseline(self) -> OpponentSpec:
        idx = int(self._rng.integers(0, len(BASELINE_POLICIES)))
        return OpponentSpec(source="baseline", policy=BASELINE_POLICIES[idx])

    def _sample_league_member(self, members: list[dict[str, Any]]) -> OpponentSpec:
        """Pick a league member, biased toward recent (high id) ones."""
        if len(members) == 1:
            return OpponentSpec(source="league", policy=members[0]["member_id"])

        r = self._weights.recent_vs_old
        # "recent" = last member; "old" = first member; middle = uniform
        roll = float(self._rng.random())
        if roll < r:
            # Pick from the most recent half
            half = max(1, len(members) // 2)
            idx = int(self._rng.integers(len(members) - half, len(members)))
        else:
            # Pick from the oldest half
            half = max(1, len(members) // 2)
            idx = int(self._rng.integers(0, half))

        return OpponentSpec(source="league", policy=members[idx]["member_id"])
