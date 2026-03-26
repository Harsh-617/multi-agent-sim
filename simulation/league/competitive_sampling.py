"""CompetitiveOpponentSampler — selects opponent policies for the Competitive archetype.

Follows the same pattern as sampling.py (Mixed archetype) but uses competitive
baseline policies and the competitive league registry path.

Deterministic sampling given an RNG.  Supports weighting between recent/old
league members, baseline policies, and optionally fixed trained policies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.league.registry import LeagueRegistry

# Competitive baseline policy names
COMPETITIVE_BASELINE_POLICIES = (
    "random",
    "always_attack",
    "always_build",
    "always_defend",
)


class CompetitiveOpponentSampler:
    """Sample opponent specs from the competitive population.

    Parameters
    ----------
    recent_weight : float
        Weight for sampling from recent league members.
    old_weight : float
        Weight for sampling from older league members.
    baseline_weight : float
        Weight for sampling from baseline policies.
    fixed_weight : float
        Weight for sampling from fixed trained policies.
    """

    def __init__(
        self,
        recent_weight: float = 0.5,
        old_weight: float = 0.1,
        baseline_weight: float = 0.3,
        fixed_weight: float = 0.1,
    ) -> None:
        self._recent_weight = recent_weight
        self._old_weight = old_weight
        self._baseline_weight = baseline_weight
        self._fixed_weight = fixed_weight

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        league_registry: LeagueRegistry,
        baseline_policies: list[str],
        fixed_policies: list[str],
        rng: np.random.Generator,
    ) -> dict[str, str]:
        """Return a policy spec dict chosen according to the weights.

        Parameters
        ----------
        league_registry : LeagueRegistry
            Provides league member listings.
        baseline_policies : list[str]
            Names of baseline policies (e.g. ``["random", "always_attack", ...]``).
        fixed_policies : list[str]
            Extra fixed policy names (e.g. ``["competitive_ppo"]``) to include.
        rng : np.random.Generator
            Random number generator for deterministic sampling.

        Returns
        -------
        dict[str, str]
            A policy spec dict with keys ``"type"`` and ``"policy"``.
        """
        members = league_registry.list_members()
        has_league = len(members) > 0
        has_fixed = len(fixed_policies) > 0

        # Build candidate buckets with associated weights
        buckets: list[tuple[str, float]] = []
        if has_league:
            buckets.append(("league_recent", self._recent_weight))
            buckets.append(("league_old", self._old_weight))
        buckets.append(("baseline", self._baseline_weight))
        if has_fixed:
            buckets.append(("fixed", self._fixed_weight))

        # Normalise
        names = [b[0] for b in buckets]
        raw_weights = np.array([b[1] for b in buckets], dtype=np.float64)
        total = raw_weights.sum()
        if total <= 0:
            # Fallback: uniform over baselines
            return self._sample_baseline(baseline_policies, rng)
        probs = raw_weights / total

        chosen_bucket = str(rng.choice(names, p=probs))

        if chosen_bucket == "league_recent":
            return self._sample_league_member(members, recent=True, rng=rng)
        if chosen_bucket == "league_old":
            return self._sample_league_member(members, recent=False, rng=rng)
        if chosen_bucket == "fixed":
            idx = int(rng.integers(0, len(fixed_policies)))
            return {"type": "fixed", "policy": fixed_policies[idx]}
        return self._sample_baseline(baseline_policies, rng)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_baseline(
        self, baseline_policies: list[str], rng: np.random.Generator
    ) -> dict[str, str]:
        idx = int(rng.integers(0, len(baseline_policies)))
        return {"type": "baseline", "policy": baseline_policies[idx]}

    def _sample_league_member(
        self,
        members: list[dict[str, Any]],
        *,
        recent: bool,
        rng: np.random.Generator,
    ) -> dict[str, str]:
        """Pick a league member, biased toward recent or old ones."""
        if len(members) == 1:
            return {"type": "league", "policy": members[0]["member_id"]}

        half = max(1, len(members) // 2)
        if recent:
            # Pick from the most recent half
            idx = int(rng.integers(len(members) - half, len(members)))
        else:
            # Pick from the oldest half
            idx = int(rng.integers(0, half))

        return {"type": "league", "policy": members[idx]["member_id"]}
