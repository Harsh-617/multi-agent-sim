"""Resolve a set of PolicySpec objects for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from simulation.league.ratings import load_ratings
from simulation.league.registry import LeagueRegistry

BASELINE_POLICIES = ("random", "always_cooperate", "always_extract", "tit_for_tat")

_DEFAULT_LEAGUE_ROOT = Path("storage/agents/league")
_DEFAULT_RATINGS_PATH = _DEFAULT_LEAGUE_ROOT / "ratings.json"
_DEFAULT_PPO_DIR = Path("storage/agents/ppo_shared")


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Describes one policy to evaluate."""

    name: str
    agent_policy: str
    source: Literal["baseline", "ppo", "champion", "league"]
    league_member_id: str | None = None
    available: bool = True
    skip_reason: str | None = None
    agent_kwargs: dict = field(default_factory=dict)


def resolve_policy_set(
    *,
    league_root: Path = _DEFAULT_LEAGUE_ROOT,
    ratings_path: Path = _DEFAULT_RATINGS_PATH,
    ppo_dir: Path = _DEFAULT_PPO_DIR,
    top_k: int = 3,
) -> list[PolicySpec]:
    """Build the full list of policies to evaluate.

    Returns both available and unavailable specs so callers can
    report which policies were skipped and why.
    """
    specs: list[PolicySpec] = []

    # --- baselines (always available) ---
    for p in BASELINE_POLICIES:
        specs.append(PolicySpec(name=p, agent_policy=p, source="baseline"))

    # --- ppo_shared ---
    ppo_policy = ppo_dir / "policy.pt"
    ppo_meta = ppo_dir / "metadata.json"
    if ppo_policy.exists() and ppo_meta.exists():
        specs.append(
            PolicySpec(
                name="ppo_shared",
                agent_policy="ppo_shared",
                source="ppo",
                agent_kwargs={"agent_dir": str(ppo_dir)},
            )
        )
    else:
        specs.append(
            PolicySpec(
                name="ppo_shared",
                agent_policy="ppo_shared",
                source="ppo",
                available=False,
                skip_reason="Missing artifacts (policy.pt / metadata.json)",
            )
        )

    # --- league members (champion + top-k) ---
    registry = LeagueRegistry(league_root)
    members = registry.list_members()

    if not members:
        specs.append(
            PolicySpec(
                name="league_champion",
                agent_policy="league_snapshot",
                source="champion",
                available=False,
                skip_reason="No league members exist",
            )
        )
    else:
        # Build ratings map
        raw_ratings = load_ratings(ratings_path)
        ratings_map: dict[str, float] = {
            r["member_id"]: r["rating"] for r in raw_ratings
        }

        # Sort members by rating desc, then by created_at desc for tie-break
        def _sort_key(m: dict) -> tuple[float, str]:
            return (
                ratings_map.get(m["member_id"], 1000.0),
                m.get("created_at") or "",
            )

        ranked = sorted(members, key=_sort_key, reverse=True)

        # Champion = highest-rated
        champ = ranked[0]
        champ_id = champ["member_id"]
        champ_dir = registry.load_member(champ_id)
        specs.append(
            PolicySpec(
                name="league_champion",
                agent_policy="league_snapshot",
                source="champion",
                league_member_id=champ_id,
                agent_kwargs={"member_dir": str(champ_dir)},
            )
        )

        # Top-k (may overlap with champion — that's fine, champion is
        # reported separately so we include it again in top-k for
        # consistency with the ranked list)
        for m in ranked[:top_k]:
            mid = m["member_id"]
            mdir = registry.load_member(mid)
            specs.append(
                PolicySpec(
                    name=f"league_{mid}",
                    agent_policy="league_snapshot",
                    source="league",
                    league_member_id=mid,
                    agent_kwargs={"member_dir": str(mdir)},
                )
            )

    return specs
