"""Resolve a set of PolicySpec objects for competitive evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from simulation.league.competitive_sampling import COMPETITIVE_BASELINE_POLICIES
from simulation.league.registry import LeagueRegistry


_DEFAULT_LEAGUE_ROOT = Path("storage/agents/competitive_league")
_DEFAULT_PPO_DIR = Path("storage/agents/competitive_ppo")


@dataclass(frozen=True, slots=True)
class CompetitivePolicySpec:
    """Describes one competitive policy to evaluate."""

    name: str
    agent_policy: str
    source: Literal["baseline", "ppo", "league"]
    league_member_id: str | None = None
    available: bool = True
    skip_reason: str | None = None
    agent_kwargs: dict = field(default_factory=dict)


def get_competitive_policy_specs(
    *,
    league_root: Path = _DEFAULT_LEAGUE_ROOT,
    ppo_dir: Path = _DEFAULT_PPO_DIR,
    top_k: int = 3,
) -> list[CompetitivePolicySpec]:
    """Build the full list of competitive policies to evaluate.

    Returns both available and unavailable specs so callers can
    report which policies were skipped and why.
    """
    specs: list[CompetitivePolicySpec] = []

    # --- baselines (always available) ---
    for p in COMPETITIVE_BASELINE_POLICIES:
        specs.append(
            CompetitivePolicySpec(name=p, agent_policy=p, source="baseline")
        )

    # --- competitive_ppo ---
    ppo_policy = ppo_dir / "policy.pt"
    if ppo_policy.exists():
        specs.append(
            CompetitivePolicySpec(
                name="competitive_ppo",
                agent_policy="competitive_ppo",
                source="ppo",
                agent_kwargs={"agent_dir": str(ppo_dir)},
            )
        )
    else:
        specs.append(
            CompetitivePolicySpec(
                name="competitive_ppo",
                agent_policy="competitive_ppo",
                source="ppo",
                available=False,
                skip_reason="Missing artifacts (policy.pt)",
            )
        )

    # --- league members (top-K) ---
    registry = LeagueRegistry(league_root)
    members = registry.list_members()

    if not members:
        specs.append(
            CompetitivePolicySpec(
                name="league_top",
                agent_policy="league_snapshot",
                source="league",
                available=False,
                skip_reason="No competitive league members exist",
            )
        )
    else:
        # Sort by created_at descending as proxy for recency/quality
        ranked = sorted(
            members,
            key=lambda m: m.get("created_at", ""),
            reverse=True,
        )

        for m in ranked[:top_k]:
            mid = m["member_id"]
            mdir = registry.load_member(mid)
            specs.append(
                CompetitivePolicySpec(
                    name=f"league_{mid}",
                    agent_policy="league_snapshot",
                    source="league",
                    league_member_id=mid,
                    agent_kwargs={"member_dir": str(mdir)},
                )
            )

    return specs
