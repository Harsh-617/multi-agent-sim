"""Competitive league member and rating endpoints."""

from __future__ import annotations

import asyncio
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simulation.agents.competitive_baselines import create_competitive_agent
from simulation.config.competitive_defaults import default_competitive_config
from simulation.core.seeding import derive_seed
from simulation.envs.competitive.env import CompetitiveEnvironment
from simulation.league.ratings import load_ratings, save_ratings, START_RATING, elo_update
from simulation.league.registry import LeagueRegistry

from backend.storage_root import STORAGE_ROOT

router = APIRouter(prefix="/api/competitive/league", tags=["competitive-league"])

LEAGUE_ROOT = STORAGE_ROOT / "agents/competitive_league"
RATINGS_PATH = LEAGUE_ROOT / "ratings.json"

_registry = LeagueRegistry(LEAGUE_ROOT)

_DEFAULT_RATING = 1000.0


# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------


@router.get("/members")
async def list_members() -> list[dict]:
    """Return metadata for every competitive league member."""
    return _registry.list_members()


@router.get("/members/{member_id}")
async def get_member(member_id: str) -> dict:
    """Return metadata for a single competitive league member."""
    try:
        return _registry.get_member_metadata(member_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown member: {member_id}")


# ------------------------------------------------------------------
# Ratings
# ------------------------------------------------------------------


class RecomputeRatingsRequest(BaseModel):
    num_matches: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)


def _run_competitive_eval_episode(
    agent: Any,
    seed: int,
) -> float:
    """Run one competitive episode with *agent* as agent_0 vs baseline opponents.

    Returns the total return for agent_0.
    """
    config = default_competitive_config(seed=seed)
    config.population.num_agents = 3
    config.population.max_steps = 50
    env = CompetitiveEnvironment(config)
    obs = env.reset(seed=seed)

    agent_ids = list(obs.keys())
    eval_id = agent_ids[0]
    opp_ids = agent_ids[1:]

    agent.reset(eval_id, seed)

    opponents: dict[str, Any] = {}
    for i, oid in enumerate(opp_ids):
        opp = create_competitive_agent("always_build")
        opp.reset(oid, derive_seed(seed, i + 100))
        opponents[oid] = opp

    total_return = 0.0
    while not env.is_done():
        actions: dict[str, Any] = {}
        if eval_id in obs:
            actions[eval_id] = agent.act(obs[eval_id])
        for oid in opp_ids:
            if oid in obs:
                actions[oid] = opponents[oid].act(obs[oid])

        results = env.step(actions)
        obs = {aid: sr.observation for aid, sr in results.items()}
        if eval_id in results:
            total_return += results[eval_id].reward

    return total_return


def _make_competitive_league_agent(registry: LeagueRegistry, member_id: str) -> Any:
    """Create an agent from a competitive league snapshot."""
    member_dir = registry.load_member(member_id)
    return create_competitive_agent("competitive_ppo", member_dir=str(member_dir))


def _compute_competitive_ratings(
    registry: LeagueRegistry,
    num_matches: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings for all competitive league members.

    Same algorithm as ratings.compute_ratings but uses CompetitiveEnvironment.
    """
    members = registry.list_members()
    if not members:
        return {}

    member_ids = [m["member_id"] for m in members]
    ratings: dict[str, float] = {mid: START_RATING for mid in member_ids}

    if len(member_ids) == 1:
        return ratings

    for match_idx in range(num_matches):
        for id_a, id_b in combinations(member_ids, 2):
            ep_seed = derive_seed(seed, match_idx * 1000 + hash(id_a + id_b) % 999)

            agent_a = _make_competitive_league_agent(registry, id_a)
            agent_b = _make_competitive_league_agent(registry, id_b)

            ret_a = _run_competitive_eval_episode(agent_a, ep_seed)
            ret_b = _run_competitive_eval_episode(agent_b, ep_seed)

            if ret_a > ret_b:
                score_a = 1.0
            elif ret_b > ret_a:
                score_a = 0.0
            else:
                score_a = 0.5

            ratings[id_a], ratings[id_b] = elo_update(
                ratings[id_a], ratings[id_b], score_a
            )

    return ratings


@router.get("/ratings")
async def get_ratings() -> list[dict]:
    """Return saved ratings, or empty list if none computed yet."""
    return load_ratings(RATINGS_PATH)


@router.post("/ratings/recompute")
async def recompute_ratings(req: RecomputeRatingsRequest) -> list[dict]:
    """Recompute Elo ratings for all competitive league members and persist them."""
    members = _registry.list_members()
    if len(members) < 2:
        raise HTTPException(
            status_code=409,
            detail=f"Ratings require at least 2 league members; found {len(members)}.",
        )

    ratings = await asyncio.to_thread(
        _compute_competitive_ratings,
        _registry,
        num_matches=req.num_matches,
        seed=req.seed,
    )
    save_ratings(RATINGS_PATH, ratings)
    return load_ratings(RATINGS_PATH)


# ------------------------------------------------------------------
# Lineage
# ------------------------------------------------------------------


def _ratings_map() -> dict[str, float]:
    """Load saved ratings as {member_id: rating}."""
    data = load_ratings(RATINGS_PATH)
    return {r["member_id"]: r["rating"] for r in data}


@router.get("/lineage")
async def get_lineage() -> dict:
    """Return all members enriched with their Elo rating."""
    members = _registry.list_members()
    ratings = _ratings_map()
    enriched = []
    for m in members:
        enriched.append({
            "member_id": m["member_id"],
            "parent_id": m.get("parent_id"),
            "created_at": m.get("created_at"),
            "notes": m.get("notes"),
            "rating": ratings.get(m["member_id"], _DEFAULT_RATING),
        })
    enriched.sort(key=lambda x: x["member_id"])
    return {"members": enriched}


# ------------------------------------------------------------------
# Champion
# ------------------------------------------------------------------


def _find_champion(members: list[dict], ratings: dict[str, float]) -> dict | None:
    """Return the member with the highest rating (tie-break: newest)."""
    if not members:
        return None
    best = None
    best_rating = -1.0
    best_created = ""
    for m in members:
        r = ratings.get(m["member_id"], _DEFAULT_RATING)
        created = m.get("created_at") or ""
        if r > best_rating or (r == best_rating and created > best_created):
            best = m
            best_rating = r
            best_created = created
    if best is None:
        return None
    return {
        "member_id": best["member_id"],
        "rating": best_rating,
        "parent_id": best.get("parent_id"),
        "created_at": best.get("created_at"),
        "notes": best.get("notes"),
    }


@router.get("/champion")
async def get_champion() -> dict:
    """Return the highest-rated competitive league member."""
    members = _registry.list_members()
    if not members:
        return {"member_id": None}
    ratings = _ratings_map()
    champ = _find_champion(members, ratings)
    if champ is None:
        return {"member_id": None}
    return champ
