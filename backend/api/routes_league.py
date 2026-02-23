"""League member and rating endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simulation.agents import create_agent
from simulation.agents.base import BaseAgent
from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.envs.mixed import MixedEnvironment
from simulation.league.ratings import load_ratings, save_ratings, compute_ratings
from simulation.league.registry import LeagueRegistry
from simulation.metrics.collector import MetricsCollector

router = APIRouter(prefix="/api/league", tags=["league"])

LEAGUE_ROOT = Path("storage/agents/league")
RATINGS_PATH = LEAGUE_ROOT / "ratings.json"
CONFIGS_DIR = Path("storage/configs")
PPO_AGENT_DIR = Path("storage/agents/ppo_shared")

_registry = LeagueRegistry(LEAGUE_ROOT)

_DEFAULT_RATING = 1000.0


# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------


@router.get("/members")
async def list_members() -> list[dict]:
    """Return metadata for every league member."""
    return _registry.list_members()


@router.get("/members/{member_id}")
async def get_member(member_id: str) -> dict:
    """Return metadata for a single league member."""
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


@router.get("/ratings")
async def get_ratings() -> list[dict]:
    """Return saved ratings, or empty list if none computed yet."""
    return load_ratings(RATINGS_PATH)


@router.post("/ratings/recompute")
async def recompute_ratings(req: RecomputeRatingsRequest) -> list[dict]:
    """Recompute Elo ratings for all league members and persist them."""
    members = _registry.list_members()
    if not members:
        return []

    ratings = compute_ratings(
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
    """Return the highest-rated league member."""
    members = _registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No league members exist.")
    ratings = _ratings_map()
    champ = _find_champion(members, ratings)
    if champ is None:
        raise HTTPException(status_code=404, detail="No league members exist.")
    return champ


# ------------------------------------------------------------------
# Champion vs Baseline benchmark
# ------------------------------------------------------------------

_BASELINE_POLICIES = [
    "random",
    "always_cooperate",
    "always_extract",
    "tit_for_tat",
]


class ChampionBenchmarkRequest(BaseModel):
    config_id: str
    episodes: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)


def _run_episode_sync(
    config: MixedEnvironmentConfig,
    agent_policy: str,
    seed: int,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Run one episode synchronously and return summary stats."""
    env = MixedEnvironment(config)
    collector = MetricsCollector(config.instrumentation)

    initial_obs = env.reset()
    _kw = agent_kwargs or {}
    agents: dict[str, BaseAgent] = {}
    for i, aid in enumerate(env.active_agents()):
        agent = create_agent(agent_policy, **_kw)
        agent.reset(aid, derive_seed(seed, i))
        agents[aid] = agent

    observations: dict[str, dict] = dict(initial_obs)
    shared_pool = config.population.initial_shared_pool

    while not env.is_done():
        actions: dict[str, Any] = {}
        for aid in env.active_agents():
            obs = observations.get(aid, {})
            actions[aid] = agents[aid].act(obs)

        results = env.step(actions)
        for aid, sr in results.items():
            observations[aid] = sr.observation

        agent_resources = {
            aid: results[aid].observation.get("own_resources", 0.0)
            for aid in results
        }
        shared_pool = next(iter(results.values())).observation.get(
            "shared_pool", 0.0
        )

        collector.collect_step(
            step=env.current_step,
            actions=actions,
            results=results,
            shared_pool=shared_pool,
            agent_resources=agent_resources,
            active_agents=env.active_agents(),
        )

    reason = env.termination_reason()
    summary = collector.episode_summary(
        episode_length=env.current_step,
        termination_reason=reason,
        final_shared_pool=shared_pool,
    )
    return summary


def _benchmark_policy(
    config: MixedEnvironmentConfig,
    agent_policy: str,
    episodes: int,
    base_seed: int,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Run *episodes* episodes for a single policy and aggregate results."""
    total_rewards: list[float] = []
    final_pools: list[float] = []
    episode_lengths: list[int] = []
    collapses = 0

    for ep in range(episodes):
        ep_seed = derive_seed(base_seed, ep)
        summary = _run_episode_sync(
            config, agent_policy, ep_seed, agent_kwargs=agent_kwargs
        )
        rewards = summary.get("total_reward_per_agent", {})
        mean_r = (
            sum(rewards.values()) / len(rewards) if rewards else 0.0
        )
        total_rewards.append(mean_r)
        final_pools.append(summary.get("final_shared_pool", 0.0))
        episode_lengths.append(summary.get("episode_length", 0))
        if summary.get("termination_reason") == "system_collapse":
            collapses += 1

    n = max(len(total_rewards), 1)
    return {
        "mean_total_reward": round(sum(total_rewards) / n, 4),
        "mean_final_shared_pool": round(sum(final_pools) / n, 4),
        "collapse_rate": round(collapses / n, 4),
        "mean_episode_length": round(sum(episode_lengths) / n, 2),
    }


@router.post("/champion/benchmark")
async def champion_benchmark(req: ChampionBenchmarkRequest) -> dict:
    """Benchmark the champion against baseline policies."""
    # Validate config
    config_path = CONFIGS_DIR / f"{req.config_id}.json"
    if not config_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Config {req.config_id} not found."
        )
    config = MixedEnvironmentConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )

    # Find champion
    members = _registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No league members exist.")
    ratings = _ratings_map()
    champ = _find_champion(members, ratings)
    if champ is None:
        raise HTTPException(status_code=404, detail="No league members exist.")

    results: list[dict] = []

    # Champion
    member_dir = _registry.load_member(champ["member_id"])
    champ_result = _benchmark_policy(
        config,
        "league_snapshot",
        req.episodes,
        req.seed,
        agent_kwargs={"member_dir": str(member_dir)},
    )
    champ_result["policy"] = "league_champion"
    results.append(champ_result)

    # Baselines
    for policy in _BASELINE_POLICIES:
        res = _benchmark_policy(config, policy, req.episodes, req.seed)
        res["policy"] = policy
        results.append(res)

    # PPO (only if artifacts exist)
    if (PPO_AGENT_DIR / "policy.pt").exists() and (
        PPO_AGENT_DIR / "metadata.json"
    ).exists():
        res = _benchmark_policy(config, "ppo_shared", req.episodes, req.seed)
        res["policy"] = "ppo_shared"
        results.append(res)

    return {"champion": champ, "results": results}
