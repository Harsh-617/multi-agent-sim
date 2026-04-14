"""Cooperative league member, rating, and evolution endpoints.

Routes:
  GET  /api/cooperative/league/members            — list all members with Elo ratings
  GET  /api/cooperative/league/members/{id}       — member detail + metadata
  GET  /api/cooperative/league/champion           — current Elo champion
  GET  /api/cooperative/league/lineage            — full lineage graph
  GET  /api/cooperative/league/evolution          — lineage + strategy labels + robustness
  POST /api/cooperative/league/champion/robustness — trigger robustness sweep on champion

Mirrors routes_competitive_league.py — adapted for cooperative registry and metrics.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel, Field

from simulation.agents.cooperative_baselines import (
    CooperativePPOAgent,
    create_cooperative_agent,
)
from simulation.analysis.cooperative_clustering import (
    cluster_cooperative_agents,
    label_cooperative_clusters,
    build_cooperative_feature_vector,
)
from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.seeding import derive_seed
from simulation.envs.cooperative.env import CooperativeEnvironment
from simulation.evaluation.cooperative_robustness import run_cooperative_robustness
from simulation.evaluation.cooperative_sweeps import build_cooperative_sweeps
from simulation.league.cooperative_ratings import (
    compute_cooperative_ratings,
    load_cooperative_ratings,
    save_cooperative_ratings,
)
from simulation.league.cooperative_registry import CooperativeLeagueRegistry

from backend.storage_root import STORAGE_ROOT

router = APIRouter(prefix="/api/cooperative/league", tags=["cooperative-league"])

LEAGUE_ROOT = STORAGE_ROOT / "agents/cooperative/league"
RATINGS_PATH = LEAGUE_ROOT / "ratings.json"
CONFIGS_DIR = STORAGE_ROOT / "configs"
COOPERATIVE_PPO_DIR = STORAGE_ROOT / "agents/cooperative/ppo_shared"
REPORTS_ROOT = STORAGE_ROOT / "reports"

_registry = CooperativeLeagueRegistry(LEAGUE_ROOT)

_DEFAULT_RATING = 1000.0


# ------------------------------------------------------------------
# Robustness manager (mirrors CompetitiveRobustnessManager)
# ------------------------------------------------------------------


class CoopRobustnessManager:
    """Process-wide cooperative robustness evaluation state."""

    def __init__(self) -> None:
        self.robustness_id: str | None = None
        self.running: bool = False
        self.stage: str = "idle"
        self.error: str | None = None
        self.report_id: str | None = None
        self._task: asyncio.Task[None] | None = None

    def attach_task(self, task: asyncio.Task[None]) -> None:
        self._task = task

    def reset_state(self) -> None:
        self.robustness_id = None
        self.running = False
        self.stage = "idle"
        self.error = None
        self.report_id = None
        self._task = None

    def set_stage(self, stage: str, detail: str = "") -> None:
        self.stage = stage


coop_robustness_manager = CoopRobustnessManager()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _ratings_map() -> dict[str, float]:
    """Load saved ratings as {member_id: rating}."""
    data = load_cooperative_ratings(RATINGS_PATH)
    return {r["member_id"]: r["rating"] for r in data}


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


def _load_newest_coop_robust_report() -> dict | None:
    """Load the newest cooperative robustness report JSON, or None."""
    if not REPORTS_ROOT.exists():
        return None
    robust_dirs = [
        d for d in REPORTS_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("cooperative_robust_")
    ]
    if not robust_dirs:
        return None
    newest = max(robust_dirs, key=lambda d: d.stat().st_mtime)
    report_path = newest / "summary.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _coop_evolution_clusters_and_labels(
    members: list[dict],
    ratings: dict[str, float],
) -> tuple[dict[str, int], dict[int, str]]:
    """Cluster members by Elo rating and assign cooperative strategy labels."""
    if not members:
        return {}, {}

    # Build synthetic feature dict: rating as mean_reward_per_step proxy
    synthetic_features: dict[str, dict[str, Any]] = {}
    for m in members:
        mid = m["member_id"]
        r = ratings.get(mid, _DEFAULT_RATING)
        # Normalize rating to [0,1] range for clustering
        norm_rating = max(0.0, (r - 800.0) / 600.0)
        synthetic_features[mid] = {
            "effort_utilization": norm_rating,
            "idle_rate": max(0.0, 1.0 - norm_rating),
            "dominant_type_fraction": norm_rating * 0.8,
            "final_specialization_score": norm_rating * 0.6,
            "role_stability": norm_rating * 0.7,
            "mean_reward_per_step": r,
        }

    clusters = cluster_cooperative_agents(synthetic_features)
    label_map = label_cooperative_clusters(clusters, synthetic_features)
    return clusters, label_map


# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------


class RecomputeCooperativeRatingsRequest(BaseModel):
    num_matches: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)


def _bg_recompute_cooperative_ratings(num_matches: int, seed: int) -> None:
    """Run cooperative rating recomputation in a background thread."""
    ratings = compute_cooperative_ratings(
        _registry,
        num_matches=num_matches,
        seed=seed,
    )
    save_cooperative_ratings(RATINGS_PATH, ratings)


@router.post("/ratings/recompute")
async def recompute_cooperative_ratings(
    background_tasks: BackgroundTasks,
    req: RecomputeCooperativeRatingsRequest = Body(default=RecomputeCooperativeRatingsRequest()),
) -> dict:
    """Start Elo rating recomputation for cooperative league members in the background.

    Returns immediately with ``{"status": "started"}``; the computation runs
    in a FastAPI background task so other API requests are not blocked.
    """
    members = _registry.list_members()
    if len(members) < 2:
        raise HTTPException(
            status_code=409,
            detail=f"Ratings require at least 2 league members; found {len(members)}.",
        )

    background_tasks.add_task(_bg_recompute_cooperative_ratings, req.num_matches, req.seed)
    return {"status": "started"}


@router.get("/members")
async def list_coop_members() -> list[dict]:
    """Return metadata for every cooperative league member."""
    members = _registry.list_members()
    ratings = _ratings_map()
    for m in members:
        m["rating"] = ratings.get(m["member_id"], _DEFAULT_RATING)
    return members


@router.get("/members/{member_id}")
async def get_coop_member(member_id: str) -> dict:
    """Return metadata for a single cooperative league member."""
    try:
        meta = _registry.get_member_metadata(member_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown member: {member_id}")
    ratings = _ratings_map()
    meta["rating"] = ratings.get(member_id, _DEFAULT_RATING)
    return meta


# ------------------------------------------------------------------
# Champion
# ------------------------------------------------------------------


@router.get("/champion")
async def get_coop_champion() -> dict:
    """Return the highest-rated cooperative league member."""
    members = _registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No cooperative league members exist.")
    ratings = _ratings_map()
    champ = _find_champion(members, ratings)
    if champ is None:
        raise HTTPException(status_code=404, detail="No cooperative league members exist.")
    return champ


# ------------------------------------------------------------------
# Lineage
# ------------------------------------------------------------------


@router.get("/lineage")
async def get_coop_lineage() -> dict:
    """Return all members enriched with Elo rating and strategy label."""
    members = _registry.list_members()
    ratings = _ratings_map()
    clusters, label_map = _coop_evolution_clusters_and_labels(members, ratings)
    enriched = []
    for m in members:
        mid = m["member_id"]
        enriched.append({
            "member_id": mid,
            "parent_id": m.get("parent_id"),
            "created_at": m.get("created_at"),
            "notes": m.get("notes"),
            "rating": ratings.get(mid, _DEFAULT_RATING),
            "label": label_map.get(clusters.get(mid, 0), "Developing"),
        })
    enriched.sort(key=lambda x: x["member_id"])
    return {"members": enriched}


# ------------------------------------------------------------------
# Evolution
# ------------------------------------------------------------------


@router.get("/evolution")
async def get_coop_evolution() -> dict:
    """Return league lineage enriched with strategy labels + robustness scores."""
    members = _registry.list_members()
    ratings = _ratings_map()

    # Load robustness data
    report = _load_newest_coop_robust_report()
    champ_rob_score: float | None = None
    if report is not None:
        per_policy = report.get("per_policy_robustness", {})
        entry = per_policy.get("cooperative_champion")
        if entry is not None:
            champ_rob_score = entry.get("robustness_score")

    champ = _find_champion(members, ratings)
    champ_id = champ["member_id"] if champ else None

    clusters, label_map = _coop_evolution_clusters_and_labels(members, ratings)

    result_members: list[dict] = []
    for m in sorted(members, key=lambda x: x["member_id"]):
        mid = m["member_id"]
        r = ratings.get(mid, _DEFAULT_RATING)
        cid = clusters.get(mid, 0)
        label = label_map.get(cid, "Developing")
        result_members.append({
            "member_id": mid,
            "parent_id": m.get("parent_id"),
            "created_at": m.get("created_at"),
            "notes": m.get("notes"),
            "rating": r,
            "strategy": {
                "cluster_id": cid,
                "label": label,
                "features": {},
            },
            "robustness_score": champ_rob_score if mid == champ_id else None,
        })

    champion_history: list[dict] = []
    for m in sorted(members, key=lambda x: x.get("created_at") or ""):
        mid = m["member_id"]
        r = ratings.get(mid, _DEFAULT_RATING)
        cid = clusters.get(mid, 0)
        label = label_map.get(cid, "Developing")
        champion_history.append({
            "member_id": mid,
            "created_at": m.get("created_at"),
            "rating": r,
            "label": label,
            "cluster_id": cid,
            "robustness_score": champ_rob_score if mid == champ_id else None,
        })

    return {"members": result_members, "champion_history": champion_history}


# ------------------------------------------------------------------
# Champion vs Baseline benchmark
# ------------------------------------------------------------------

_COOP_BASELINE_POLICIES = [
    "random",
    "always_work",
    "always_idle",
    "specialist",
    "balancer",
]


class CoopChampionBenchmarkRequest(BaseModel):
    config_id: str = Field(default="default")
    episodes: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42)


def _run_coop_benchmark_episode(
    config: CooperativeEnvironmentConfig,
    policy_name: str,
    seed: int,
    agent_dir: Path | None = None,
) -> dict:
    """Run one cooperative episode and return summary stats."""
    env = CooperativeEnvironment(config)
    obs = env.reset(seed=seed)
    agent_ids = env.active_agents()
    num_task_types = config.population.num_task_types

    agents: dict[str, Any] = {}
    for i, aid in enumerate(agent_ids):
        if policy_name == "cooperative_champion" and agent_dir is not None:
            a: Any = CooperativePPOAgent(agent_dir=agent_dir)
        else:
            a = create_cooperative_agent(policy_name, num_task_types=num_task_types)
        a.reset(aid, derive_seed(seed, i))
        agents[aid] = a

    total_reward = 0.0
    step = 0

    while not env.is_done():
        active = env.active_agents()
        actions: dict[str, Any] = {}
        for aid in active:
            actions[aid] = agents[aid].act(obs.get(aid, {}))

        results = env.step(actions)
        obs = {aid: sr.observation for aid, sr in results.items()}
        total_reward += sum(sr.reward for sr in results.values())
        step += 1

    state = env._state
    total_completed = float(sum(state.tasks_completed_total)) if state is not None else 0.0
    backlog = float(state.backlog_level) if state is not None else 0.0
    completion_ratio = total_completed / max(total_completed + backlog, 1.0)
    mean_return = total_reward / max(len(agent_ids), 1)

    return {
        "completion_ratio": completion_ratio,
        "mean_return": mean_return,
        "episode_length": step,
    }


def _benchmark_coop_policy(
    config: CooperativeEnvironmentConfig,
    policy_name: str,
    episodes: int,
    base_seed: int,
    agent_dir: Path | None = None,
) -> dict:
    """Run *episodes* cooperative episodes for a single policy and aggregate."""
    completion_ratios: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        ep_seed = derive_seed(base_seed, ep)
        result = _run_coop_benchmark_episode(
            config, policy_name, ep_seed, agent_dir=agent_dir
        )
        completion_ratios.append(result["completion_ratio"])
        returns.append(result["mean_return"])
        lengths.append(result["episode_length"])

    n = max(len(completion_ratios), 1)
    return {
        "mean_completion_ratio": round(sum(completion_ratios) / n, 4),
        "mean_return": round(sum(returns) / n, 4),
        "mean_episode_length": round(sum(lengths) / n, 2),
    }


@router.post("/champion/benchmark")
async def coop_champion_benchmark(req: CoopChampionBenchmarkRequest) -> dict:
    """Benchmark the cooperative champion against baseline policies."""
    if req.config_id == "default":
        config = default_cooperative_config()
    else:
        config_path = CONFIGS_DIR / f"{req.config_id}.json"
        if not config_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Config '{req.config_id}' not found."
            )
        raw = config_path.read_text(encoding="utf-8")
        raw_data = json.loads(raw)
        if raw_data.get("identity", {}).get("environment_type") != "cooperative":
            raise HTTPException(
                status_code=422,
                detail=f"Config '{req.config_id}' is not a cooperative config.",
            )
        config = CooperativeEnvironmentConfig.model_validate_json(raw)

    members = _registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No cooperative league members exist.")
    ratings = _ratings_map()
    champ = _find_champion(members, ratings)
    if champ is None:
        raise HTTPException(status_code=404, detail="No cooperative league members exist.")

    results: list[dict] = []

    # Champion
    champ_dir = _registry.load_member(champ["member_id"])
    champ_result = await asyncio.to_thread(
        _benchmark_coop_policy,
        config,
        "cooperative_champion",
        req.episodes,
        req.seed,
        champ_dir,
    )
    champ_result["policy"] = "cooperative_champion"
    results.append(champ_result)

    # Baselines
    for policy in _COOP_BASELINE_POLICIES:
        res = await asyncio.to_thread(
            _benchmark_coop_policy, config, policy, req.episodes, req.seed
        )
        res["policy"] = policy
        results.append(res)

    return {"champion": champ, "results": results}


# ------------------------------------------------------------------
# Champion robustness
# ------------------------------------------------------------------


class CoopChampionRobustnessRequest(BaseModel):
    config_id: str = Field(default="default")
    seeds: int = Field(default=3, ge=1, le=20)
    episodes_per_seed: int = Field(default=2, ge=1, le=10)
    limit_sweeps: int | None = Field(default=None, ge=1)
    seed: int = Field(default=42)


async def _run_coop_robustness_task(
    rm: CoopRobustnessManager,
    robustness_id: str,
    req: CoopChampionRobustnessRequest,
) -> None:
    """Run cooperative robustness evaluation in thread-pool executor."""
    loop = asyncio.get_running_loop()
    rm.running = True
    rm.robustness_id = robustness_id
    rm.stage = "loading_config"

    try:
        # 1. Load config
        if req.config_id == "default":
            config = default_cooperative_config()
        else:
            config_path = CONFIGS_DIR / f"{req.config_id}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config '{req.config_id}' not found.")
            config = CooperativeEnvironmentConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )

        # 2. Find champion
        members = _registry.list_members()
        if not members:
            raise ValueError("No cooperative league members exist.")
        ratings = _ratings_map()
        champ = _find_champion(members, ratings)
        if champ is None:
            raise ValueError("No cooperative league members exist.")

        champ_dir = _registry.load_member(champ["member_id"])

        # 3. Build sweeps
        sweeps = build_cooperative_sweeps()
        if req.limit_sweeps is not None:
            sweeps = sweeps[: req.limit_sweeps]

        # 4. Derive seed list
        seeds = [req.seed + i for i in range(req.seeds)]

        # 5. Run robustness evaluation
        rm.stage = "evaluating"
        report_dir = await loop.run_in_executor(
            None,
            lambda: run_cooperative_robustness(
                config,
                agent_dir=champ_dir,
                sweeps=sweeps,
                seeds=seeds,
                episodes_per_seed=req.episodes_per_seed,
                policy_name="cooperative_champion",
                report_root=REPORTS_ROOT,
            ),
        )

        rm.report_id = report_dir.name
        rm.stage = "done"
    except Exception as exc:  # noqa: BLE001
        rm.error = str(exc)
        rm.stage = "error"
    finally:
        rm.running = False


@router.post("/champion/robustness")
async def coop_champion_robustness(req: CoopChampionRobustnessRequest) -> dict:
    """Start a robustness sweep for the cooperative champion in the background."""
    if coop_robustness_manager.running:
        raise HTTPException(
            status_code=409, detail="A cooperative robustness evaluation is already running."
        )

    robustness_id = uuid.uuid4().hex[:12]
    coop_robustness_manager.reset_state()
    coop_robustness_manager.robustness_id = robustness_id

    task = asyncio.create_task(
        _run_coop_robustness_task(coop_robustness_manager, robustness_id, req)
    )
    coop_robustness_manager.attach_task(task)
    await asyncio.sleep(0)

    return {"robustness_id": robustness_id}


@router.get("/champion/robustness/{robustness_id}/status")
async def coop_robustness_status(robustness_id: str) -> dict:
    """Return the current status of a cooperative robustness evaluation."""
    if coop_robustness_manager.robustness_id != robustness_id:
        raise HTTPException(
            status_code=404,
            detail=f"Robustness run '{robustness_id}' not found.",
        )

    response: dict[str, Any] = {
        "robustness_id": robustness_id,
        "running": coop_robustness_manager.running,
        "stage": coop_robustness_manager.stage,
    }
    if coop_robustness_manager.error is not None:
        response["error"] = coop_robustness_manager.error
    if coop_robustness_manager.report_id is not None:
        response["report_id"] = coop_robustness_manager.report_id

    return response
