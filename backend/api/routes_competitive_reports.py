"""Competitive evaluation & robustness report browsing endpoints."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.storage_root import STORAGE_ROOT

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

router = APIRouter(prefix="/api/competitive", tags=["competitive-reports"])

_REPORTS_ROOT = STORAGE_ROOT / "reports"

LEAGUE_ROOT = STORAGE_ROOT / "agents/competitive_league"
RATINGS_PATH = LEAGUE_ROOT / "ratings.json"
PPO_DIR = STORAGE_ROOT / "agents/competitive_ppo"


def _is_competitive_report(folder_name: str) -> bool:
    """Return True if the folder name has the competitive_ prefix."""
    return folder_name.startswith("competitive_")


def _parse_competitive_report_meta(folder: Path) -> dict | None:
    """Extract lightweight metadata from a competitive report folder."""
    json_path = folder / "summary.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    return {
        "report_id": data.get("report_id", folder.name),
        "kind": "competitive",
        "config_hash": data.get("config_hash", ""),
        "timestamp": data.get("timestamp", ""),
        "path_name": folder.name,
    }


@router.get("/reports")
async def list_competitive_reports() -> list[dict]:
    """List all competitive reports, newest first."""
    if not _REPORTS_ROOT.exists():
        return []

    items: list[dict] = []
    for child in _REPORTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        if not _is_competitive_report(child.name):
            continue
        meta = _parse_competitive_report_meta(child)
        if meta is not None:
            items.append(meta)

    _EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)

    def _ts_key(r: dict) -> datetime:
        ts = r.get("timestamp") or ""
        try:
            dt = datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return _EPOCH_MIN

    items.sort(key=_ts_key, reverse=True)
    return items


@router.get("/reports/{report_id}")
async def get_competitive_report(report_id: str) -> dict:
    """Return full summary.json content for a given competitive report."""
    if not _SAFE_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id")
    if not _is_competitive_report(report_id):
        raise HTTPException(
            status_code=400,
            detail="Report ID must have competitive_ prefix",
        )
    report_dir = _REPORTS_ROOT / report_id
    if not report_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    json_path = report_dir / "summary.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404, detail=f"summary.json missing in '{report_id}'"
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse report: {exc}"
        ) from exc

    data["kind"] = "competitive"
    return data


@router.get("/reports/{report_id}/strategies")
async def get_competitive_strategies(report_id: str) -> dict:
    """Return strategy analysis (features, clusters, labels) for a competitive report."""
    from simulation.analysis.competitive_strategy_features import (
        extract_competitive_strategy_features,
    )
    from simulation.analysis.competitive_strategy_clustering import (
        cluster_competitive_strategies,
    )
    from simulation.analysis.competitive_strategy_labels import (
        competitive_cluster_summaries,
        get_competitive_strategy_label,
    )

    if not _SAFE_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id")
    if not _is_competitive_report(report_id):
        raise HTTPException(
            status_code=400,
            detail="Report ID must have competitive_ prefix",
        )
    report_dir = _REPORTS_ROOT / report_id
    if not report_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    json_path = report_dir / "summary.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404, detail=f"summary.json missing in '{report_id}'"
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse report: {exc}"
        ) from exc

    # Build per-policy input for feature extraction from per_policy_robustness
    per_policy = data.get("per_policy_robustness", {})
    policy_results: dict[str, dict] = {}
    for name, pr in per_policy.items():
        policy_results[name] = {
            "mean_reward": pr.get("overall_mean_reward"),
            "worst_case_reward": pr.get("worst_case_mean_reward"),
            "robustness_score": pr.get("robustness_score"),
            "winner_rate": pr.get("mean_winner_rate"),
            "sweep_rewards": None,  # not stored per-sweep in aggregate
        }

    features = extract_competitive_strategy_features(policy_results)
    clusters = cluster_competitive_strategies(features)

    # Group features by cluster to compute labels
    labels: dict[int, str] = {}
    for cid in set(clusters.values()):
        cluster_feats = {
            name: features[name]
            for name, c in clusters.items()
            if c == cid
        }
        labels[cid] = get_competitive_strategy_label(cluster_feats)

    summaries = competitive_cluster_summaries(labels)

    return {
        "features": features,
        "clusters": clusters,
        "labels": {str(k): v for k, v in labels.items()},
        "summaries": {str(k): v for k, v in summaries.items()},
    }


# ------------------------------------------------------------------
# Champion robustness evaluation
# ------------------------------------------------------------------


class CompetitiveChampionRobustnessRequest(BaseModel):
    config_id: str = Field(default="default")
    seeds: int = Field(default=3, ge=1, le=20)
    episodes_per_seed: int = Field(default=2, ge=1, le=10)
    limit_sweeps: int | None = Field(default=None, ge=1)
    seed: int = Field(default=42)


@router.post("/league/champion/robustness")
async def competitive_champion_robustness(
    req: CompetitiveChampionRobustnessRequest,
) -> dict:
    """Run a robustness sweep for the competitive champion and save a report.

    Returns the report_id (folder name) and report_path so the caller
    can navigate to /competitive/reports/{report_id}.
    """
    from simulation.config.competitive_defaults import default_competitive_config
    from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
    from simulation.evaluation.competitive_policy_set import (
        get_competitive_policy_specs,
    )
    from simulation.evaluation.competitive_robustness import (
        run_competitive_robustness,
    )
    from simulation.evaluation.competitive_reporting import write_competitive_report
    from simulation.evaluation.competitive_sweeps import (
        build_competitive_default_sweeps,
    )
    from simulation.league.ratings import load_ratings
    from simulation.league.registry import LeagueRegistry

    # 1. Load config
    if req.config_id == "default":
        config = default_competitive_config()
    else:
        config_path = STORAGE_ROOT / "configs" / f"{req.config_id}.json"
        if not config_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Config '{req.config_id}' not found."
            )
        config = CompetitiveEnvironmentConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )

    # 2. Find champion
    registry = LeagueRegistry(LEAGUE_ROOT)
    members = registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No competitive league members exist.")

    ratings_data = load_ratings(RATINGS_PATH)
    ratings_map = {r["member_id"]: r["rating"] for r in ratings_data}

    # Find champion: highest rating, tie-break newest
    best = None
    best_rating = -1.0
    best_created = ""
    for m in members:
        r = ratings_map.get(m["member_id"], 1000.0)
        created = m.get("created_at") or ""
        if r > best_rating or (r == best_rating and created > best_created):
            best = m
            best_rating = r
            best_created = created

    if best is None:
        raise HTTPException(status_code=404, detail="No competitive league members exist.")

    # 3. Build policy set (top_k=0 avoids duplicate top-k entries)
    specs = get_competitive_policy_specs(
        league_root=LEAGUE_ROOT,
        ppo_dir=PPO_DIR,
        top_k=0,
    )

    # 4. Build and optionally cap sweeps
    sweeps = build_competitive_default_sweeps(config)
    if req.limit_sweeps is not None:
        sweeps = sweeps[: req.limit_sweeps]

    # 5. Derive seed list
    seeds = [req.seed + i for i in range(req.seeds)]

    # 6. Run competitive robustness evaluation
    result = await asyncio.to_thread(
        run_competitive_robustness,
        config,
        specs,
        seeds=seeds,
        episodes_per_seed=req.episodes_per_seed,
        sweeps=sweeps,
    )

    # 7. Persist report
    report_dir = write_competitive_report(result, output_dir=_REPORTS_ROOT)

    return {"report_id": report_dir.name, "report_path": str(report_dir)}
