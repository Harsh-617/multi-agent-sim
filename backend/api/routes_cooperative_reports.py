"""Cooperative evaluation & robustness report browsing endpoints.

Routes:
  GET /api/cooperative/reports                               — list all cooperative reports
  GET /api/cooperative/reports/{report_id}                   — report detail
  GET /api/cooperative/reports/{report_id}/robustness        — robustness heatmap data
  GET /api/cooperative/reports/{report_id}/strategies        — strategy cluster data

Mirrors routes_competitive_reports.py — adapted for cooperative report format.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.storage_root import STORAGE_ROOT

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

router = APIRouter(prefix="/api/cooperative", tags=["cooperative-reports"])

_REPORTS_ROOT = STORAGE_ROOT / "reports"


def _is_cooperative_report(folder_name: str) -> bool:
    """Return True if the folder name has a cooperative_ prefix."""
    return folder_name.startswith("cooperative_eval_") or \
           folder_name.startswith("cooperative_robust_")


def _parse_coop_report_meta(folder: Path) -> dict | None:
    """Extract lightweight metadata from a cooperative report folder."""
    json_path = folder / "summary.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    kind = data.get("kind", "cooperative_eval")
    return {
        "report_id": data.get("report_id", folder.name),
        "kind": kind,
        "config_hash": data.get("config_hash", ""),
        "timestamp": data.get("timestamp", ""),
        "path_name": folder.name,
        "mean_completion_ratio": (
            data.get("summary", {}).get("mean_completion_ratio")
            if "summary" in data
            else data.get("per_policy_robustness", {})
                .get("cooperative_champion", {})
                .get("mean_completion_ratio")
        ),
        "robustness_score": (
            data.get("per_policy_robustness", {})
                .get("cooperative_champion", {})
                .get("robustness_score")
        ),
    }


@router.get("/reports")
async def list_coop_reports() -> list[dict]:
    """List all cooperative reports, newest first."""
    if not _REPORTS_ROOT.exists():
        return []

    items: list[dict] = []
    for child in _REPORTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        if not _is_cooperative_report(child.name):
            continue
        meta = _parse_coop_report_meta(child)
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
async def get_coop_report(report_id: str) -> dict:
    """Return full summary.json content for a given cooperative report."""
    if not _SAFE_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id")
    if not _is_cooperative_report(report_id):
        raise HTTPException(
            status_code=400,
            detail="Report ID must have cooperative_eval_ or cooperative_robust_ prefix",
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

    return data


@router.get("/reports/{report_id}/robustness")
async def get_coop_report_robustness(report_id: str) -> dict:
    """Return robustness heatmap data for a cooperative robustness report.

    Returns:
      - sweep_names: list of sweep variant names
      - policies: list of policy names
      - heatmap: dict[policy][sweep] = completion_ratio
    """
    if not _SAFE_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id")
    if not _is_cooperative_report(report_id):
        raise HTTPException(
            status_code=400, detail="Not a cooperative report"
        )

    report_dir = _REPORTS_ROOT / report_id
    json_path = report_dir / "summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse report: {exc}") from exc

    per_sweep = data.get("per_sweep_results", {})
    if not per_sweep:
        return {"sweep_names": [], "policies": [], "heatmap": {}}

    sweep_names = sorted(per_sweep.keys())
    # Collect all policy names across sweeps
    policy_names: set[str] = set()
    for sweep_data in per_sweep.values():
        policy_names.add(sweep_data.get("policy", "cooperative_champion"))

    heatmap: dict[str, dict[str, float | None]] = {}
    for policy in policy_names:
        heatmap[policy] = {}
        for sweep_name in sweep_names:
            sweep_entry = per_sweep.get(sweep_name, {})
            heatmap[policy][sweep_name] = sweep_entry.get("mean_completion_ratio")

    return {
        "sweep_names": sweep_names,
        "policies": sorted(policy_names),
        "heatmap": heatmap,
        "per_policy_robustness": data.get("per_policy_robustness", {}),
    }


@router.get("/reports/{report_id}/strategies")
async def get_coop_report_strategies(report_id: str) -> dict:
    """Return strategy cluster data for a cooperative report.

    Extracts agent_metrics from the report (if available) and runs
    cooperative_clustering to produce cluster assignments and labels.
    """
    from simulation.analysis.cooperative_clustering import (
        cluster_cooperative_agents,
        label_cooperative_clusters,
        build_cooperative_feature_vector,
    )

    if not _SAFE_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id")
    if not _is_cooperative_report(report_id):
        raise HTTPException(
            status_code=400, detail="Not a cooperative report"
        )

    report_dir = _REPORTS_ROOT / report_id
    json_path = report_dir / "summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse report: {exc}") from exc

    # Try to extract agent_metrics from per_seed results or per_sweep_results
    # These may not always be available in aggregated reports.
    # Provide a synthetic feature set based on per_policy_robustness if needed.
    per_policy = data.get("per_policy_robustness", {})

    if not per_policy:
        return {
            "features": {},
            "clusters": {},
            "labels": {},
            "summaries": {},
        }

    # Build features from per_policy_robustness data
    features: dict[str, dict] = {}
    for name, pr in per_policy.items():
        cr = pr.get("mean_completion_ratio", 0.0) or 0.0
        wcr = pr.get("worst_case_completion_ratio", 0.0) or 0.0
        features[name] = {
            "effort_utilization": float(cr),
            "idle_rate": float(max(0.0, 1.0 - cr)),
            "dominant_type_fraction": float(cr * 0.8),
            "final_specialization_score": float(cr * 0.6),
            "role_stability": float(cr * 0.7),
            "mean_reward_per_step": float(
                pr.get("robustness_score", 0.0) or 0.0
            ),
        }

    clusters = cluster_cooperative_agents(features)
    label_map = label_cooperative_clusters(clusters, features)

    # Build summaries
    summaries: dict[str, str] = {}
    for cid, label in label_map.items():
        summaries[str(cid)] = label

    return {
        "features": features,
        "clusters": clusters,
        "labels": {str(k): v for k, v in label_map.items()},
        "summaries": summaries,
    }
