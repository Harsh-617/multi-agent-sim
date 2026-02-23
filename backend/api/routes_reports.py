"""Evaluation & robustness report browsing endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["reports"])

_REPORTS_ROOT = Path("storage/reports")


def _classify_report(folder_name: str) -> str:
    """Return 'eval' or 'robust' based on folder-name prefix."""
    if folder_name.startswith("robust_"):
        return "robust"
    return "eval"


def _parse_report_meta(folder: Path) -> dict | None:
    """Extract lightweight metadata from a report folder."""
    json_path = folder / "report.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    return {
        "report_id": data.get("report_id", folder.name),
        "kind": _classify_report(folder.name),
        "config_hash": data.get("config_hash", ""),
        "timestamp": data.get("timestamp", ""),
        "path_name": folder.name,
    }


@router.get("/reports")
async def list_reports() -> list[dict]:
    """List all evaluation/robustness reports, newest first."""
    if not _REPORTS_ROOT.exists():
        return []

    items: list[dict] = []
    for child in _REPORTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        meta = _parse_report_meta(child)
        if meta is not None:
            items.append(meta)

    items.sort(key=lambda r: r["timestamp"], reverse=True)
    return items


@router.get("/reports/{report_id}")
async def get_report(report_id: str) -> dict:
    """Return full report.json content for a given report."""
    report_dir = _REPORTS_ROOT / report_id
    if not report_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    json_path = report_dir / "report.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404, detail=f"report.json missing in '{report_id}'"
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse report: {exc}"
        ) from exc

    data["kind"] = _classify_report(report_id)
    return data


@router.get("/reports/{report_id}/strategies")
async def get_strategies(report_id: str) -> dict:
    """Return strategy analysis (features, clusters, labels) for a report."""
    from simulation.analysis.strategy_features import extract_features
    from simulation.analysis.strategy_clustering import cluster_policies
    from simulation.analysis.strategy_labels import cluster_summaries, label_clusters

    report_dir = _REPORTS_ROOT / report_id
    if not report_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    json_path = report_dir / "report.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404, detail=f"report.json missing in '{report_id}'"
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse report: {exc}"
        ) from exc

    data["kind"] = _classify_report(report_id)

    features = extract_features(data)
    clusters = cluster_policies(features)
    labels = label_clusters(clusters, features)
    summaries = cluster_summaries(labels)

    return {
        "features": features,
        "clusters": clusters,
        "labels": {str(k): v for k, v in labels.items()},
        "summaries": {str(k): v for k, v in summaries.items()},
    }
