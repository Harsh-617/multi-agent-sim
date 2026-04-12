"""Cooperative archetype run-browsing and replay endpoints.

Routes:
  GET /api/cooperative/runs                        — list all cooperative runs
  GET /api/cooperative/runs/{run_id}               — run detail + episode summary
  GET /api/cooperative/runs/{run_id}/summary       — episode summary JSON only
  GET /api/cooperative/runs/{run_id}/replay        — stream metrics.jsonl as SSE
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.storage_root import STORAGE_ROOT

router = APIRouter(prefix="/api/cooperative", tags=["cooperative"])

RUNS_DIR = STORAGE_ROOT / "runs"

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cooperative_run(run_dir: Path) -> bool:
    """Return True when config.json declares environment_type == 'cooperative'."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("identity", {}).get("environment_type") == "cooperative"
    except (json.JSONDecodeError, OSError):
        return False


def _read_config_meta(run_dir: Path) -> dict:
    """Read lightweight metadata from config.json."""
    config_path = run_dir / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    identity = data.get("identity", {})
    pop = data.get("population", {})
    return {
        "run_id": run_dir.name,
        "seed": identity.get("seed"),
        "num_agents": pop.get("num_agents"),
        "max_steps": pop.get("max_steps"),
        "num_task_types": pop.get("num_task_types"),
        "agent_policy": data.get("_agent_policy"),
        "written_at": data.get("written_at"),
    }


def _read_episode_summary(run_dir: Path) -> dict | None:
    """Read episode_summary.json, returning None if missing or malformed."""
    summary_path = run_dir / "episode_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data.pop("written_at", None)
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _validate_run_id(run_id: str) -> None:
    if not _SAFE_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")


def _require_cooperative_run(run_id: str) -> Path:
    _validate_run_id(run_id)
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if not _is_cooperative_run(run_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' is not a cooperative run.",
        )
    return run_dir


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_cooperative_runs() -> list[dict]:
    """List all cooperative runs, newest first (by written_at timestamp)."""
    if not RUNS_DIR.exists():
        return []

    items: list[dict] = []
    for child in RUNS_DIR.iterdir():
        if not child.is_dir():
            continue
        if not _is_cooperative_run(child):
            continue
        meta = _read_config_meta(child)
        summary = _read_episode_summary(child)
        if summary is not None:
            meta["termination_reason"] = summary.get("termination_reason")
            meta["episode_length"] = summary.get("episode_length")
            meta["completion_ratio"] = summary.get("completion_ratio")
        else:
            meta["termination_reason"] = None
            meta["episode_length"] = None
            meta["completion_ratio"] = None
        items.append(meta)

    # Sort newest first by written_at, fallback to run_id lexicographic
    items.sort(key=lambda r: r.get("written_at") or "", reverse=True)
    return items


@router.get("/runs/{run_id}")
async def get_cooperative_run(run_id: str) -> dict:
    """Return run metadata and full episode summary for a cooperative run."""
    run_dir = _require_cooperative_run(run_id)
    meta = _read_config_meta(run_dir)
    summary = _read_episode_summary(run_dir)
    meta["episode_summary"] = summary
    return meta


@router.get("/runs/{run_id}/summary")
async def get_cooperative_run_summary(run_id: str) -> dict:
    """Return the episode summary for a cooperative run."""
    run_dir = _require_cooperative_run(run_id)
    summary = _read_episode_summary(run_dir)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode summary not yet available for run '{run_id}'.",
        )
    return summary


@router.get("/runs/{run_id}/replay")
async def replay_cooperative_run(run_id: str) -> StreamingResponse:
    """Stream stored metrics.jsonl step-by-step as Server-Sent Events.

    Each SSE event contains one step's metrics (all agents at that step).
    The final event is type=done with the episode_summary.
    """
    run_dir = _require_cooperative_run(run_id)
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No metrics recorded for run '{run_id}'.",
        )

    async def _generate():
        # Read and parse all metric records
        try:
            raw = metrics_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        from collections import defaultdict
        by_step: dict[int, list[dict]] = defaultdict(list)
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_step[rec.get("step", 0)].append(rec)
            except json.JSONDecodeError:
                continue

        # Stream step-by-step
        for step in sorted(by_step.keys()):
            payload = json.dumps({
                "type": "step",
                "run_id": run_id,
                "t": step,
                "metrics": by_step[step],
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.02)

        # Final done message with episode summary
        summary = _read_episode_summary(run_dir)
        done_payload = json.dumps({
            "type": "done",
            "run_id": run_id,
            "termination_reason": (
                summary.get("termination_reason") if summary else None
            ),
            "episode_summary": summary,
        })
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
