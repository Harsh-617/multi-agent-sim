"""Cooperative pipeline automation endpoints.

Routes:
  POST /api/cooperative/pipeline/run                     — trigger full cooperative pipeline
  GET  /api/cooperative/pipeline/status/{pipeline_id}    — pipeline status
  GET  /api/cooperative/pipeline/runs                    — list pipeline runs

Mirrors routes_pipeline.py competitive section — adapted for cooperative.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline.pipeline_manager import PipelineManager
from backend.storage_root import STORAGE_ROOT
from simulation.pipeline.cooperative_pipeline_run import run_cooperative_pipeline

router = APIRouter(prefix="/api/cooperative/pipeline", tags=["cooperative-pipeline"])

_PIPELINES_DIR = STORAGE_ROOT / "pipelines"

# Separate manager instance for cooperative pipelines
coop_pipeline_manager = PipelineManager()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class CoopPipelineRunRequest(BaseModel):
    config_id: str = Field(default="default", description='Config id or "default".')
    seed: int = Field(default=42, description="Master seed for reproducibility.")
    seeds: int = Field(default=3, ge=1, description="Number of evaluation seeds.")
    episodes_per_seed: int = Field(
        default=2, ge=1, description="Episodes per seed per sweep."
    )
    max_steps: int | None = Field(
        default=None, description="Optional max_steps override for faster evaluation."
    )
    total_timesteps: int = Field(
        default=50_000, ge=1, description="PPO training timesteps."
    )
    snapshot_every_timesteps: int = Field(
        default=10_000, ge=1, description="League snapshot interval."
    )
    max_league_members: int = Field(
        default=50, ge=1, description="Maximum league members to retain."
    )
    num_matches: int = Field(
        default=10, ge=1, description="Elo rating matches per member pair."
    )
    limit_sweeps: int | None = Field(
        default=None, description="Cap robustness sweeps for faster runs."
    )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _run_coop_pipeline_task(
    pm: PipelineManager,
    pipeline_id: str,
    kwargs: dict[str, Any],
) -> None:
    """Run cooperative pipeline in thread-pool executor."""
    loop = asyncio.get_running_loop()
    pm.running = True
    pm.pipeline_id = pipeline_id
    pm.stage = "loading_config"

    def _progress(stage: str, detail: str = "") -> None:
        if stage != "done":
            pm.set_stage(stage, detail)

    def _run() -> Path:
        return run_cooperative_pipeline(progress_callback=_progress, **kwargs)

    try:
        result_path = await loop.run_in_executor(None, _run)
        summary_file = result_path / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        pm.report_id = summary.get("report_id")
        pm.summary_path = str(summary_file)
        pm.stage = "done"
    except Exception as exc:  # noqa: BLE001
        pm.error = str(exc)
        pm.stage = "error"
    finally:
        pm.running = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run")
async def start_coop_pipeline(req: CoopPipelineRunRequest) -> dict:
    """Start the cooperative automation pipeline in the background.

    Returns ``{ pipeline_id }`` immediately; poll ``/status/{id}`` for progress.
    """
    if coop_pipeline_manager.running:
        raise HTTPException(
            status_code=409, detail="A cooperative pipeline is already running."
        )

    pipeline_id = uuid.uuid4().hex[:12]
    coop_pipeline_manager.reset_state()
    coop_pipeline_manager.pipeline_id = pipeline_id

    kwargs: dict[str, Any] = {
        "config_id": req.config_id,
        "seed": req.seed,
        "seeds": req.seeds,
        "episodes_per_seed": req.episodes_per_seed,
        "max_steps": req.max_steps,
        "total_timesteps": req.total_timesteps,
        "snapshot_every_timesteps": req.snapshot_every_timesteps,
        "max_league_members": req.max_league_members,
        "num_matches": req.num_matches,
        "limit_sweeps": req.limit_sweeps,
    }

    task = asyncio.create_task(
        _run_coop_pipeline_task(coop_pipeline_manager, pipeline_id, kwargs)
    )
    coop_pipeline_manager.attach_task(task)
    await asyncio.sleep(0)

    return {"pipeline_id": pipeline_id}


@router.get("/status/{pipeline_id}")
async def coop_pipeline_status(pipeline_id: str) -> dict:
    """Return the current status of a cooperative pipeline run."""
    if coop_pipeline_manager.pipeline_id != pipeline_id:
        raise HTTPException(
            status_code=404,
            detail=f"Cooperative pipeline '{pipeline_id}' not found.",
        )

    response: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "running": coop_pipeline_manager.running,
        "stage": coop_pipeline_manager.stage,
    }
    if coop_pipeline_manager.error is not None:
        response["error"] = coop_pipeline_manager.error
    if coop_pipeline_manager.report_id is not None:
        response["report_id"] = coop_pipeline_manager.report_id
    if coop_pipeline_manager.summary_path is not None:
        response["summary_path"] = coop_pipeline_manager.summary_path

    return response


@router.get("/runs")
async def list_coop_pipeline_runs() -> list[dict]:
    """List all cooperative pipeline run summaries, newest first."""
    if not _PIPELINES_DIR.exists():
        return []

    runs: list[dict] = []
    for child in _PIPELINES_DIR.iterdir():
        if not child.is_dir() or not child.name.startswith("cooperative_pipeline_"):
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs.append({
                "pipeline_id": summary.get("pipeline_id", child.name),
                "timestamp": summary.get("timestamp", ""),
                "config_id": summary.get("config_id", ""),
                "config_hash": summary.get("config_hash", ""),
                "report_id": summary.get("report_id"),
                "archetype": "cooperative",
            })
        except (json.JSONDecodeError, OSError):
            continue

    runs.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return runs
