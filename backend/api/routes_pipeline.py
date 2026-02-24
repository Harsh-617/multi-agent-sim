"""Pipeline automation endpoints.

POST /api/pipeline/run
    Start a full pipeline (training → ratings → robustness → summary).
    Returns { pipeline_id }.

GET /api/pipeline/{pipeline_id}/status
    Poll status of a previously started pipeline.
    Returns { running, pipeline_id, stage, error?, report_id?, summary_path? }.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline.pipeline_manager import pipeline_manager
from simulation.pipeline.pipeline_run import run_pipeline

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    config_id: str = Field(default="default", description='Config id or "default".')
    seed: int = Field(default=42, description="Master seed for reproducibility.")
    seeds: int = Field(
        default=3, ge=1, description="Number of evaluation seeds."
    )
    episodes_per_seed: int = Field(
        default=2, ge=1, description="Evaluation episodes per seed per policy per sweep."
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


async def _run_pipeline_task(
    pm: Any,
    pipeline_id: str,
    kwargs: dict[str, Any],
) -> None:
    """Run pipeline in thread-pool executor to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    pm.running = True
    pm.pipeline_id = pipeline_id
    pm.stage = "loading_config"

    def _progress(stage: str, detail: str = "") -> None:
        # Guard: "done" is set by the task after report_id is populated so
        # that consumers never observe stage=done with report_id still None.
        if stage != "done":
            pm.set_stage(stage, detail)

    def _run() -> Path:
        return run_pipeline(progress_callback=_progress, **kwargs)

    try:
        result_path = await loop.run_in_executor(None, _run)
        summary_file = result_path / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        # Populate result fields BEFORE transitioning to "done" so that
        # a concurrent status poll never sees done without report_id.
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
async def start_pipeline(req: PipelineRunRequest) -> dict:
    """Start the automation pipeline in the background.

    Returns ``{ pipeline_id }`` immediately; poll ``/status`` for progress.
    """
    if pipeline_manager.running:
        raise HTTPException(
            status_code=409, detail="A pipeline is already running."
        )

    pipeline_id = uuid.uuid4().hex[:12]
    pipeline_manager.reset_state()
    pipeline_manager.pipeline_id = pipeline_id

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
        _run_pipeline_task(pipeline_manager, pipeline_id, kwargs)
    )
    pipeline_manager.attach_task(task)

    # Yield control so the task can begin before the response is sent
    await asyncio.sleep(0)

    return {"pipeline_id": pipeline_id}


@router.get("/{pipeline_id}/status")
async def pipeline_status(pipeline_id: str) -> dict:
    """Return the current status of a pipeline run.

    HTTP 404 is returned if *pipeline_id* does not match the current (or
    most recently completed) pipeline.
    """
    if pipeline_manager.pipeline_id != pipeline_id:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_id}' not found.",
        )

    response: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "running": pipeline_manager.running,
        "stage": pipeline_manager.stage,
    }
    if pipeline_manager.error is not None:
        response["error"] = pipeline_manager.error
    if pipeline_manager.report_id is not None:
        response["report_id"] = pipeline_manager.report_id
    if pipeline_manager.summary_path is not None:
        response["summary_path"] = pipeline_manager.summary_path

    return response
