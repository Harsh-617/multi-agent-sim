"""Run control endpoints — start, stop, status."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.config.schema import MixedEnvironmentConfig
from simulation.league.registry import LeagueRegistry
from simulation.runner.competitive_experiment_runner import run_competitive_experiment

from backend.runner.experiment_runner import run_experiment
from backend.runner.run_manager import manager
from backend.schemas.api_models import RunStatus, StartRunRequest, StartRunResponse
from backend.storage_root import STORAGE_ROOT

router = APIRouter(prefix="/api/runs", tags=["runs"])

CONFIGS_DIR = STORAGE_ROOT / "configs"
RUNS_DIR = STORAGE_ROOT / "runs"
PPO_AGENT_DIR = STORAGE_ROOT / "agents/ppo_shared"
LEAGUE_ROOT = STORAGE_ROOT / "agents/league"


@router.post("/start", response_model=StartRunResponse)
async def start_run(req: StartRunRequest) -> StartRunResponse:
    """Start a new run from a saved config. One run at a time."""
    if manager.running:
        raise HTTPException(status_code=409, detail="A run is already in progress.")

    config_path = CONFIGS_DIR / f"{req.config_id}.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config {req.config_id} not found.")

    # Validate PPO artifacts exist before starting the run
    if req.agent_policy == "ppo_shared":
        if not (PPO_AGENT_DIR / "policy.pt").exists() or not (PPO_AGENT_DIR / "metadata.json").exists():
            raise HTTPException(
                status_code=422,
                detail="PPO artifacts not found in storage/agents/ppo_shared/. "
                       "Train a policy first with: python -m simulation.training.ppo_shared",
            )

    # Validate league snapshot
    league_member_dir: str | None = None
    if req.agent_policy == "league_snapshot":
        if not req.league_member_id:
            raise HTTPException(
                status_code=422,
                detail="league_member_id is required when agent_policy is 'league_snapshot'.",
            )
        registry = LeagueRegistry(LEAGUE_ROOT)
        try:
            member_path = registry.load_member(req.league_member_id)
            league_member_dir = str(member_path)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"League member not found: {req.league_member_id}",
            )

    raw_text = config_path.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text)
    env_type = raw_data.get("identity", {}).get("environment_type", "mixed")

    run_id = uuid.uuid4().hex[:12]
    manager.reset_state()

    agent_kwargs = {}
    if league_member_dir:
        agent_kwargs["member_dir"] = league_member_dir

    if env_type == "competitive":
        config = CompetitiveEnvironmentConfig.model_validate(raw_data)

        async def _run_competitive() -> dict:
            return run_competitive_experiment(
                config, run_id, RUNS_DIR, manager,
                agent_policy=req.agent_policy,
                agent_kwargs=agent_kwargs,
            )

        task = asyncio.create_task(_run_competitive())
    else:
        config = MixedEnvironmentConfig.model_validate_json(raw_text)
        task = asyncio.create_task(
            run_experiment(
                config, run_id, RUNS_DIR, manager,
                agent_policy=req.agent_policy,
                agent_kwargs=agent_kwargs,
            )
        )

    manager.attach_task(task)

    # Give the task a moment to initialise so status is immediately consistent
    await asyncio.sleep(0)

    return StartRunResponse(run_id=run_id)


@router.post("/stop")
async def stop_run() -> dict:
    """Gracefully stop the current run."""
    if not manager.running:
        raise HTTPException(status_code=409, detail="No run is currently active.")
    manager.request_stop()
    return {"detail": "Stop requested.", "run_id": manager.run_id}


@router.get("/status", response_model=RunStatus)
async def run_status() -> RunStatus:
    return RunStatus(
        running=manager.running,
        run_id=manager.run_id,
        step=manager.step if manager.running else None,
        max_steps=manager.max_steps if manager.running else None,
        termination_reason=manager.termination_reason,
    )
