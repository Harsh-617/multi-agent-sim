"""Run history, replay, and benchmark endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from simulation.config.schema import MixedEnvironmentConfig

from backend.registry.run_registry import RunRegistry
from backend.runner.experiment_runner import run_experiment
from backend.runner.run_manager import RunManager
from backend.schemas.api_models import BenchmarkRequest

router = APIRouter(prefix="/api", tags=["history"])

RUNS_DIR = Path("storage/runs")
CONFIGS_DIR = Path("storage/configs")


def _registry() -> RunRegistry:
    return RunRegistry(RUNS_DIR)


# ------------------------------------------------------------------
# Run History
# ------------------------------------------------------------------

@router.get("/runs/history")
async def list_runs() -> list[dict]:
    """List all completed runs with summary metadata."""
    return _registry().list_runs()


@router.get("/runs/{run_id}/detail")
async def get_run(run_id: str) -> dict:
    """Get full metadata + episode_summary for a single run."""
    result = _registry().get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return result


# ------------------------------------------------------------------
# Replay (SSE)
# ------------------------------------------------------------------

@router.get("/runs/{run_id}/replay")
async def replay_run(run_id: str) -> StreamingResponse:
    """Stream stored metrics.jsonl step-by-step as Server-Sent Events.

    Each SSE event contains one step's worth of metric records.
    Final event is type=done with the episode_summary.
    """
    run_dir = RUNS_DIR / run_id
    metrics_path = run_dir / "metrics.jsonl"

    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics for run {run_id}.")

    async def _generate():
        # Read all metric records
        lines = metrics_path.read_text(encoding="utf-8").strip().split("\n")
        records = [json.loads(line) for line in lines if line.strip()]

        # Group by step
        from collections import defaultdict
        by_step: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            by_step[rec["step"]].append(rec)

        # Stream step-by-step
        for step in sorted(by_step.keys()):
            payload = json.dumps({
                "type": "step",
                "run_id": run_id,
                "t": step,
                "metrics": by_step[step],
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.05)  # pace replay

        # Send episode summary if available
        summary_path = run_dir / "episode_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.pop("written_at", None)
            payload = json.dumps({
                "type": "done",
                "run_id": run_id,
                "termination_reason": summary.get("termination_reason"),
                "episode_summary": summary,
            })
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------

@router.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest) -> dict:
    """Run the same config with multiple agent policies and compare results.

    Runs experiments sequentially (reuses experiment_runner) and returns
    a comparison table.
    """
    config_path = CONFIGS_DIR / f"{req.config_id}.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config {req.config_id} not found.")

    config = MixedEnvironmentConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )

    results = []
    for policy in req.agent_policies:
        mgr = RunManager()
        run_id = f"bench_{req.config_id[:8]}_{policy}"

        await run_experiment(config, run_id, RUNS_DIR, mgr, agent_policy=policy)

        # Read the produced summary
        summary_path = RUNS_DIR / run_id / "episode_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            total_rewards = summary.get("total_reward_per_agent", {})
            mean_reward = (
                sum(total_rewards.values()) / len(total_rewards)
                if total_rewards else 0.0
            )
            results.append({
                "agent_policy": policy,
                "mean_reward": round(mean_reward, 4),
                "final_shared_pool": summary.get("final_shared_pool"),
                "termination_reason": summary.get("termination_reason"),
                "episode_length": summary.get("episode_length"),
            })
        else:
            results.append({
                "agent_policy": policy,
                "mean_reward": None,
                "final_shared_pool": None,
                "termination_reason": None,
                "episode_length": None,
            })

    return {"config_id": req.config_id, "results": results}
