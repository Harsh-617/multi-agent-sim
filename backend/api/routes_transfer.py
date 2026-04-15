"""Transfer experiment API routes.

POST /api/transfer/run          — validate request, start experiment as background task
GET  /api/transfer/status/{id}  — poll status / results
GET  /api/transfer/reports      — list all saved transfer reports
GET  /api/transfer/reports/{id} — fetch a specific transfer report
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.storage_root import STORAGE_ROOT

router = APIRouter(prefix="/api/transfer", tags=["transfer"])

REPORTS_ROOT = STORAGE_ROOT / "reports"
CONFIGS_DIR = STORAGE_ROOT / "configs"

# ---------------------------------------------------------------------------
# Status stages: pending → running_transfer → running_baseline → saving → done → error
# ---------------------------------------------------------------------------

_VALID_ARCHETYPES = {"mixed", "competitive", "cooperative"}


# ---------------------------------------------------------------------------
# In-process state manager (one transfer at a time per process)
# ---------------------------------------------------------------------------

class TransferManager:
    """Holds state for the most-recently-submitted transfer experiment."""

    def __init__(self) -> None:
        self.transfer_id: str | None = None
        self.running: bool = False
        self.stage: str = "idle"
        self.error: str | None = None
        self.result: dict | None = None
        self._task: asyncio.Task | None = None

    def reset_state(self) -> None:
        self.transfer_id = None
        self.running = False
        self.stage = "idle"
        self.error = None
        self.result = None
        self._task = None

    def attach_task(self, task: asyncio.Task) -> None:
        self._task = task


transfer_manager = TransferManager()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TransferRunRequest(BaseModel):
    source_archetype: str = Field(
        description='"mixed" | "competitive" | "cooperative"',
    )
    source_member_id: str = Field(
        description="League registry member ID of the source champion.",
    )
    target_archetype: str = Field(
        description='"mixed" | "competitive" | "cooperative" — must differ from source.',
    )
    target_config_id: str = Field(
        description='Saved config ID or "default".',
    )
    episodes: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of episodes per condition (1–20).",
    )
    seed: int = Field(
        default=42,
        description="Base random seed.",
    )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_transfer_task(
    tm: TransferManager,
    transfer_id: str,
    req: TransferRunRequest,
) -> None:
    """Execute the transfer experiment in a thread-pool executor."""
    loop = asyncio.get_running_loop()
    tm.running = True
    tm.transfer_id = transfer_id
    tm.stage = "running_transfer"

    try:
        from simulation.transfer.transfer_runner import run_transfer_experiment

        def _run() -> dict:
            return run_transfer_experiment(
                source_archetype=req.source_archetype,
                source_member_id=req.source_member_id,
                target_archetype=req.target_archetype,
                target_config_id=req.target_config_id,
                episodes=req.episodes,
                seed=req.seed,
            )

        tm.stage = "running_transfer"
        # The runner internally progresses through transfer → baseline → saving.
        # We use a progress hook to surface intermediate stages.
        result = await loop.run_in_executor(None, _run)

        tm.stage = "saving"
        tm.result = result
        tm.stage = "done"

    except Exception as exc:  # noqa: BLE001
        tm.error = str(exc)
        tm.stage = "error"
    finally:
        tm.running = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run")
async def start_transfer(req: TransferRunRequest) -> dict:
    """Validate request and launch transfer experiment as a background task.

    Returns ``{ transfer_id, status: "pending" }`` immediately.
    Raises HTTP 422 if source == target archetype or config type mismatches.
    """
    # Validate archetypes
    if req.source_archetype not in _VALID_ARCHETYPES:
        raise HTTPException(
            status_code=422,
            detail=f"source_archetype must be one of {sorted(_VALID_ARCHETYPES)}",
        )
    if req.target_archetype not in _VALID_ARCHETYPES:
        raise HTTPException(
            status_code=422,
            detail=f"target_archetype must be one of {sorted(_VALID_ARCHETYPES)}",
        )
    if req.source_archetype == req.target_archetype:
        raise HTTPException(
            status_code=422,
            detail=(
                f"source_archetype and target_archetype must differ "
                f"(both are {req.source_archetype!r})."
            ),
        )

    # Validate target config archetype match (skip for "default")
    if req.target_config_id != "default":
        config_path = CONFIGS_DIR / f"{req.target_config_id}.json"
        if not config_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Config {req.target_config_id!r} not found.",
            )
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        env_type = raw.get("identity", {}).get("environment_type", "")
        if env_type != req.target_archetype:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Config {req.target_config_id!r} has environment_type={env_type!r} "
                    f"but target_archetype is {req.target_archetype!r}."
                ),
            )

    transfer_id = uuid.uuid4().hex[:12]
    transfer_manager.reset_state()
    transfer_manager.transfer_id = transfer_id
    transfer_manager.stage = "pending"

    task = asyncio.create_task(
        _run_transfer_task(transfer_manager, transfer_id, req)
    )
    transfer_manager.attach_task(task)

    await asyncio.sleep(0)

    return {"transfer_id": transfer_id, "status": "pending"}


@router.get("/status/{transfer_id}")
async def transfer_status(transfer_id: str) -> dict[str, Any]:
    """Poll the status of a transfer experiment.

    Returns ``{ transfer_id, status, result? }`` where *result* is populated
    when status == "done".
    """
    if transfer_manager.transfer_id != transfer_id:
        raise HTTPException(
            status_code=404,
            detail=f"Transfer run {transfer_id!r} not found.",
        )

    response: dict[str, Any] = {
        "transfer_id": transfer_id,
        "status": transfer_manager.stage,
        "running": transfer_manager.running,
    }
    if transfer_manager.error is not None:
        response["error"] = transfer_manager.error
    if transfer_manager.result is not None:
        response["result"] = transfer_manager.result

    return response


@router.get("/reports")
async def list_transfer_reports() -> list[dict[str, Any]]:
    """List all transfer reports under storage/reports/.

    Returns a list of lightweight summaries (without per-episode data).
    """
    if not REPORTS_ROOT.exists():
        return []

    reports: list[dict[str, Any]] = []
    for d in sorted(REPORTS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("transfer_"):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            # Return lightweight listing (omit large per-episode arrays)
            reports.append({
                "report_id": data.get("report_id", d.name),
                "report_type": data.get("report_type", "transfer"),
                "source_archetype": data.get("source_archetype"),
                "source_member_id": data.get("source_member_id"),
                "target_archetype": data.get("target_archetype"),
                "target_config_hash": data.get("target_config_hash"),
                "episodes": data.get("episodes"),
                "transferred_mean": data.get("transferred_mean"),
                "baseline_mean": data.get("baseline_mean"),
                "vs_baseline_delta": data.get("vs_baseline_delta"),
                "vs_baseline_pct": data.get("vs_baseline_pct"),
            })
        except Exception:  # noqa: BLE001
            continue

    return reports


@router.get("/reports/{report_id}")
async def get_transfer_report(report_id: str) -> dict[str, Any]:
    """Return the full summary.json for a specific transfer report.

    HTTP 404 if the report directory or summary.json does not exist.
    """
    report_dir = REPORTS_ROOT / report_id
    if not report_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Transfer report {report_id!r} not found.",
        )
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"summary.json not found in report {report_id!r}.",
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))
