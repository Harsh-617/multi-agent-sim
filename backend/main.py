"""FastAPI application assembly."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes_config import router as config_router
from backend.api.routes_cooperative import router as cooperative_router
from backend.api.routes_cooperative_league import router as cooperative_league_router
from backend.api.routes_cooperative_pipeline import router as cooperative_pipeline_router
from backend.api.routes_cooperative_reports import router as cooperative_reports_router
from backend.api.routes_experiment import router as experiment_router
from backend.api.routes_history import router as history_router
from backend.api.routes_competitive_league import router as competitive_league_router
from backend.api.routes_competitive_reports import router as competitive_reports_router
from backend.api.routes_league import router as league_router
from backend.api.routes_pipeline import competitive_router as competitive_pipeline_router
from backend.api.routes_pipeline import router as pipeline_router
from backend.api.routes_reports import router as reports_router
from backend.api.routes_export import router as export_router
from backend.api.routes_transfer import router as transfer_router
from backend.api.ws_metrics import router as ws_router
from backend.storage_root import STORAGE_ROOT

_STORAGE_DIRS = [
    STORAGE_ROOT,
    STORAGE_ROOT / "configs",
    STORAGE_ROOT / "runs",
    STORAGE_ROOT / "reports",
    STORAGE_ROOT / "pipelines",
    STORAGE_ROOT / "agents",
    STORAGE_ROOT / "agents" / "league",
    STORAGE_ROOT / "agents" / "competitive_league",
    STORAGE_ROOT / "agents" / "cooperative",
    STORAGE_ROOT / "agents" / "cooperative" / "league",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in _STORAGE_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Multi-Agent Simulation", version="0.1.0", lifespan=lifespan)

app.include_router(competitive_league_router)
app.include_router(cooperative_router)
app.include_router(cooperative_league_router)
app.include_router(cooperative_pipeline_router)
app.include_router(cooperative_reports_router)
app.include_router(competitive_reports_router)
app.include_router(config_router)
app.include_router(experiment_router)
app.include_router(history_router)
app.include_router(league_router)
app.include_router(competitive_pipeline_router)
app.include_router(pipeline_router)
app.include_router(reports_router)
app.include_router(export_router)
app.include_router(transfer_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}
