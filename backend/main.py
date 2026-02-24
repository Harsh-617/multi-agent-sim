"""FastAPI application assembly."""

from __future__ import annotations

from fastapi import FastAPI

from backend.api.routes_config import router as config_router
from backend.api.routes_experiment import router as experiment_router
from backend.api.routes_history import router as history_router
from backend.api.routes_league import router as league_router
from backend.api.routes_pipeline import router as pipeline_router
from backend.api.routes_reports import router as reports_router
from backend.api.ws_metrics import router as ws_router

app = FastAPI(title="Multi-Agent Simulation", version="0.1.0")

app.include_router(config_router)
app.include_router(experiment_router)
app.include_router(history_router)
app.include_router(league_router)
app.include_router(pipeline_router)
app.include_router(reports_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}
