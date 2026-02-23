"""Config CRUD endpoints — filesystem-backed, no database."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from simulation.config.schema import MixedEnvironmentConfig

from backend.schemas.api_models import ConfigCreatedResponse, ConfigListItem

router = APIRouter(prefix="/api/configs", tags=["configs"])

CONFIGS_DIR = Path("storage/configs")


def _configs_dir() -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIGS_DIR


@router.post("", response_model=ConfigCreatedResponse, status_code=201)
async def create_config(config: MixedEnvironmentConfig) -> ConfigCreatedResponse:
    """Validate and persist a config. Returns its generated ID."""
    config_id = uuid.uuid4().hex[:12]
    path = _configs_dir() / f"{config_id}.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return ConfigCreatedResponse(config_id=config_id)


@router.get("", response_model=list[ConfigListItem])
async def list_configs() -> list[ConfigListItem]:
    """Return summary of all saved configs."""
    items: list[ConfigListItem] = []
    for p in sorted(_configs_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(ConfigListItem(
                config_id=p.stem,
                seed=data["identity"]["seed"],
                num_agents=data["population"]["num_agents"],
                max_steps=data["population"]["max_steps"],
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    return items


@router.get("/{config_id}")
async def get_config(config_id: str) -> dict:
    """Fetch a saved config by ID."""
    path = _configs_dir() / f"{config_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config {config_id} not found.")
    return json.loads(path.read_text(encoding="utf-8"))
