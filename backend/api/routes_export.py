"""Policy export endpoints.

Routes
------
GET /api/export/{archetype}/champion
    Download the current champion's policy as a zip file.

GET /api/export/{archetype}/members/{member_id}
    Download a specific league member's policy as a zip file.

archetype must be one of: mixed | competitive | cooperative
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.storage_root import STORAGE_ROOT
from simulation.export.policy_exporter import build_export_zip
from simulation.league.cooperative_registry import CooperativeLeagueRegistry
from simulation.league.registry import LeagueRegistry

router = APIRouter(prefix="/api/export", tags=["export"])

# ---------------------------------------------------------------------------
# Storage roots (module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------

_MIXED_LEAGUE_ROOT = STORAGE_ROOT / "agents/league"
_COMPETITIVE_LEAGUE_ROOT = STORAGE_ROOT / "agents/competitive_league"
_COOPERATIVE_LEAGUE_ROOT = STORAGE_ROOT / "agents/cooperative/league"

_DEFAULT_RATING = 1000.0


# ---------------------------------------------------------------------------
# Archetype enum
# ---------------------------------------------------------------------------

class Archetype(str, Enum):
    mixed = "mixed"
    competitive = "competitive"
    cooperative = "cooperative"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_league_root(archetype: Archetype) -> Path:
    if archetype == Archetype.mixed:
        return _MIXED_LEAGUE_ROOT
    if archetype == Archetype.competitive:
        return _COMPETITIVE_LEAGUE_ROOT
    return _COOPERATIVE_LEAGUE_ROOT


def _get_registry(archetype: Archetype, league_root: Path):
    """Return an appropriate registry instance for the archetype."""
    if archetype == Archetype.cooperative:
        return CooperativeLeagueRegistry(league_root)
    return LeagueRegistry(league_root)


def _load_ratings(league_root: Path) -> dict[str, float]:
    """Load ratings.json → {member_id: rating} or empty dict."""
    ratings_path = league_root / "ratings.json"
    if not ratings_path.exists():
        return {}
    try:
        data = json.loads(ratings_path.read_text(encoding="utf-8"))
        return {r["member_id"]: r["rating"] for r in data}
    except Exception:
        return {}


def _find_champion(members: list[dict], ratings: dict[str, float]) -> dict | None:
    """Return member with highest rating (tie-break: newest created_at)."""
    if not members:
        return None
    best: dict | None = None
    best_rating = -1.0
    best_created = ""
    for m in members:
        r = ratings.get(m["member_id"], _DEFAULT_RATING)
        created = m.get("created_at") or ""
        if r > best_rating or (r == best_rating and created > best_created):
            best = m
            best_rating = r
            best_created = created
    return best


def _streaming_zip(zip_bytes: bytes, filename: str) -> StreamingResponse:
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Champion endpoint
# ---------------------------------------------------------------------------

@router.get("/{archetype}/champion")
async def export_champion(archetype: Archetype) -> StreamingResponse:
    """Download the current champion's policy zip.

    Returns 404 if no champion exists or policy.pt is missing.
    Returns 422 if archetype is not one of mixed | competitive | cooperative.
    """
    league_root = _get_league_root(archetype)
    registry = _get_registry(archetype, league_root)

    members = registry.list_members()
    if not members:
        raise HTTPException(status_code=404, detail="No league members found.")

    ratings = _load_ratings(league_root)
    champion = _find_champion(members, ratings)
    if champion is None:
        raise HTTPException(status_code=404, detail="No champion found.")

    member_id = champion["member_id"]
    try:
        member_dir = registry.load_member(member_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Champion member {member_id!r} not found.")

    # Load metadata — metadata.json must exist (registry.list_members() guarantees this)
    meta_path = member_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Enrich metadata with rating for embedding in policy.py
    rating_value = ratings.get(member_id, _DEFAULT_RATING)
    metadata.setdefault("rating", rating_value)

    try:
        zip_bytes = build_export_zip(member_dir, metadata, archetype.value)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filename = f"policy_{archetype.value}_{member_id}.zip"
    return _streaming_zip(zip_bytes, filename)


# ---------------------------------------------------------------------------
# Member endpoint
# ---------------------------------------------------------------------------

@router.get("/{archetype}/members/{member_id}")
async def export_member(archetype: Archetype, member_id: str) -> StreamingResponse:
    """Download a specific member's policy zip.

    Returns 404 if member does not exist or policy.pt is missing.
    Returns 422 if archetype is not one of mixed | competitive | cooperative.
    """
    league_root = _get_league_root(archetype)
    registry = _get_registry(archetype, league_root)

    try:
        member_dir = registry.load_member(member_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"League member {member_id!r} not found in {archetype.value} league.",
        )

    meta_path = member_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"metadata.json missing for member {member_id!r}.",
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Enrich with rating if available
    ratings = _load_ratings(league_root)
    metadata.setdefault("rating", ratings.get(member_id, _DEFAULT_RATING))

    try:
        zip_bytes = build_export_zip(member_dir, metadata, archetype.value)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filename = f"policy_{archetype.value}_{member_id}.zip"
    return _streaming_zip(zip_bytes, filename)
