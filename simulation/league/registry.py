"""LeagueRegistry – manages saved agent snapshots under a league directory.

Each member is stored as a numbered folder (league_000001, league_000002, …)
containing a copy of ``policy.pt`` and an enriched ``metadata.json``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


_REQUIRED_ARTIFACTS = ("policy.pt", "metadata.json")


class LeagueRegistry:
    """Filesystem-backed registry of league member snapshots."""

    def __init__(self, league_root: str | Path = "storage/agents/league") -> None:
        self.root = Path(league_root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        source_dir: str | Path,
        *,
        parent_id: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Copy artifacts from *source_dir* into a new league member folder.

        Returns the assigned ``member_id`` (e.g. ``"league_000001"``).

        Raises ``FileNotFoundError`` if *source_dir* is missing required files.
        """
        source_dir = Path(source_dir)
        for name in _REQUIRED_ARTIFACTS:
            if not (source_dir / name).exists():
                raise FileNotFoundError(
                    f"Source directory is missing required artifact: {name}"
                )

        member_id = self._next_member_id()
        dest = self.root / member_id
        dest.mkdir(parents=True, exist_ok=False)

        # Copy artifacts
        for name in _REQUIRED_ARTIFACTS:
            shutil.copy2(source_dir / name, dest / name)

        # Enrich metadata
        meta_path = dest / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["member_id"] = member_id
        meta["parent_id"] = parent_id
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        if notes is not None:
            meta["notes"] = notes
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return member_id

    def list_members(self) -> list[dict]:
        """Return metadata dicts for every league member, sorted by id."""
        members: list[dict] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or not child.name.startswith("league_"):
                continue
            meta_path = child / "metadata.json"
            if meta_path.exists():
                members.append(
                    json.loads(meta_path.read_text(encoding="utf-8"))
                )
        return members

    def load_member(self, member_id: str) -> Path:
        """Return the filesystem path to a member's artifact folder.

        Raises ``KeyError`` if the member does not exist.
        """
        member_dir = self.root / member_id
        if not member_dir.is_dir():
            raise KeyError(f"Unknown league member: {member_id!r}")
        return member_dir

    def get_member_metadata(self, member_id: str) -> dict:
        """Load and return the metadata dict for *member_id*.

        Raises ``KeyError`` if the member does not exist.
        """
        member_dir = self.load_member(member_id)
        meta_path = member_dir / "metadata.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_member_id(self) -> str:
        """Determine the next monotonic member id."""
        max_num = 0
        for child in self.root.iterdir():
            if child.is_dir() and child.name.startswith("league_"):
                try:
                    num = int(child.name.split("_", 1)[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue
        return f"league_{max_num + 1:06d}"
