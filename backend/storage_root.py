"""Centralised, CWD-independent storage root.

All backend modules should derive storage paths from STORAGE_ROOT rather
than using bare relative ``Path("storage/...")`` so that the server can be
started from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# Repo root is the parent of the ``backend`` package directory.
STORAGE_ROOT: Path = Path(__file__).resolve().parent.parent / "storage"
