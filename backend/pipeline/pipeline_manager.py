"""In-memory singleton tracking the current pipeline's lifecycle.

Only one pipeline may run at a time (same restriction as RunManager).
Stage values progress through:
  idle → loading_config → training → ratings → robustness → saving → done
or: any_stage → error  (on failure)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class PipelineManager:
    """Process-wide pipeline state."""

    def __init__(self) -> None:
        self.pipeline_id: str | None = None
        self.running: bool = False
        self.stage: str = "idle"
        self.error: str | None = None
        self.report_id: str | None = None
        self.summary_path: str | None = None
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    def attach_task(self, task: asyncio.Task[None]) -> None:
        self._task = task

    def reset_state(self) -> None:
        """Prepare for a fresh pipeline run."""
        self.pipeline_id = None
        self.running = False
        self.stage = "idle"
        self.error = None
        self.report_id = None
        self.summary_path = None
        self._task = None

    # ------------------------------------------------------------------
    # Stage update (called from worker thread via progress_callback)
    # ------------------------------------------------------------------

    def set_stage(self, stage: str, detail: str = "") -> None:
        """Update stage; safe to call from a non-async thread."""
        self.stage = stage


# Module-level singleton
pipeline_manager = PipelineManager()
