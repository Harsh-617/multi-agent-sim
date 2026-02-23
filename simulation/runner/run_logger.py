"""Run artifact logger — writes structured files into storage/runs/{run_id}/.

Produces:
  - config.json          Full environment config snapshot
  - metrics.jsonl        Per-step metric records (append)
  - events.jsonl         Semantic event records (append)
  - episode_summary.json Episode-level summary (written once at end)

Uses only stdlib (json, pathlib, datetime). No database dependency.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes simulation artifacts to a run directory."""

    def __init__(self, base_dir: str | Path, run_id: str) -> None:
        self._run_dir = Path(base_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._run_dir / "metrics.jsonl"
        self._events_path = self._run_dir / "events.jsonl"

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def write_config(self, config_dict: dict[str, Any]) -> None:
        """Write the full environment config as config.json."""
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            **config_dict,
        }
        (self._run_dir / "config.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Step metrics (append)
    # ------------------------------------------------------------------

    def log_step_metrics(self, records: list[dict[str, Any]]) -> None:
        """Append step metric records to metrics.jsonl."""
        if not records:
            return
        with self._metrics_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------
    # Events (append)
    # ------------------------------------------------------------------

    def log_events(self, events: list[dict[str, Any]]) -> None:
        """Append semantic events to events.jsonl."""
        if not events:
            return
        with self._events_path.open("a", encoding="utf-8") as f:
            for evt in events:
                f.write(json.dumps(evt, default=str) + "\n")

    # ------------------------------------------------------------------
    # Episode summary (write once)
    # ------------------------------------------------------------------

    def write_episode_summary(self, summary: dict[str, Any]) -> None:
        """Write the episode summary as episode_summary.json."""
        if not summary:
            return
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        (self._run_dir / "episode_summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
