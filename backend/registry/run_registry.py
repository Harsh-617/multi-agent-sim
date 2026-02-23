"""Lightweight run registry — scans storage/runs/ for completed run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunRecord:
    """Summary of a single run, built from stored artifacts."""

    __slots__ = (
        "run_id", "config_id", "agent_policy", "termination_reason",
        "episode_length", "timestamp",
    )

    def __init__(
        self,
        run_id: str,
        config_id: str | None,
        agent_policy: str | None,
        termination_reason: str | None,
        episode_length: int | None,
        timestamp: str | None,
    ) -> None:
        self.run_id = run_id
        self.config_id = config_id
        self.agent_policy = agent_policy
        self.termination_reason = termination_reason
        self.episode_length = episode_length
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "agent_policy": self.agent_policy,
            "termination_reason": self.termination_reason,
            "episode_length": self.episode_length,
            "timestamp": self.timestamp,
        }


class RunRegistry:
    """Scan-based registry over storage/runs/."""

    def __init__(self, runs_dir: str | Path) -> None:
        self._runs_dir = Path(runs_dir)

    def list_runs(self) -> list[dict[str, Any]]:
        """Return summary dicts for every run directory that has a config.json."""
        if not self._runs_dir.exists():
            return []

        records: list[RunRecord] = []
        for run_dir in sorted(self._runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            record = self._build_record(run_dir)
            records.append(record)

        # Most recent first
        records.sort(key=lambda r: r.timestamp or "", reverse=True)
        return [r.to_dict() for r in records]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return full metadata + episode_summary for a single run."""
        run_dir = self._runs_dir / run_id
        if not run_dir.is_dir() or not (run_dir / "config.json").exists():
            return None

        record = self._build_record(run_dir).to_dict()

        # Attach episode_summary if available
        summary_path = run_dir / "episode_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.pop("written_at", None)
            record["episode_summary"] = summary
        else:
            record["episode_summary"] = None

        return record

    def _build_record(self, run_dir: Path) -> RunRecord:
        run_id = run_dir.name

        config_data = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        timestamp = config_data.get("written_at")
        config_id = config_data.get("identity", {}).get("config_id")
        agent_policy = config_data.get("_agent_policy")

        # Try episode_summary for termination_reason / episode_length
        termination_reason: str | None = None
        episode_length: int | None = None
        summary_path = run_dir / "episode_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            termination_reason = summary.get("termination_reason")
            episode_length = summary.get("episode_length")

        return RunRecord(
            run_id=run_id,
            config_id=config_id,
            agent_policy=agent_policy,
            termination_reason=termination_reason,
            episode_length=episode_length,
            timestamp=timestamp,
        )
