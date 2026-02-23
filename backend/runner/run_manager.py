"""In-memory singleton managing the current run's lifecycle and WS broadcast.

V1 supports one run at a time.  The RunManager holds:
  - current run state (id, running flag, step, stop signal)
  - a set of connected WebSocket queues for fan-out broadcast
"""

from __future__ import annotations

import asyncio
from typing import Any


class RunManager:
    """Process-wide run state and WebSocket broadcast hub."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.running: bool = False
        self.step: int = 0
        self.max_steps: int = 0
        self.termination_reason: str | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # Each connected WS gets its own queue
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Stop signal
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Subscriber management (WS fan-out)
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        for q in self._subscribers:
            await q.put(message)

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    def attach_task(self, task: asyncio.Task[None]) -> None:
        self._task = task

    def reset_state(self) -> None:
        """Prepare for a fresh run."""
        self.run_id = None
        self.running = False
        self.step = 0
        self.max_steps = 0
        self.termination_reason = None
        self._stop_event = asyncio.Event()
        self._task = None


# Module-level singleton
manager = RunManager()
