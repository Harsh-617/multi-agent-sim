"""WebSocket endpoint for live metric streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.runner.run_manager import manager

router = APIRouter()


@router.websocket("/api/ws/metrics/{run_id}")
async def ws_metrics(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    queue = manager.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # If run finished and queue is empty, close cleanly
                if not manager.running and queue.empty():
                    break
                continue

            # Only forward messages for the requested run
            if msg.get("run_id") != run_id:
                continue

            await websocket.send_json(msg)

            if msg.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass
