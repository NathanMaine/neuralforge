"""Server-Sent Events (SSE) stream with heartbeat for real-time updates."""
import asyncio
import json
import logging
import time
from collections import deque
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["events"])
logger = logging.getLogger(__name__)

# In-memory event queue (bounded deque for back-pressure)
_event_queue: deque[dict] = deque(maxlen=1000)
_subscribers: list[asyncio.Queue] = []

# Heartbeat interval in seconds
_HEARTBEAT_INTERVAL = 15


def publish_event(event_type: str, data: dict) -> None:
    """Publish an event to all SSE subscribers.

    Parameters
    ----------
    event_type:
        Event type string (e.g. ``ingest_complete``, ``graph_updated``).
    data:
        Event payload.
    """
    event = {
        "type": event_type,
        "data": data,
        "timestamp": time.time(),
    }
    _event_queue.append(event)
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


async def _event_stream(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted lines from the event queue."""
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
                line = f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                yield line
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f": heartbeat {time.time()}\n\n"
    except asyncio.CancelledError:
        pass


@router.get("/events")
async def event_stream():
    """SSE endpoint -- streams real-time NeuralForge events with heartbeat."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(queue)

    async def cleanup_generator():
        try:
            async for chunk in _event_stream(queue):
                yield chunk
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        cleanup_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/recent")
async def recent_events(limit: int = 50):
    """Return the most recent events from the in-memory buffer."""
    events = list(_event_queue)[-limit:]
    return {"events": events, "total_buffered": len(_event_queue)}
