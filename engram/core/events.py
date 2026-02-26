"""In-process event bus for context change notifications.

Enables real-time updates via SSE or WebSocket. Each context can have
multiple subscribers (asyncio.Queue instances). Events are pushed to
all subscribers; if a subscriber's queue is full, the oldest event
is dropped to prevent backpressure.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engram.core.models import ContextEvent, _utcnow

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 100


class EventBus:
    """Pub/sub event bus for context-level change notifications."""

    def __init__(self, max_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[ContextEvent]]] = {}
        self._max_queue_size = max_queue_size

    def subscribe(self, context_id: str) -> asyncio.Queue[ContextEvent]:
        """Subscribe to events for a context. Returns a queue to read from."""
        queue: asyncio.Queue[ContextEvent] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        if context_id not in self._subscribers:
            self._subscribers[context_id] = []
        self._subscribers[context_id].append(queue)
        logger.debug("New subscriber for context %s (total: %d)",
                      context_id, len(self._subscribers[context_id]))
        return queue

    def unsubscribe(self, context_id: str, queue: asyncio.Queue[ContextEvent]) -> None:
        """Stop receiving events for a context."""
        if context_id in self._subscribers:
            try:
                self._subscribers[context_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[context_id]:
                del self._subscribers[context_id]

    def emit(
        self,
        context_id: str,
        event_type: str,
        agent_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to all subscribers of a context.

        If a subscriber's queue is full, the oldest event is dropped.
        """
        event = ContextEvent(
            event_type=event_type,
            context_id=context_id,
            agent_id=agent_id,
            data=data or {},
            timestamp=_utcnow(),
        )
        queues = self._subscribers.get(context_id, [])
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()  # Drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full for context %s, dropping event", context_id)

    def cleanup(self, context_id: str) -> None:
        """Remove all subscribers for a context."""
        self._subscribers.pop(context_id, None)
