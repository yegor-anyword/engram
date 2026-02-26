"""Per-context advisory lock manager for safe multi-agent writes.

Agents can compute in parallel — the lock only serializes delta application.
Uses asyncio.Lock per context ID. SQLite advisory locks would require
a separate connection per lock; asyncio locks are simpler and correct
for the single-process case.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from engram.core.exceptions import ConcurrencyError

logger = logging.getLogger(__name__)


class ContextLockManager:
    """Manages per-context advisory locks for serializing delta application."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, context_id: str) -> asyncio.Lock:
        """Get or create a lock for a context."""
        if context_id not in self._locks:
            self._locks[context_id] = asyncio.Lock()
        return self._locks[context_id]

    @asynccontextmanager
    async def acquire(
        self, context_id: str, timeout: float = 30.0
    ) -> AsyncIterator[None]:
        """Acquire the lock for a context with timeout.

        Uses an `acquired` boolean to ensure we only release a lock
        we actually acquired (not someone else's).

        Raises:
            ConcurrencyError: If the lock cannot be acquired within timeout.
        """
        lock = self._get_lock(context_id)
        acquired = False
        try:
            acquired = await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConcurrencyError(context_id, timeout)

        try:
            yield
        finally:
            if acquired:
                lock.release()

    def cleanup(self, context_id: str) -> None:
        """Remove lock entry for a deleted context."""
        self._locks.pop(context_id, None)
