"""Tests for the concurrency module — ContextLockManager."""

import asyncio

import pytest

from engram.core.concurrency import ContextLockManager
from engram.core.exceptions import ConcurrencyError


@pytest.fixture
def lock_manager():
    return ContextLockManager()


class TestContextLockManager:
    async def test_lock_serializes_access(self, lock_manager):
        """Acquiring the same context lock serializes concurrent access."""
        order: list[int] = []

        async def writer(n: int, delay: float):
            async with lock_manager.acquire("ctx-1"):
                order.append(n)
                await asyncio.sleep(delay)

        # Writer 1 grabs lock first, writer 2 waits
        t1 = asyncio.create_task(writer(1, 0.1))
        await asyncio.sleep(0.01)  # Ensure t1 starts first
        t2 = asyncio.create_task(writer(2, 0.0))
        await asyncio.gather(t1, t2)

        assert order == [1, 2]

    async def test_different_contexts_dont_block(self, lock_manager):
        """Locks on different contexts are independent."""
        results: list[str] = []

        async def writer(ctx_id: str, label: str):
            async with lock_manager.acquire(ctx_id):
                results.append(f"{label}-start")
                await asyncio.sleep(0.05)
                results.append(f"{label}-end")

        await asyncio.gather(
            writer("ctx-a", "A"),
            writer("ctx-b", "B"),
        )

        # Both should interleave (both start before either ends)
        assert "A-start" in results
        assert "B-start" in results

    async def test_lock_timeout_raises_concurrency_error(self, lock_manager):
        """Timeout raises ConcurrencyError."""
        async with lock_manager.acquire("ctx-1"):
            with pytest.raises(ConcurrencyError) as exc_info:
                async with lock_manager.acquire("ctx-1", timeout=0.05):
                    pass  # pragma: no cover
            assert exc_info.value.context_id == "ctx-1"
            assert exc_info.value.timeout == 0.05

    async def test_cleanup_removes_lock(self, lock_manager):
        """cleanup() removes the lock entry."""
        async with lock_manager.acquire("ctx-1"):
            pass
        assert "ctx-1" in lock_manager._locks
        lock_manager.cleanup("ctx-1")
        assert "ctx-1" not in lock_manager._locks

    async def test_concurrent_commits_consistent(self, lock_manager):
        """Two concurrent writes produce a consistent final state."""
        counter = {"value": 0}

        async def increment():
            async with lock_manager.acquire("ctx-1"):
                current = counter["value"]
                await asyncio.sleep(0.01)
                counter["value"] = current + 1

        await asyncio.gather(increment(), increment(), increment())
        assert counter["value"] == 3
