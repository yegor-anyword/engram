"""Tests for the event bus module."""

import asyncio

import pytest

from engram.core.events import EventBus


@pytest.fixture
def event_bus():
    return EventBus(max_queue_size=5)


class TestEventBus:
    async def test_subscribe_receives_events(self, event_bus):
        """Subscriber receives emitted events."""
        queue = event_bus.subscribe("ctx-1")
        event_bus.emit("ctx-1", event_type="commit", agent_id="agent-1", data={"key": "value"})

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event.event_type == "commit"
        assert event.context_id == "ctx-1"
        assert event.agent_id == "agent-1"
        assert event.data["key"] == "value"

    async def test_unsubscribe_stops_events(self, event_bus):
        """Unsubscribed queue no longer receives events."""
        queue = event_bus.subscribe("ctx-1")
        event_bus.unsubscribe("ctx-1", queue)
        event_bus.emit("ctx-1", event_type="commit")

        assert queue.empty()

    async def test_multiple_subscribers_receive_same_event(self, event_bus):
        """Multiple subscribers both receive the same event."""
        q1 = event_bus.subscribe("ctx-1")
        q2 = event_bus.subscribe("ctx-1")
        event_bus.emit("ctx-1", event_type="commit")

        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert e1.event_type == "commit"
        assert e2.event_type == "commit"

    async def test_queue_full_drops_oldest(self, event_bus):
        """When queue is full, oldest event is dropped."""
        queue = event_bus.subscribe("ctx-1")

        # Fill the queue (max_queue_size=5)
        for i in range(5):
            event_bus.emit("ctx-1", event_type=f"event-{i}")

        # Queue is now full. Emit one more — should drop event-0
        event_bus.emit("ctx-1", event_type="event-5")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        # Should have events 1-5 (event-0 was dropped)
        assert len(events) == 5
        assert events[0].event_type == "event-1"
        assert events[-1].event_type == "event-5"

    async def test_events_for_different_contexts_isolated(self, event_bus):
        """Events for one context don't leak to another."""
        q1 = event_bus.subscribe("ctx-1")
        q2 = event_bus.subscribe("ctx-2")

        event_bus.emit("ctx-1", event_type="commit")

        assert not q1.empty()
        assert q2.empty()

    async def test_cleanup_removes_subscribers(self, event_bus):
        """cleanup() removes all subscribers for a context."""
        event_bus.subscribe("ctx-1")
        event_bus.subscribe("ctx-1")
        assert "ctx-1" in event_bus._subscribers

        event_bus.cleanup("ctx-1")
        assert "ctx-1" not in event_bus._subscribers
