"""Tests for the delta operation engine."""

import uuid

import pytest

from engram.core.delta import DeltaEngine
from engram.core.models import (
    Bullet,
    BulletType,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    IntentAnchor,
    SchemaNode,
)
from engram.storage.sqlite import SQLiteBackend


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test deltas")
    ctx = Context(name="Delta Test", intent=intent)
    await storage.create_context(ctx)
    return str(ctx.id)


class TestDeltaEngine:
    async def test_apply_add_bullet(self, storage, context_id):
        engine = DeltaEngine(storage)
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.ADD_BULLET,
                    target_id="test-b1",
                    section="testing",
                    content="Test fact",
                    bullet_type="fact",
                    source=DeltaSource.CURATOR,
                    confidence=0.8,
                ),
            ],
        )
        result = await engine.apply_batch(batch)
        assert result.bullets_added == 1
        assert result.bullets_updated == 0

        bullet = await storage.get_bullet("test-b1")
        assert bullet is not None
        assert bullet.content == "Test fact"
        assert bullet.section == "testing"

    async def test_apply_update_bullet(self, storage, context_id):
        engine = DeltaEngine(storage)
        # First add a bullet
        bullet = Bullet(id="upd-b1", content="Original", section="test")
        await storage.add_bullet(context_id, bullet)

        # Then update via delta
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.UPDATE_BULLET,
                    target_id="upd-b1",
                    content="Updated content",
                    section="new_section",
                ),
            ],
        )
        result = await engine.apply_batch(batch)
        assert result.bullets_updated == 1

        loaded = await storage.get_bullet("upd-b1")
        assert loaded.content == "Updated content"
        assert loaded.section == "new_section"

    async def test_apply_remove_bullet(self, storage, context_id):
        engine = DeltaEngine(storage)
        bullet = Bullet(id="rm-b1", content="Will be removed")
        await storage.add_bullet(context_id, bullet)

        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.REMOVE_BULLET,
                    target_id="rm-b1",
                ),
            ],
        )
        result = await engine.apply_batch(batch)
        assert result.bullets_removed == 1

        bullets = await storage.list_bullets(context_id)
        assert len(bullets) == 0

    async def test_apply_merge_bullets(self, storage, context_id):
        engine = DeltaEngine(storage)
        b1 = Bullet(id="merge-a", content="Fact A", salience=0.8, hit_count=2)
        b2 = Bullet(id="merge-b", content="Fact B (duplicate)", salience=0.6, hit_count=1)
        await storage.add_bullet(context_id, b1)
        await storage.add_bullet(context_id, b2)

        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.MERGE_BULLETS,
                    target_ids=["merge-a", "merge-b"],
                    content="Merged fact",
                ),
            ],
        )
        result = await engine.apply_batch(batch)
        assert result.bullets_merged == 1

        # The higher-salience bullet should survive
        surviving = await storage.get_bullet("merge-a")
        assert surviving is not None
        assert surviving.content == "Merged fact"
        assert surviving.hit_count == 3  # Aggregated
        assert surviving.salience == 0.8  # Max of both

        # The lower-salience one should be removed
        removed = await storage.list_bullets(context_id)
        assert len(removed) == 1

    async def test_apply_add_schema(self, storage, context_id):
        engine = DeltaEngine(storage)
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.ADD_SCHEMA,
                    content="error_handling",
                    reasoning="Pattern for retry with backoff",
                ),
            ],
        )
        await engine.apply_batch(batch)
        schemas = await storage.list_schemas(context_id)
        assert len(schemas) == 1
        assert schemas[0].name == "error_handling"

    async def test_batch_persisted(self, storage, context_id):
        engine = DeltaEngine(storage)
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="p1", content="Fact"),
            ],
        )
        result = await engine.apply_batch(batch)

        loaded = await storage.get_delta_batch(result.id)
        assert loaded is not None
        assert loaded.bullets_added == 1

    async def test_rollback_add(self, storage, context_id):
        engine = DeltaEngine(storage)
        # Add a bullet via delta
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.ADD_BULLET,
                    target_id="rollback-b1",
                    content="Will be rolled back",
                    previous_state={},  # Needed for rollback
                ),
            ],
        )
        batch = await engine.apply_batch(batch)

        # Now rollback
        success = await engine.rollback_batch(batch.id)
        assert success is True

        # Bullet should be removed
        bullet = await storage.get_bullet("rollback-b1")
        # After rollback of an ADD, the bullet is removed
        if bullet is not None:
            assert bullet.is_active is False or bullet is None

    async def test_rollback_nonexistent(self, storage, context_id):
        engine = DeltaEngine(storage)
        success = await engine.rollback_batch("nonexistent-id")
        assert success is False

    async def test_multiple_operations_in_batch(self, storage, context_id):
        engine = DeltaEngine(storage)
        batch = DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="m1", content="Fact 1", bullet_type="fact"),
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="m2", content="Fact 2", bullet_type="fact"),
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="m3", content="Strategy 1", bullet_type="strategy"),
            ],
        )
        result = await engine.apply_batch(batch)
        assert result.bullets_added == 3

        bullets = await storage.list_bullets(context_id)
        assert len(bullets) == 3
