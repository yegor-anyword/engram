"""Tests for the data lifecycle module — archive, restore, purge, capacity."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from engram.core.models import (
    Bullet,
    BulletType,
    CapacityStatus,
    Context,
    IntentAnchor,
    LifecycleConfig,
    LifecycleState,
    SourceType,
)
from engram.storage.sqlite import SQLiteBackend


def _utcnow():
    return datetime.now(timezone.utc)


@pytest.fixture
async def storage(tmp_path):
    db = SQLiteBackend(db_path=str(tmp_path / "test.db"))
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def ctx(storage):
    """Create a test context."""
    context = Context(
        name="Test Lifecycle",
        intent=IntentAnchor(objective="Test lifecycle transitions"),
        lifecycle_config=LifecycleConfig(max_active_bullets=100),
    )
    return await storage.create_context(context)


@pytest.fixture
async def bullet(storage, ctx):
    """Create a test bullet."""
    b = Bullet(content="Test bullet for lifecycle", bullet_type=BulletType.FACT)
    return await storage.add_bullet(str(ctx.id), b)


class TestArchiveBullet:
    async def test_archive_sets_lifecycle_state(self, storage, ctx, bullet):
        """Archive transitions lifecycle_state to 'archived'."""
        success = await storage.archive_bullet(str(ctx.id), bullet.id, reason="stale")
        assert success is True

        archived = await storage.get_bullet(bullet.id)
        assert archived is not None
        assert archived.lifecycle_state == LifecycleState.ARCHIVED
        assert archived.is_archived is True
        assert archived.archive_reason == "stale"
        assert archived.archived_at is not None

    async def test_archive_nonexistent_returns_false(self, storage, ctx):
        """Archiving a non-existent bullet returns False."""
        success = await storage.archive_bullet(str(ctx.id), "nonexistent")
        assert success is False

    async def test_archive_already_archived_returns_false(self, storage, ctx, bullet):
        """Archiving an already-archived bullet returns False."""
        await storage.archive_bullet(str(ctx.id), bullet.id)
        success = await storage.archive_bullet(str(ctx.id), bullet.id)
        assert success is False


class TestRestoreBullet:
    async def test_restore_sets_lifecycle_active(self, storage, ctx, bullet):
        """Restore transitions lifecycle_state back to 'active'."""
        await storage.archive_bullet(str(ctx.id), bullet.id)
        restored = await storage.restore_bullet(str(ctx.id), bullet.id)

        assert restored is not None
        assert restored.lifecycle_state == LifecycleState.ACTIVE
        assert restored.is_archived is False
        assert restored.archive_reason is None
        assert restored.archived_at is None

    async def test_archive_restore_preserves_content(self, storage, ctx, bullet):
        """Archive-restore roundtrip preserves content and salience."""
        original_content = bullet.content
        original_salience = bullet.salience

        await storage.archive_bullet(str(ctx.id), bullet.id)
        restored = await storage.restore_bullet(str(ctx.id), bullet.id)

        assert restored is not None
        assert restored.content == original_content
        assert restored.salience == original_salience

    async def test_restore_nonexistent_returns_none(self, storage, ctx):
        """Restoring a non-existent bullet returns None."""
        result = await storage.restore_bullet(str(ctx.id), "nonexistent")
        assert result is None


class TestPurgeBullet:
    async def test_purge_permanently_deletes(self, storage, ctx, bullet):
        """Purge permanently deletes a bullet."""
        success = await storage.purge_bullet(str(ctx.id), bullet.id)
        assert success is True

        result = await storage.get_bullet(bullet.id)
        assert result is None

    async def test_purge_nonexistent_returns_false(self, storage, ctx):
        """Purging a non-existent bullet returns False."""
        success = await storage.purge_bullet(str(ctx.id), "nonexistent")
        assert success is False


class TestPurgeExpiredArchives:
    async def test_purge_expired_clears_old(self, storage, ctx):
        """Purge expired archives clears old archived bullets."""
        # Create and archive a bullet with old archived_at
        b = Bullet(content="Old bullet", bullet_type=BulletType.FACT)
        b = await storage.add_bullet(str(ctx.id), b)
        await storage.archive_bullet(str(ctx.id), b.id)

        # Manually backdate the archived_at
        db = await storage._get_db()
        old_date = (_utcnow() - timedelta(days=200)).isoformat()
        await db.execute(
            "UPDATE bullets SET archived_at=? WHERE id=?", (old_date, b.id)
        )
        await db.commit()

        count = await storage.purge_expired_archives(str(ctx.id), purge_after_days=180)
        assert count == 1

        result = await storage.get_bullet(b.id)
        assert result is None

    async def test_purge_expired_keeps_recent(self, storage, ctx):
        """Purge expired archives keeps recently-archived bullets."""
        b = Bullet(content="Recent bullet", bullet_type=BulletType.FACT)
        b = await storage.add_bullet(str(ctx.id), b)
        await storage.archive_bullet(str(ctx.id), b.id)

        count = await storage.purge_expired_archives(str(ctx.id), purge_after_days=180)
        assert count == 0

        result = await storage.get_bullet(b.id)
        assert result is not None


class TestGetArchivedBullets:
    async def test_list_archived_bullets(self, storage, ctx):
        """Get archived bullets returns only archived bullets."""
        b1 = Bullet(content="Active bullet", bullet_type=BulletType.FACT)
        b2 = Bullet(content="Archived bullet", bullet_type=BulletType.FACT)
        await storage.add_bullet(str(ctx.id), b1)
        b2 = await storage.add_bullet(str(ctx.id), b2)
        await storage.archive_bullet(str(ctx.id), b2.id)

        archived = await storage.get_archived_bullets(str(ctx.id))
        assert len(archived) == 1
        assert archived[0].id == b2.id


class TestCapacityStatus:
    async def test_capacity_status_counts(self, storage, ctx, bullet):
        """Capacity status returns correct counts."""
        capacity = await storage.get_capacity_status(str(ctx.id), 100)
        assert capacity.active_bullet_count == 1
        assert capacity.max_active_bullets == 100
        assert capacity.archived_bullet_count == 0

    async def test_pressure_level_normal(self, storage, ctx):
        """Pressure level is 'normal' below 80%."""
        capacity = await storage.get_capacity_status(str(ctx.id), 100)
        assert capacity.pressure_level == "normal"

    async def test_pressure_level_computation(self):
        """Pressure level is computed correctly."""
        assert CapacityStatus(active_bullet_count=79, max_active_bullets=100).pressure_level == "normal"
        assert CapacityStatus(active_bullet_count=80, max_active_bullets=100).pressure_level == "high"
        assert CapacityStatus(active_bullet_count=95, max_active_bullets=100).pressure_level == "critical"
        assert CapacityStatus(active_bullet_count=100, max_active_bullets=100).pressure_level == "full"


class TestPurgeContext:
    async def test_purge_context_deletes_all_data(self, storage, ctx, bullet):
        """Purge context deletes all data for the context."""
        success = await storage.purge_context(str(ctx.id))
        assert success is True

        # Context should be gone
        result = await storage.get_context(ctx.id)
        assert result is None

        # Bullet should be gone
        result = await storage.get_bullet(bullet.id)
        assert result is None

    async def test_purge_nonexistent_context(self, storage):
        """Purging a non-existent context returns False."""
        success = await storage.purge_context("nonexistent-id")
        assert success is False


class TestPurgeUser:
    async def test_purge_user_deletes_all_contexts(self, storage):
        """Purge user deletes all contexts owned by a user."""
        ctx1 = Context(
            name="User1 Context A",
            owner="user-1",
            intent=IntentAnchor(objective="Test A"),
        )
        ctx2 = Context(
            name="User1 Context B",
            owner="user-1",
            intent=IntentAnchor(objective="Test B"),
        )
        ctx3 = Context(
            name="User2 Context",
            owner="user-2",
            intent=IntentAnchor(objective="Test C"),
        )
        await storage.create_context(ctx1)
        await storage.create_context(ctx2)
        await storage.create_context(ctx3)

        count = await storage.purge_user("user-1")
        assert count == 2

        # user-1's contexts should be gone
        contexts = await storage.list_contexts(owner="user-1")
        assert len(contexts) == 0

        # user-2's context should remain
        contexts = await storage.list_contexts(owner="user-2")
        assert len(contexts) == 1


class TestProtectedTypes:
    async def test_lifecycle_config_protected_types(self):
        """Protected types default to decision and principle."""
        config = LifecycleConfig()
        assert "decision" in config.protected_types
        assert "principle" in config.protected_types


class TestLifecycleConfigStorage:
    async def test_lifecycle_config_persisted(self, storage):
        """Lifecycle config is saved and loaded correctly."""
        config = LifecycleConfig(max_active_bullets=500, purge_after_days=90)
        ctx = Context(
            name="Config Test",
            intent=IntentAnchor(objective="Test config"),
            lifecycle_config=config,
        )
        created = await storage.create_context(ctx)

        loaded = await storage.get_context(created.id)
        assert loaded is not None
        assert loaded.lifecycle_config.max_active_bullets == 500
        assert loaded.lifecycle_config.purge_after_days == 90
