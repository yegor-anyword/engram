"""Delta operation engine — all graph mutations go through here.

Every mutation to the concept graph is expressed as atomic delta operations.
The full context is NEVER regenerated wholesale. This prevents context collapse
and enables full auditability + rollback.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    BulletType,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    SchemaNode,
    SourceType,
)
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeltaEngine:
    """Applies delta operations atomically to the concept graph.

    ALL mutations to the graph MUST go through this engine.
    No direct writes to bullets or edges that bypass delta tracking.
    """

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    async def apply_batch(self, batch: DeltaBatch) -> DeltaBatch:
        """Apply a batch of delta operations atomically.

        Returns the batch with updated stats.
        """
        added = 0
        updated = 0
        removed = 0
        merged = 0

        for op in batch.operations:
            match op.op_type:
                case DeltaOpType.ADD_BULLET:
                    await self._apply_add_bullet(batch.context_id, op)
                    added += 1

                case DeltaOpType.UPDATE_BULLET:
                    await self._apply_update_bullet(op)
                    updated += 1

                case DeltaOpType.REMOVE_BULLET:
                    await self._apply_remove_bullet(op)
                    removed += 1

                case DeltaOpType.MERGE_BULLETS:
                    await self._apply_merge_bullets(batch.context_id, op)
                    merged += 1

                case DeltaOpType.ADD_SCHEMA:
                    await self._apply_add_schema(batch.context_id, op)

                case DeltaOpType.UPDATE_SCHEMA:
                    await self._apply_update_schema(op)

                case _:
                    logger.warning("Unknown delta op type: %s", op.op_type)

        batch.bullets_added = added
        batch.bullets_updated = updated
        batch.bullets_removed = removed
        batch.bullets_merged = merged

        # Persist the batch for auditability
        await self.storage.save_delta_batch(batch)

        return batch

    async def rollback_batch(self, delta_batch_id: str) -> bool:
        """Roll back a delta batch by applying inverse operations."""
        batch = await self.storage.get_delta_batch(delta_batch_id)
        if batch is None:
            return False

        # Apply inverse operations in reverse order
        for op in reversed(batch.operations):
            if op.previous_state is None:
                continue

            match op.op_type:
                case DeltaOpType.ADD_BULLET:
                    # Undo add → remove
                    if op.target_id:
                        await self.storage.remove_bullet(op.target_id)

                case DeltaOpType.UPDATE_BULLET:
                    # Undo update → restore previous state
                    if op.target_id:
                        bullet = await self.storage.get_bullet(op.target_id)
                        if bullet:
                            prev = op.previous_state
                            bullet.content = prev.get("content", bullet.content)
                            bullet.salience = prev.get("salience", bullet.salience)
                            bullet.confidence = prev.get("confidence", bullet.confidence)
                            bullet.section = prev.get("section", bullet.section)
                            await self.storage.update_bullet(bullet)

                case DeltaOpType.REMOVE_BULLET:
                    # Undo remove → reactivate
                    if op.target_id:
                        bullet = await self.storage.get_bullet(op.target_id)
                        if bullet:
                            bullet.is_active = True
                            await self.storage.update_bullet(bullet)

        return True

    async def _apply_add_bullet(self, context_id: str, op: DeltaOperation) -> None:
        # Map DeltaSource → SourceType (they have different enum values)
        source_map = {
            DeltaSource.REFLECTOR: SourceType.REFLECTION,
            DeltaSource.CURATOR: SourceType.REFLECTION,
            DeltaSource.USER: SourceType.USER_INPUT,
            DeltaSource.CONSOLIDATION: SourceType.CONSOLIDATION,
        }
        source_type = source_map.get(op.source, SourceType.REFLECTION) if op.source else SourceType.REFLECTION

        bullet = Bullet(
            id=op.target_id or str(uuid.uuid4())[:8],
            section=op.section or "general",
            content=op.content or "",
            bullet_type=BulletType(op.bullet_type) if op.bullet_type else BulletType.FACT,
            source_type=source_type,
            salience=op.confidence,
            confidence=op.confidence,
            source_session=op.session_id,
            source_agent=op.agent_id,
        )
        await self.storage.add_bullet(context_id, bullet)

    async def _apply_update_bullet(self, op: DeltaOperation) -> None:
        if not op.target_id:
            return
        bullet = await self.storage.get_bullet(op.target_id)
        if bullet is None:
            return

        # Save previous state for rollback
        op.previous_state = {
            "content": bullet.content,
            "salience": bullet.salience,
            "confidence": bullet.confidence,
            "section": bullet.section,
        }

        if op.content:
            bullet.content = op.content
        if op.section:
            bullet.section = op.section
        if op.confidence:
            bullet.confidence = op.confidence
        await self.storage.update_bullet(bullet)

    async def _apply_remove_bullet(self, op: DeltaOperation) -> None:
        if not op.target_id:
            return
        bullet = await self.storage.get_bullet(op.target_id)
        if bullet:
            op.previous_state = {"is_active": bullet.is_active}
            await self.storage.remove_bullet(op.target_id)

    async def _apply_merge_bullets(self, context_id: str, op: DeltaOperation) -> None:
        """Merge multiple bullets into one — keep the most specific/highest salience."""
        if not op.target_ids or len(op.target_ids) < 2:
            return

        bullets_to_merge = []
        for bid in op.target_ids:
            bullet = await self.storage.get_bullet(bid)
            if bullet:
                bullets_to_merge.append(bullet)

        if len(bullets_to_merge) < 2:
            return

        # Keep the bullet with highest salience, deactivate others
        best = max(bullets_to_merge, key=lambda b: b.salience)
        if op.content:
            best.content = op.content
        best.recall_count = sum(b.recall_count for b in bullets_to_merge)
        best.hit_count = sum(b.hit_count for b in bullets_to_merge)
        best.miss_count = sum(b.miss_count for b in bullets_to_merge)
        best.salience = max(b.salience for b in bullets_to_merge)
        await self.storage.update_bullet(best)

        for b in bullets_to_merge:
            if b.id != best.id:
                await self.storage.remove_bullet(b.id)

    async def _apply_add_schema(self, context_id: str, op: DeltaOperation) -> None:
        if not op.content:
            return
        schema = SchemaNode(
            name=op.content,
            description=op.reasoning,
        )
        await self.storage.add_schema(context_id, schema)

    async def _apply_update_schema(self, op: DeltaOperation) -> None:
        if not op.target_id:
            return
        schema = await self.storage.get_schema(op.target_id)
        if schema and op.content:
            schema.description = op.content
            await self.storage.update_schema(schema)
