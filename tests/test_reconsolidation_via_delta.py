"""Tests for Phase 3b: reconsolidation routed through DeltaEngine.

Coverage:
1. After a successful-outcome commit referencing a materialization_id, a
   delta batch with RECONSOLIDATE_BULLET ops is persisted.
2. SUCCESS outcome → hit_count + recall_count incremented, salience boosted.
3. FAILURE outcome → miss_count + recall_count incremented, salience decayed.
4. PARTIAL / UNKNOWN outcome → recall_count only; counters and salience steady.
5. Rolling back the reconsolidation batch restores the original stats.
6. salience is clamped to [0.05, 1.0] after repeated multiplications.
"""

from __future__ import annotations

import json
import uuid

import pytest

from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.ingestion import IngestionEngine
from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    Bullet,
    BulletType,
    ContentType,
    Context,
    DeltaOpType,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
    MaterializationRecord,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class CannedLLM(LLMAdapter):
    def __init__(self, response: dict | None = None):
        self.response = response or {
            "new_insights": [],
            "strategies_that_worked": [],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.6,
        }

    async def complete(self, *a, **kw):
        return json.dumps(self.response)

    async def embed(self, text):
        vec = [0.0] * 32
        for w in text.lower().split():
            vec[hash(w) % 32] += 1.0
        import math
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "rc.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="rc", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)
    return ctx.id


async def _seed_bullet(storage, context_id, salience: float = 0.5) -> Bullet:
    b = Bullet(
        id=str(uuid.uuid4())[:8],
        context_id=str(context_id),
        section="general",
        content="seed bullet content",
        bullet_type=BulletType.FACT,
        salience=salience,
        confidence=0.7,
    )
    await storage.add_bullet(str(context_id), b)
    return b


async def _seed_materialization(storage, context_id, bullet_ids: list[str]) -> str:
    mat_id = str(uuid.uuid4())
    rec = MaterializationRecord(
        id=mat_id, context_id=str(context_id),
        bullets_included=bullet_ids, token_count=100, target_model="claude",
        query="anything",
    )
    await storage.save_materialization(rec)
    return mat_id


# ── 1. SUCCESS: stats updated + delta batch saved ────────────────────────────

async def test_success_reconsolidation_writes_delta_batch(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])

    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="t",
        content="post-recall commit content",
        content_type=ContentType.CONVERSATION,
        materialization_id=mat_id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
    )

    updated = await storage.get_bullet(b.id)
    assert updated.recall_count == 1
    assert updated.hit_count == 1
    assert updated.miss_count == 0
    assert updated.salience == pytest.approx(0.5 * 1.05, rel=0.01)

    # A reconsolidation delta batch should be present.
    batches = await storage.list_delta_batches(str(context_id), limit=20)
    reconsolidation_batches = [
        b for b in batches if b.trigger == "reconsolidation"
        and any(op.op_type == DeltaOpType.RECONSOLIDATE_BULLET for op in b.operations)
    ]
    assert len(reconsolidation_batches) == 1
    op = reconsolidation_batches[0].operations[0]
    # After apply, rollback_state carries the prior values for rollback. The
    # caller-provided deltas remain untouched in previous_state.
    assert op.rollback_state["recall_count"] == 0
    assert op.rollback_state["hit_count"] == 0
    assert op.rollback_state["salience"] == pytest.approx(0.5)
    # previous_state still carries the original input deltas — unchanged by apply.
    assert op.previous_state["recall_delta"] == 1
    assert op.previous_state["hit_delta"] == 1


# ── 2. FAILURE outcome → miss + decay ────────────────────────────────────────

async def test_failure_reconsolidation_decays_salience(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])

    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="t",
        content="post-recall failure",
        materialization_id=mat_id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.FAILURE),
    )

    updated = await storage.get_bullet(b.id)
    assert updated.recall_count == 1
    assert updated.hit_count == 0
    assert updated.miss_count == 1
    assert updated.salience == pytest.approx(0.5 * 0.95, rel=0.01)


# ── 3. PARTIAL / UNKNOWN: recall counts only ─────────────────────────────────

async def test_partial_outcome_only_bumps_recall_count(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])

    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="t",
        content="post-recall partial",
        materialization_id=mat_id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.PARTIAL),
    )

    updated = await storage.get_bullet(b.id)
    assert updated.recall_count == 1
    assert updated.hit_count == 0
    assert updated.miss_count == 0
    assert updated.salience == pytest.approx(0.5)


# ── 4. Rollback restores prior stats ─────────────────────────────────────────

async def test_rollback_reconsolidation_batch(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])

    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="t",
        content="commit success",
        materialization_id=mat_id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
    )

    # find the reconsolidation batch
    batches = await storage.list_delta_batches(str(context_id), limit=20)
    rc_batch = next(b for b in batches if b.trigger == "reconsolidation")

    de = DeltaEngine(storage)
    ok = await de.rollback_batch(rc_batch.id)
    assert ok is True

    restored = await storage.get_bullet(b.id)
    assert restored.recall_count == 0
    assert restored.hit_count == 0
    assert restored.miss_count == 0
    assert restored.salience == pytest.approx(0.5)


# ── 5. Salience floor 0.05 holds over many failures ──────────────────────────

async def test_salience_floor_holds(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.06)
    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())

    # 50 failing commits each multiply salience by 0.95 — should clamp to 0.05.
    for _ in range(50):
        mat_id = await _seed_materialization(storage, context_id, [b.id])
        await engine.commit(
            context_id=context_id, agent_id="t",
            content=f"failure step {_}",  # unique content to avoid hash dedup
            materialization_id=mat_id,
            feedback=ExecutionFeedback(outcome=FeedbackOutcome.FAILURE),
        )

    updated = await storage.get_bullet(b.id)
    assert updated.salience >= 0.05
    assert updated.miss_count == 50
