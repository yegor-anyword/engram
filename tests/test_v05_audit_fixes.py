"""Regression tests for the v0.5 audit fixes.

Each test targets a specific bug found in the post-v0.5 audit. They assert
*persisted storage state* (not just delta-op types/counts), which is the gap
that let several of these bugs hide behind green tests.

Covered:
  Bug #1 — Curator near-duplicate folds into the existing bullet (UPDATE_BULLET)
           instead of an inert MERGE over a non-existent "__new__" target.
  Bug #2 — Reconsolidation is idempotent: a replayed materialization_id does not
           double-count hits / compound salience.
  Bug #3 — Reconsolidation skips archived/inactive bullets.
  Bug #4 — Worked-example text is truncated so it can't blow the token budget.
  Bug #5 — The ≤512-token core-memory bound is enforced at the delta-apply layer.
  Bug #6 — update_context() persists core_memory.
  Bug #7 — MaterializeRequest validates mmr_lambda / worked_example_threshold.
"""

from __future__ import annotations

import json
import math
import uuid

import pytest
from pydantic import ValidationError

from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.ingestion import CuratorEngine, IngestionEngine
from engram.core.materialization import (
    WORKED_EXAMPLE_INPUT_MAX_CHARS,
    _truncate,
)
from engram.core.models import (
    CORE_MEMORY_MAX_TOKENS,
    Bullet,
    BulletType,
    ContentType,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
    MaterializationRecord,
    MaterializeRequest,
    Reflection,
    ReflectionInsight,
    cap_core_memory,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class CannedLLM(LLMAdapter):
    """Empty reflection by default; deterministic word-bucket embedding."""

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
        import zlib
        vec = [0.0] * 32
        for w in text.lower().split():
            vec[zlib.crc32(w.encode("utf-8")) % 32] += 1.0
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


class FixedEmbeddingLLM(LLMAdapter):
    """Returns caller-specified embeddings so similarity is fully controlled."""

    def __init__(self, embed_map: dict[str, list[float]]):
        self.embed_map = embed_map

    async def complete(self, *a, **kw):
        return json.dumps({"new_insights": [], "confidence": 0.6})

    async def embed(self, text):
        return self.embed_map[text]


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "audit.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="audit", intent=IntentAnchor(objective="x"))
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


# ── Bug #1: near-duplicate folds into the existing bullet (and persists) ─────

async def test_fact_near_duplicate_folds_into_existing_bullet(storage, context_id):
    """A near-duplicate fact (cos ≥ 0.92) updates the existing bullet in place,
    persisting the richer content — it is NOT silently dropped."""
    v1 = _norm([1.0, 0.0, 0.0])
    v2 = _norm([0.97, 0.243, 0.0])  # cos ≈ 0.97 ≥ 0.92 fact threshold
    llm = FixedEmbeddingLLM({
        "PaddleOCR is fast": v1,
        "PaddleOCR is fast on large multi-column PDF documents": v2,
    })
    curator = CuratorEngine(storage, llm)

    seeded = Bullet(
        id=str(uuid.uuid4())[:8], context_id=str(context_id), section="ocr",
        content="PaddleOCR is fast", bullet_type=BulletType.FACT, embedding=v1,
    )
    await storage.add_bullet(str(context_id), seeded)

    insight = ReflectionInsight(
        content="PaddleOCR is fast on large multi-column PDF documents",
        insight_type="fact", suggested_section="ocr",
    )
    batch = await curator.curate(
        str(context_id), Reflection(new_insights=[insight], confidence=0.7),
    )
    assert DeltaOpType.UPDATE_BULLET in [op.op_type for op in batch.operations]

    await DeltaEngine(storage).apply_batch(batch)
    bullets = await storage.list_bullets(str(context_id))
    assert len(bullets) == 1                       # new insight not dropped
    assert bullets[0].id == seeded.id
    assert "multi-column PDF documents" in bullets[0].content  # richer content kept


# ── Bug #2: reconsolidation is idempotent on replay ──────────────────────────

async def test_reconsolidation_idempotent_on_replayed_materialization(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])
    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())

    for i in range(3):  # same materialization_id three times
        await engine.commit(
            context_id=context_id, agent_id="t",
            content=f"post-recall commit {i}",
            content_type=ContentType.CONVERSATION,
            materialization_id=mat_id,
            feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
        )

    updated = await storage.get_bullet(b.id)
    # Reinforced exactly once, not three times.
    assert updated.recall_count == 1
    assert updated.hit_count == 1
    assert updated.salience == pytest.approx(0.5 * 1.05, rel=0.01)

    # The record is stamped consumed.
    rec = await storage.get_materialization(mat_id)
    assert rec.reconsolidated_at is not None


# ── Bug #3: archived bullets are not reinforced ──────────────────────────────

async def test_reconsolidation_skips_archived_bullet(storage, context_id):
    b = await _seed_bullet(storage, context_id, salience=0.5)
    mat_id = await _seed_materialization(storage, context_id, [b.id])

    # Archive the bullet after the (simulated) recall but before feedback.
    archived = await storage.archive_bullet(str(context_id), b.id)
    assert archived

    engine = IngestionEngine(storage, CannedLLM(), ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="t",
        content="post-recall commit on archived bullet",
        content_type=ContentType.CONVERSATION,
        materialization_id=mat_id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
    )

    after = await storage.get_bullet(b.id)
    # Untouched — an archived bullet should not be reinforced.
    assert after.recall_count == 0
    assert after.hit_count == 0
    assert after.salience == pytest.approx(0.5)


# ── Bug #4: worked-example text is truncated ─────────────────────────────────

def test_truncate_caps_long_text():
    long_text = "word " * 5000  # ~25k chars
    out = _truncate(long_text, WORKED_EXAMPLE_INPUT_MAX_CHARS)
    assert len(out) <= WORKED_EXAMPLE_INPUT_MAX_CHARS + len(" …[truncated]")
    assert out.endswith("…[truncated]")
    # Short text is returned unchanged.
    assert _truncate("short", WORKED_EXAMPLE_INPUT_MAX_CHARS) == "short"


# ── Bug #5: ≤512-token bound enforced at the delta-apply layer ───────────────

async def test_core_memory_cap_enforced_in_delta_apply(storage, context_id):
    oversized = "token " * 4000  # ~4000 tokens, far over the 512 cap
    op = DeltaOperation(
        op_type=DeltaOpType.UPDATE_CORE_MEMORY,
        target_id=str(context_id),
        content=oversized,
        source=DeltaSource.REFLECTOR,
        confidence=0.9,
    )
    batch = DeltaBatch(context_id=str(context_id), operations=[op])
    await DeltaEngine(storage).apply_batch(batch)

    ctx = await storage.get_context(context_id)
    # Stored value is the capped value, even though we bypassed IngestionEngine.
    assert ctx.core_memory == cap_core_memory(oversized)
    assert len(ctx.core_memory) < len(oversized)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(ctx.core_memory)) <= CORE_MEMORY_MAX_TOKENS
    except Exception:
        assert len(ctx.core_memory) <= CORE_MEMORY_MAX_TOKENS * 4


# ── Bug #6: update_context persists core_memory ──────────────────────────────

async def test_update_context_persists_core_memory(storage, context_id):
    ctx = await storage.get_context(context_id)
    ctx.core_memory = "a durable running summary"
    await storage.update_context(ctx)

    reloaded = await storage.get_context(context_id)
    assert reloaded.core_memory == "a durable running summary"


# ── Bug #7: MaterializeRequest validates its tunables ────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"mmr_lambda": -0.1},
    {"mmr_lambda": 1.5},
    {"worked_example_threshold": -0.01},
    {"worked_example_threshold": 1.01},
])
def test_materialize_request_rejects_out_of_range_tunables(kwargs):
    with pytest.raises(ValidationError):
        MaterializeRequest(**kwargs)


def test_materialize_request_accepts_in_range_tunables():
    req = MaterializeRequest(mmr_lambda=0.0, worked_example_threshold=1.0)
    assert req.mmr_lambda == 0.0
    assert req.worked_example_threshold == 1.0
