"""Tests for Phase 6: Mem-α-inspired episodic bullet sub-type.

Coverage:
1. BulletType.EPISODIC exists and is accepted by the Bullet model.
2. The Reflector prompt mentions "episodic" as a valid insight_type.
3. Curator merges two near-paraphrase EPISODIC insights at the looser
   threshold (0.85), even though FACT-typed paraphrases at the same
   similarity would not merge.
4. Curator does NOT merge an episodic with a non-episodic of the same
   topic — semantic vs episodic memories stay separate.
5. The contradiction-detection branch is skipped for episodic insights
   (events don't "contradict" each other).
"""

from __future__ import annotations

import json
import math
import uuid

import pytest

from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.ingestion import (
    CuratorEngine,
    IngestionEngine,
    REFLECTOR_SYSTEM_PROMPT,
)
from engram.core.models import (
    Bullet,
    BulletType,
    Context,
    DeltaOpType,
    IntentAnchor,
    Reflection,
    ReflectionInsight,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class FixedEmbeddingLLM(LLMAdapter):
    """LLM that maps text -> embedding via a fixed table so tests can
    deterministically place pairs above/below thresholds."""

    def __init__(self, embed_map: dict[str, list[float]]):
        self.embed_map = embed_map

    async def complete(self, *a, **k):
        return json.dumps({
            "new_insights": [], "strategies_that_worked": [],
            "failure_modes": [], "prediction_errors": [],
            "open_questions": [], "confidence": 0.5,
        })

    async def embed(self, text):
        if text in self.embed_map:
            return self.embed_map[text]
        # default: deterministic word-bucket embedding
        vec = [0.0] * 32
        for w in text.lower().split():
            vec[hash(w) % 32] += 1.0
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "ep.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="ep", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)
    return ctx.id


# ── 1. Enum value present ────────────────────────────────────────────────────

def test_episodic_bullet_type_exists():
    assert BulletType.EPISODIC.value == "episodic"
    b = Bullet(content="At 14:30 the user agreed", bullet_type=BulletType.EPISODIC)
    assert b.bullet_type == BulletType.EPISODIC


# ── 2. Reflector prompt mentions episodic ───────────────────────────────────

def test_reflector_prompt_mentions_episodic():
    assert "episodic" in REFLECTOR_SYSTEM_PROMPT.lower()


# ── 3. Episodic merges at 0.88, fact would not ──────────────────────────────

def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


async def test_episodic_merges_at_loose_threshold(storage, context_id):
    # Construct two vectors with cosine ≈ 0.88 — above the episodic threshold
    # (0.85) but below the fact threshold (0.92).
    v1 = _norm([1.0, 0.0, 0.0, 0.0])
    v2 = _norm([0.88, 0.475, 0.0, 0.0])  # cos(v1, v2) ≈ 0.88
    embed_map = {
        "At 14:30 user agreed PaddleOCR": v1,
        "At 14:30 user said yes PaddleOCR": v2,
    }
    llm = FixedEmbeddingLLM(embed_map)
    curator = CuratorEngine(storage, llm)

    # Seed an existing episodic bullet
    seeded = Bullet(
        id=str(uuid.uuid4())[:8], context_id=str(context_id),
        section="events",
        content="At 14:30 user agreed PaddleOCR",
        bullet_type=BulletType.EPISODIC,
        embedding=v1,
    )
    await storage.add_bullet(str(context_id), seeded)

    insight = ReflectionInsight(
        content="At 14:30 user said yes PaddleOCR",
        insight_type="episodic", suggested_section="events",
    )
    reflection = Reflection(new_insights=[insight], confidence=0.7)
    batch = await curator.curate(str(context_id), reflection)
    # The near-duplicate episodic folds into the existing bullet via UPDATE_BULLET
    # (not a no-op MERGE over a non-existent "__new__" target, and not a new ADD).
    op_types = [op.op_type for op in batch.operations]
    assert DeltaOpType.UPDATE_BULLET in op_types
    assert DeltaOpType.ADD_BULLET not in op_types

    update_op = next(op for op in batch.operations
                     if op.op_type == DeltaOpType.UPDATE_BULLET)
    assert update_op.target_id == seeded.id

    # Storage-state assertion (the gap that let the inert-merge bug hide):
    # apply the batch and confirm the new insight was actually persisted —
    # exactly one bullet remains, carrying the richer (longer) content.
    engine = DeltaEngine(storage)
    await engine.apply_batch(batch)
    bullets = await storage.list_bullets(str(context_id))
    assert len(bullets) == 1
    assert bullets[0].id == seeded.id
    # The longer phrasing wins ("...said yes..." is longer than "...agreed...").
    assert bullets[0].content == "At 14:30 user said yes PaddleOCR"


async def test_fact_does_not_merge_at_same_threshold(storage, context_id):
    """Same vectors as the episodic test (cos ≈ 0.88), but as type=fact.
    Should NOT merge because facts use threshold 0.92."""
    v1 = _norm([1.0, 0.0, 0.0, 0.0])
    v2 = _norm([0.88, 0.475, 0.0, 0.0])
    embed_map = {
        "PaddleOCR is fast on PDFs": v1,
        "PaddleOCR fast for PDFs": v2,
    }
    llm = FixedEmbeddingLLM(embed_map)
    curator = CuratorEngine(storage, llm)

    seeded = Bullet(
        id=str(uuid.uuid4())[:8], context_id=str(context_id),
        section="ocr",
        content="PaddleOCR is fast on PDFs",
        bullet_type=BulletType.FACT,
        embedding=v1,
    )
    await storage.add_bullet(str(context_id), seeded)

    insight = ReflectionInsight(
        content="PaddleOCR fast for PDFs",
        insight_type="fact", suggested_section="ocr",
    )
    reflection = Reflection(new_insights=[insight], confidence=0.7)
    batch = await curator.curate(str(context_id), reflection)
    op_types = [op.op_type for op in batch.operations]
    # Not merged/folded; should be added as new (cos 0.88 < 0.92 fact threshold).
    assert DeltaOpType.ADD_BULLET in op_types
    assert DeltaOpType.MERGE_BULLETS not in op_types
    assert DeltaOpType.UPDATE_BULLET not in op_types


# ── 4. Episodic doesn't collapse into a same-topic non-episodic ──────────────

async def test_episodic_doesnt_merge_with_non_episodic(storage, context_id):
    v1 = _norm([1.0, 0.0, 0.0, 0.0])
    v2 = _norm([0.95, 0.31, 0.0, 0.0])  # cos ≈ 0.95 — above both thresholds
    embed_map = {
        "PaddleOCR is fast": v1,
        "At 14:30 user noticed PaddleOCR is fast": v2,
    }
    llm = FixedEmbeddingLLM(embed_map)
    curator = CuratorEngine(storage, llm)

    # Seed a FACT bullet
    seeded = Bullet(
        id=str(uuid.uuid4())[:8], context_id=str(context_id),
        section="ocr",
        content="PaddleOCR is fast",
        bullet_type=BulletType.FACT,
        embedding=v1,
    )
    await storage.add_bullet(str(context_id), seeded)

    # New EPISODIC insight with very high cosine to the existing FACT
    insight = ReflectionInsight(
        content="At 14:30 user noticed PaddleOCR is fast",
        insight_type="episodic", suggested_section="events",
    )
    reflection = Reflection(new_insights=[insight], confidence=0.7)
    batch = await curator.curate(str(context_id), reflection)

    # Episodic stays separate — added, not merged/folded into the fact.
    op_types = [op.op_type for op in batch.operations]
    assert DeltaOpType.ADD_BULLET in op_types
    assert DeltaOpType.MERGE_BULLETS not in op_types
    assert DeltaOpType.UPDATE_BULLET not in op_types


# ── 5. Contradiction detection skipped for episodic ──────────────────────────

async def test_contradiction_branch_skipped_for_episodic(storage, context_id):
    """The hard-coded negation cue list would normally flag two opposing
    statements at high similarity (>=0.75 plus a negation cue). Episodic
    memories are events and shouldn't "contradict" each other — both should
    be stored. Use cos ≈ 0.80, which is above the contradiction min (0.75)
    but below the episodic merge threshold (0.85), so we land in the
    contradiction branch where the skip is observable."""
    v1 = _norm([1.0, 0.0])
    v2 = _norm([0.80, 0.6])  # cos = 0.80
    embed_map = {
        "At 14:30 deploy succeeded": v1,
        "At 14:32 deploy failed": v2,
    }
    llm = FixedEmbeddingLLM(embed_map)
    curator = CuratorEngine(storage, llm)

    seeded = Bullet(
        id=str(uuid.uuid4())[:8], context_id=str(context_id),
        section="events",
        content="At 14:30 deploy succeeded",
        bullet_type=BulletType.EPISODIC,
        embedding=v1,
    )
    await storage.add_bullet(str(context_id), seeded)

    insight = ReflectionInsight(
        content="At 14:32 deploy failed",
        insight_type="episodic", suggested_section="events",
    )
    reflection = Reflection(new_insights=[insight], confidence=0.7)
    batch = await curator.curate(str(context_id), reflection)

    add_ops = [op for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert len(add_ops) >= 1
    # No contradiction-flagging language in the reasoning of the add op.
    assert not any("contradicts" in (op.reasoning or "").lower() for op in add_ops)
