"""Tests for Phase 5: surfacing per-bullet usage counts in rendered output.

The goal: when include_usage_stats=True, the consumer LLM sees
"(used N×, success Y/Z)" appended to each bullet that has any usage history.
This nudges the model toward bullets with proven track record.

Coverage:
1. Default off — usage stats absent from rendered output.
2. Enabled — stats appear next to bullets with non-zero recall_count.
3. Stats omitted for bullets with zero recall (no signal yet — would just be noise).
4. All three renderers carry the suffix.
"""

from __future__ import annotations

import json
import uuid

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import IngestionEngine
from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    Bullet,
    BulletType,
    Context,
    IntentAnchor,
)
from engram.llm.adapter import LLMAdapter
from engram.renderers.claude import ClaudeRenderer
from engram.renderers.generic import GenericRenderer
from engram.renderers.gpt import GPTRenderer
from engram.storage.sqlite import SQLiteBackend


class StubLLM(LLMAdapter):
    async def complete(self, *a, **k):
        return json.dumps({
            "new_insights": [], "strategies_that_worked": [],
            "failure_modes": [], "prediction_errors": [],
            "open_questions": [], "confidence": 0.5,
        })

    async def embed(self, text):
        vec = [0.0] * 32
        for w in text.lower().split():
            vec[hash(w) % 32] += 1.0
        import math
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "us.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="us", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)
    return ctx.id


async def _seed_bullet(storage, context_id: uuid.UUID, *,
                       content: str, hit: int = 0, recall: int = 0, miss: int = 0,
                       salience: float = 0.5):
    b = Bullet(
        id=str(uuid.uuid4())[:8],
        context_id=str(context_id), section="general",
        content=content, bullet_type=BulletType.FACT,
        salience=salience, confidence=0.7,
        hit_count=hit, recall_count=recall, miss_count=miss,
    )
    await storage.add_bullet(str(context_id), b)
    return b


# ── 1. Default off ───────────────────────────────────────────────────────────

async def test_usage_stats_absent_when_disabled(storage, context_id):
    await _seed_bullet(storage, context_id, content="proven bullet",
                       hit=7, recall=10)
    mat = MaterializationEngine(storage, StubLLM())
    result = await mat.materialize(
        context_id=context_id, query="proven", token_budget=2000,
        target_model="claude", include_worked_examples=False,
    )
    assert "used 10×" not in result["rendered_text"]
    assert "success 7/10" not in result["rendered_text"]


# ── 2. Enabled — stats surface in render ─────────────────────────────────────

async def test_usage_stats_surface_in_render(storage, context_id):
    await _seed_bullet(storage, context_id, content="proven bullet",
                       hit=7, recall=10, salience=0.9)
    mat = MaterializationEngine(storage, StubLLM())
    result = await mat.materialize(
        context_id=context_id, query="proven", token_budget=2000,
        target_model="claude", include_worked_examples=False,
        include_usage_stats=True,
    )
    text = result["rendered_text"]
    assert "used 10×" in text or "used 10x" in text  # × char preserved
    assert "success 7/10" in text


# ── 3. Bullets with zero recall don't get a noise suffix ─────────────────────

async def test_usage_stats_skipped_when_no_signal(storage, context_id):
    await _seed_bullet(storage, context_id, content="fresh bullet",
                       hit=0, recall=0, salience=0.9)
    mat = MaterializationEngine(storage, StubLLM())
    result = await mat.materialize(
        context_id=context_id, query="fresh", token_budget=2000,
        target_model="claude", include_worked_examples=False,
        include_usage_stats=True,
    )
    text = result["rendered_text"]
    assert "fresh bullet" in text
    assert "used 0×" not in text  # no suffix because nothing to show


# ── 4. All renderers carry the suffix when given usage_stats ─────────────────

def test_all_renderers_emit_usage_suffix():
    from engram.core.models import ConceptNode, ConceptType
    concept = ConceptNode(type=ConceptType.FACT, content="proven bullet")
    usage = {"proven bullet": "(used 10×, success 7/10)"}

    claude_out = ClaudeRenderer().render(
        concepts=[concept], intent=None, token_budget=2000,
        usage_stats=usage,
    )
    gpt_out = GPTRenderer().render(
        concepts=[concept], intent=None, token_budget=2000,
        usage_stats=usage,
    )
    gen_out = GenericRenderer().render(
        concepts=[concept], intent=None, token_budget=2000,
        usage_stats=usage,
    )
    for out in (claude_out, gpt_out, gen_out):
        assert "(used 10×, success 7/10)" in out
