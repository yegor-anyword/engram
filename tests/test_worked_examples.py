"""Tests for Phase 2: DC-inspired worked-example injection at materialization.

Coverage:
1. IngestionEngine.commit embeds the raw_input and persists it on the activity.
2. storage.find_similar_activities returns nearest activities above threshold.
3. find_similar_activities applies exclude_hash to skip the current commit.
4. MaterializationEngine fetches worked examples and the renderer includes them.
5. include_worked_examples=False suppresses the lookup entirely.
6. When no prior activities clear the threshold, no worked-example block is rendered.
7. Renderers (Claude/GPT/Generic) all emit the worked-example block when provided.
"""

from __future__ import annotations

import json

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import IngestionEngine, ReflectorEngine
from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    Activity,
    BulletType,
    ContentType,
    Context,
    IntentAnchor,
)
from engram.llm.adapter import LLMAdapter
from engram.renderers.claude import ClaudeRenderer
from engram.renderers.generic import GenericRenderer
from engram.renderers.gpt import GPTRenderer
from engram.storage.sqlite import SQLiteBackend


class ScriptedLLM(LLMAdapter):
    """LLM where embed() returns embeddings derived from a keyword map so we
    can deterministically simulate "these two texts are semantically close"."""

    def __init__(self, response: dict | None = None, embed_map: dict[str, list[float]] | None = None):
        self.response = response or {
            "new_insights": [],
            "strategies_that_worked": [],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.6,
            "core_memory_update": None,
        }
        # Fallback: bucket by lowercase first word so similar prompts get similar vectors.
        self.embed_map = embed_map or {}

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        return json.dumps(self.response)

    async def embed(self, text):
        # Token-bucket embedding: each word maps to one of 64 dimensions.
        # At 64 dims, disjoint word sets are nearly orthogonal so the cosine
        # threshold behaves like it would in a real embedding model.
        vec = [0.0] * 64
        for word in text.lower().split():
            vec[hash(word) % 64] += 1.0
        import math
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@pytest.fixture
async def storage(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "wx.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test worked examples")
    ctx = Context(name="WX", intent=intent)
    await storage.create_context(ctx)
    return ctx.id


# ── 1. Embedding is persisted on activity ────────────────────────────────────

async def test_commit_persists_raw_input_embedding(storage, context_id):
    llm = ScriptedLLM(response={
        "new_insights": [{
            "content": "PaddleOCR is fast",
            "insight_type": "fact",
            "suggested_section": "ocr",
            "evidence": "",
            "novelty": 0.5,
        }],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id,
        agent_id="t",
        content="PaddleOCR benchmark results — it is fast",
        content_type=ContentType.CONVERSATION,
    )
    activities = await storage.list_activities(context_id, limit=10)
    assert any(a.raw_input_embedding is not None for a in activities)


# ── 2. find_similar_activities returns hits above threshold ──────────────────

async def test_find_similar_activities_above_threshold(storage, context_id):
    llm = ScriptedLLM()
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    # Commit a few raw inputs.
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="benchmark PaddleOCR results — fast and accurate",
    )
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="dinner ideas for tomorrow night",  # unrelated
    )
    # Query embedding shares words with the first.
    query_emb = await llm.embed("benchmark PaddleOCR results")
    hits = await storage.find_similar_activities(
        str(context_id), query_emb, limit=3, threshold=0.5,
    )
    assert len(hits) >= 1
    top_activity, top_sim = hits[0]
    assert "PaddleOCR" in (top_activity.raw_input or "")
    assert top_sim >= 0.5


# ── 3. exclude_hash filters out a specific activity ──────────────────────────

async def test_find_similar_activities_exclude_hash(storage, context_id):
    llm = ScriptedLLM()
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="alpha beta gamma delta",
    )
    activities = await storage.list_activities(context_id, limit=10)
    target_hash = activities[0].raw_input_hash

    query_emb = await llm.embed("alpha beta gamma delta")
    hits_all = await storage.find_similar_activities(
        str(context_id), query_emb, limit=5, threshold=0.5,
    )
    hits_excluded = await storage.find_similar_activities(
        str(context_id), query_emb, limit=5, threshold=0.5,
        exclude_hash=target_hash,
    )
    assert len(hits_all) > len(hits_excluded)
    assert all(a.raw_input_hash != target_hash for a, _ in hits_excluded)


# ── 4. Materialization attaches worked examples ──────────────────────────────

async def test_materialize_includes_worked_examples(storage, context_id):
    llm = ScriptedLLM(response={
        "new_insights": [{
            "content": "PaddleOCR works on PDFs",
            "insight_type": "fact",
            "suggested_section": "ocr",
            "evidence": "",
            "novelty": 0.6,
        }],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="benchmark PaddleOCR PDFs results",
    )

    mat = MaterializationEngine(storage, llm)
    result = await mat.materialize(
        context_id=context_id,
        query="benchmark PaddleOCR PDFs",
        token_budget=2000,
        target_model="claude",
        worked_example_threshold=0.5,
    )
    assert "<worked_examples>" in result["rendered_text"]
    assert "benchmark PaddleOCR PDFs results" in result["rendered_text"]


# ── 5. include_worked_examples=False skips lookup ────────────────────────────

async def test_materialize_skips_worked_examples_when_disabled(storage, context_id):
    llm = ScriptedLLM(response={
        "new_insights": [{
            "content": "PaddleOCR fact",
            "insight_type": "fact",
            "suggested_section": "ocr",
            "evidence": "",
            "novelty": 0.5,
        }],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="benchmark PaddleOCR",
    )

    mat = MaterializationEngine(storage, llm)
    result = await mat.materialize(
        context_id=context_id,
        query="benchmark PaddleOCR",
        token_budget=2000,
        target_model="claude",
        include_worked_examples=False,
    )
    assert "<worked_examples>" not in result["rendered_text"]


# ── 6. Threshold gates inclusion ─────────────────────────────────────────────

async def test_high_threshold_excludes_unrelated_activities(storage, context_id):
    llm = ScriptedLLM(response={
        "new_insights": [{
            "content": "Just some fact",
            "insight_type": "fact",
            "suggested_section": "x",
            "evidence": "",
            "novelty": 0.5,
        }],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id, agent_id="a",
        content="grocery list: bread, eggs, milk",
    )

    mat = MaterializationEngine(storage, llm)
    result = await mat.materialize(
        context_id=context_id,
        query="quantum chromodynamics renormalization",  # share no words
        token_budget=2000,
        target_model="claude",
        worked_example_threshold=0.5,
    )
    assert "<worked_examples>" not in result["rendered_text"]


# ── 7. Renderers ─────────────────────────────────────────────────────────────

def test_claude_renderer_emits_worked_examples():
    out = ClaudeRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        worked_examples=[{"input": "raw question", "output": "bullet a; bullet b"}],
    )
    assert "<worked_examples>" in out
    assert "raw question" in out
    assert "bullet a; bullet b" in out


def test_gpt_renderer_emits_worked_examples():
    out = GPTRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        worked_examples=[{"input": "raw question", "output": "bullet a; bullet b"}],
    )
    assert "## Worked Examples" in out
    assert "raw question" in out
    assert "bullet a; bullet b" in out


def test_generic_renderer_emits_worked_examples():
    out = GenericRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        worked_examples=[{"input": "raw question", "output": "bullet a; bullet b"}],
    )
    assert "WORKED EXAMPLES" in out
    assert "raw question" in out
