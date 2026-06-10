"""End-to-end integration test for v0.5 changes (Mem-α + DC mechanisms).

Exercises the composition of all new features in one flow:
  - Core memory slot is written by the Reflector and prepended at render.
  - Worked-example retrieval surfaces a prior raw input at recall time.
  - MMR diversity re-ranking is active in the materialization path.
  - Reconsolidation runs as a delta batch (audit-clean, rollback-able).
  - Usage stats surface after at least one recall + outcome cycle.
  - Validity gate (when enabled) drops malformed candidates.
  - Episodic bullet type round-trips end-to-end.

The point is composition: each individual mechanism has its own dedicated
test file. If any of them silently regress while plumbing together, this
test should catch the contradiction.
"""

from __future__ import annotations

import itertools
import json
import zlib
import math
import uuid

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import IngestionEngine
from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    BulletType,
    ContentType,
    Context,
    DeltaOpType,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class ScriptedLLM(LLMAdapter):
    """LLM whose `complete()` returns scripted responses in order, and whose
    `embed()` returns a deterministic word-bucket embedding."""

    def __init__(self, complete_responses: list[str]):
        self.responses = iter(complete_responses)
        self.calls: list[tuple[str, str | None]] = []

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        self.calls.append((prompt, system))
        try:
            return next(self.responses)
        except StopIteration as e:
            raise RuntimeError(
                f"ScriptedLLM exhausted at call #{len(self.calls)}; "
                f"last prompt: {prompt[:120]}"
            ) from e

    async def embed(self, text):
        # Use a stable hash (not builtin hash(), which is randomized per process
        # via PYTHONHASHSEED) so worked-example cosine similarity is reproducible
        # across runs — otherwise the 0.5-threshold assertion below is flaky.
        vec = [0.0] * 64
        for w in text.lower().split():
            bucket = zlib.crc32(w.encode("utf-8")) % 64
            vec[bucket] += 1.0
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


def _reflection(insights, core_memory_update=None, confidence=0.7):
    return json.dumps({
        "new_insights": insights,
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": confidence,
        "core_memory_update": core_memory_update,
    })


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "v05.db"))
    await b.initialize()
    yield b
    await b.close()


async def test_end_to_end_v05_composition(storage):
    # Create a context.
    ctx = Context(
        name="Integration",
        intent=IntentAnchor(objective="Build a PDF extractor for Jia"),
    )
    await storage.create_context(ctx)

    # ─── Commit #1 ───
    # Reflector returns:
    #   - 2 facts (one near-duplicate of the other, to exercise MMR later)
    #   - 1 episodic
    #   - a core_memory_update
    insights_1 = [
        {"content": "PaddleOCR is fast on PDFs", "insight_type": "fact",
         "suggested_section": "ocr", "evidence": "", "novelty": 0.7},
        {"content": "PaddleOCR speeds PDF processing", "insight_type": "fact",
         "suggested_section": "ocr", "evidence": "", "novelty": 0.7},
        {"content": "At 14:30 Jia approved the PaddleOCR decision",
         "insight_type": "episodic", "suggested_section": "events",
         "evidence": "", "novelty": 0.8},
    ]
    llm = ScriptedLLM([
        _reflection(insights_1, core_memory_update=(
            "Project: build a PDF form extractor for Jia on AWS. "
            "Stack decisions in progress; OCR engine chosen 2026-05-14."
        )),
    ])
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    batch1 = await engine.commit(
        context_id=ctx.id, agent_id="claude-agent",
        content="benchmarked PaddleOCR vs Textract for PDF extraction; "
                "Jia approved going with PaddleOCR at 14:30",
        content_type=ContentType.CONVERSATION,
    )

    # Core memory was updated via an UPDATE_CORE_MEMORY delta op.
    op_types_1 = [op.op_type for op in batch1.operations]
    assert DeltaOpType.UPDATE_CORE_MEMORY in op_types_1
    refreshed = await storage.get_context(ctx.id)
    assert "PDF form extractor for Jia" in refreshed.core_memory

    # The activity carries a raw_input_embedding for worked-example retrieval.
    activities = await storage.list_activities(ctx.id, limit=5)
    assert any(a.raw_input_embedding for a in activities)

    # ─── Materialize: should include core memory + a worked example ───
    mat_engine = MaterializationEngine(storage, llm)
    # The query shares words with commit #1's raw input so worked-example
    # retrieval will find it.
    result_1 = await mat_engine.materialize(
        context_id=ctx.id,
        query="benchmarked PaddleOCR PDF extraction",
        token_budget=3000,
        target_model="claude",
        worked_example_threshold=0.5,
        worked_example_limit=1,
        mmr_lambda=0.5,  # diversity ranking active
    )
    text = result_1["rendered_text"]
    assert "PDF form extractor for Jia" in text  # core memory
    assert "<worked_examples>" in text            # DC worked-example block
    assert "PaddleOCR" in text                    # bullets

    # ─── Commit #2 with materialization_id + outcome=success ───
    # Reflector emits one more insight; we don't expect a core memory update.
    insights_2 = [
        {"content": "Confirmed PaddleOCR works on multi-column layouts",
         "insight_type": "fact", "suggested_section": "ocr",
         "evidence": "", "novelty": 0.6},
    ]
    llm.responses = iter([_reflection(insights_2)])
    await engine.commit(
        context_id=ctx.id, agent_id="gpt-agent",
        content="Re-tested PaddleOCR — multi-column layouts work.",
        materialization_id=result_1["materialization_id"],
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
    )

    # A reconsolidation batch should be present.
    all_batches = await storage.list_delta_batches(str(ctx.id), limit=20)
    rc_batches = [b for b in all_batches if b.trigger == "reconsolidation"]
    assert len(rc_batches) == 1
    assert any(
        op.op_type == DeltaOpType.RECONSOLIDATE_BULLET
        for op in rc_batches[0].operations
    )

    # The bullets included in result_1's materialization should now have
    # recall_count >= 1 (since the reconsolidation ran).
    recalled_ids = result_1["bullets_included"]
    assert recalled_ids, "expected at least one bullet in the first materialization"
    recalled_bullets = await storage.get_bullets_by_ids(str(ctx.id), recalled_ids)
    assert any(b.recall_count >= 1 for b in recalled_bullets)
    assert any(b.hit_count >= 1 for b in recalled_bullets)

    # ─── Materialize again with usage stats turned on ───
    llm.responses = iter([])  # no LLM calls expected for this read-only path
    result_2 = await mat_engine.materialize(
        context_id=ctx.id,
        query="PaddleOCR PDF",
        token_budget=3000,
        target_model="claude",
        worked_example_threshold=0.5,
        include_usage_stats=True,
        mmr_lambda=0.5,
    )
    assert "used " in result_2["rendered_text"]
    assert "success " in result_2["rendered_text"]

    # ─── Episodic bullet survived as its own type ───
    bullets = await storage.list_bullets(str(ctx.id))
    episodic = [b for b in bullets if (
        b.bullet_type.value if hasattr(b.bullet_type, "value") else str(b.bullet_type)
    ) == "episodic"]
    assert episodic, "episodic bullet should round-trip from the commit"
    assert "At 14:30" in episodic[0].content


async def test_validity_gate_in_full_pipeline(storage):
    """When the validity gate is enabled, trivial candidates are dropped
    before reaching the graph, but valid ones still land."""
    ctx = Context(name="VG-Int", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)

    insights = [
        {"content": "Postgres pgvector index uses HNSW by default",
         "insight_type": "fact", "suggested_section": "db",
         "evidence": "", "novelty": 0.6},
        {"content": "ok thanks",
         "insight_type": "fact", "suggested_section": "general",
         "evidence": "", "novelty": 0.1},
    ]
    verdicts = json.dumps({"verdicts": [
        {"idx": 0, "keep": True},
        {"idx": 1, "keep": False, "reason": "conversational filler"},
    ]})
    llm = ScriptedLLM([_reflection(insights), verdicts])
    cfg = IngestionConfig(enable_validity_gate=True)
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    batch = await engine.commit(
        context_id=ctx.id, agent_id="t",
        content="some raw input",
        content_type=ContentType.CONVERSATION,
    )
    kept = [op.content for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert "Postgres pgvector index uses HNSW by default" in kept
    assert "ok thanks" not in kept
