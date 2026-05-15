"""Tests for Phase 1: Mem-α inspired core memory slot.

Coverage:
1. Context model carries core_memory field; default is empty.
2. Storage round-trip persists and retrieves core_memory.
3. update_core_memory() updates the blob (and bumps updated_at).
4. Reflector parses core_memory_update from JSON; blank strings become None.
5. Reflector receives current core_memory in the prompt.
6. IngestionEngine.commit() emits an UPDATE_CORE_MEMORY delta op when the
   Reflector proposes a change, and DeltaEngine applies it via storage.
7. Long core_memory updates are capped to CORE_MEMORY_MAX_TOKENS.
8. MaterializationEngine always prepends core_memory in the rendered text,
   regardless of whether any bullets match the query.
9. Renderers (Claude / GPT / Generic) all surface core_memory.
10. Rolling back an UPDATE_CORE_MEMORY delta restores the prior value.
"""

from __future__ import annotations

import json
import uuid

import pytest

from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.ingestion import (
    CuratorEngine,
    IngestionEngine,
    ReflectorEngine,
    _cap_core_memory,
)
from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    CORE_MEMORY_MAX_TOKENS,
    Context,
    ContentType,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    IntentAnchor,
)
from engram.llm.adapter import LLMAdapter
from engram.renderers.claude import ClaudeRenderer
from engram.renderers.generic import GenericRenderer
from engram.renderers.gpt import GPTRenderer
from engram.storage.sqlite import SQLiteBackend


class CannedLLM(LLMAdapter):
    """LLM that returns a controllable canned response from the Reflector and
    a deterministic embedding for any text."""

    def __init__(self, response: dict | None = None):
        self.response = response or {
            "new_insights": [],
            "strategies_that_worked": [],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.6,
            "core_memory_update": None,
        }
        self.last_prompt: str | None = None

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None):
        self.last_prompt = prompt
        return json.dumps(self.response)

    async def embed(self, text):
        h = hash(text)
        return [(h >> i & 0xFF) / 255.0 for i in range(8)]


@pytest.fixture
async def storage(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "core_memory.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test core memory")
    ctx = Context(name="Core Memory Test", intent=intent)
    await storage.create_context(ctx)
    return ctx.id


# ── 1. Model default ─────────────────────────────────────────────────────────

def test_context_default_core_memory_is_empty():
    ctx = Context(name="x", intent=IntentAnchor(objective="y"))
    assert ctx.core_memory == ""


# ── 2. Storage round-trip ────────────────────────────────────────────────────

async def test_storage_roundtrip_core_memory(storage):
    intent = IntentAnchor(objective="rt")
    ctx = Context(name="rt", intent=intent, core_memory="user prefers brevity")
    await storage.create_context(ctx)
    loaded = await storage.get_context(ctx.id)
    assert loaded is not None
    assert loaded.core_memory == "user prefers brevity"


# ── 3. update_core_memory writes through ─────────────────────────────────────

async def test_update_core_memory_persists(storage, context_id):
    await storage.update_core_memory(str(context_id), "new running summary")
    loaded = await storage.get_context(context_id)
    assert loaded.core_memory == "new running summary"


# ── 4. Reflector JSON parsing ────────────────────────────────────────────────

async def test_reflector_parses_core_memory_update():
    llm = CannedLLM({
        "new_insights": [],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
        "core_memory_update": "User is building a PDF extractor.",
    })
    reflection = await ReflectorEngine(llm).reflect(raw_input="hi")
    assert reflection.core_memory_update == "User is building a PDF extractor."


async def test_reflector_treats_blank_core_memory_update_as_none():
    llm = CannedLLM({
        "new_insights": [],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
        "core_memory_update": "   ",
    })
    reflection = await ReflectorEngine(llm).reflect(raw_input="hi")
    assert reflection.core_memory_update is None


async def test_reflector_handles_missing_field():
    llm = CannedLLM({
        "new_insights": [],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })
    reflection = await ReflectorEngine(llm).reflect(raw_input="hi")
    assert reflection.core_memory_update is None


# ── 5. Reflector receives existing core_memory in prompt ─────────────────────

async def test_reflector_sees_existing_core_memory():
    llm = CannedLLM()
    await ReflectorEngine(llm).reflect(
        raw_input="hello",
        existing_core_memory="user is named Jia",
    )
    assert "user is named Jia" in (llm.last_prompt or "")
    assert "core memory" in (llm.last_prompt or "").lower()


# ── 6. Commit emits + applies UPDATE_CORE_MEMORY ─────────────────────────────

async def test_commit_applies_core_memory_update(storage, context_id):
    llm = CannedLLM({
        "new_insights": [],
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.8,
        "core_memory_update": "Project: build a PDF form extractor on AWS.",
    })
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    batch = await engine.commit(
        context_id=context_id,
        agent_id="t",
        content="we are building a PDF extractor",
        content_type=ContentType.CONVERSATION,
    )
    # core memory updated on context
    ctx = await storage.get_context(context_id)
    assert "PDF form extractor" in ctx.core_memory
    # batch contains an UPDATE_CORE_MEMORY op
    op_types = [op.op_type for op in batch.operations]
    assert DeltaOpType.UPDATE_CORE_MEMORY in op_types


async def test_commit_without_core_memory_update_leaves_blob_unchanged(storage, context_id):
    await storage.update_core_memory(str(context_id), "preexisting")
    llm = CannedLLM()  # default response has core_memory_update=None
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    await engine.commit(
        context_id=context_id,
        agent_id="t",
        content="raw text",
        content_type=ContentType.CONVERSATION,
    )
    ctx = await storage.get_context(context_id)
    assert ctx.core_memory == "preexisting"


# ── 7. Truncation cap ────────────────────────────────────────────────────────

def test_cap_core_memory_truncates_long_text():
    import tiktoken
    long = " ".join(["word"] * 5000)
    capped = _cap_core_memory(long)
    # Must be strictly shorter than input and within the token budget.
    assert len(capped) < len(long)
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(capped)) <= CORE_MEMORY_MAX_TOKENS


def test_cap_core_memory_passes_through_short_text():
    short = "user is named Jia and works on engram"
    assert _cap_core_memory(short) == short


# ── 8. Materialization prepends core_memory regardless of bullets ────────────

async def test_materialization_prepends_core_memory_even_with_no_bullets(
    storage, context_id,
):
    await storage.update_core_memory(str(context_id), "RUNTIME SUMMARY: alpha")
    engine = MaterializationEngine(storage, CannedLLM())
    result = await engine.materialize(
        context_id=context_id,
        query="anything",
        token_budget=2000,
        target_model="claude",
    )
    assert "RUNTIME SUMMARY: alpha" in result["rendered_text"]


async def test_materialization_prepends_core_memory_with_bullets(storage, context_id):
    # Seed a single bullet via direct add
    llm = CannedLLM()
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())
    from engram.core.models import BulletType
    await engine.add_bullet_directly(
        context_id=str(context_id),
        content="OCR is hard",
        bullet_type=BulletType.FACT,
    )
    await storage.update_core_memory(str(context_id), "RUNTIME SUMMARY: beta")

    mat = MaterializationEngine(storage, llm)
    result = await mat.materialize(
        context_id=context_id,
        query="OCR",
        token_budget=2000,
        target_model="gpt-4o",
    )
    assert "RUNTIME SUMMARY: beta" in result["rendered_text"]
    assert "Core Memory" in result["rendered_text"]


# ── 9. Renderers ─────────────────────────────────────────────────────────────

def test_claude_renderer_emits_core_memory():
    out = ClaudeRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        core_memory="hello world",
    )
    assert "<core_memory>" in out
    assert "hello world" in out


def test_gpt_renderer_emits_core_memory():
    out = GPTRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        core_memory="hello world",
    )
    assert "## Core Memory" in out
    assert "hello world" in out


def test_generic_renderer_emits_core_memory():
    out = GenericRenderer().render(
        concepts=[], intent=None, token_budget=2000,
        core_memory="hello world",
    )
    assert "CORE MEMORY: hello world" in out


# ── 10. Rollback restores prior core memory ──────────────────────────────────

async def test_rollback_update_core_memory(storage, context_id):
    # Apply a batch that updates core memory
    ctx_id_str = str(context_id)
    await storage.update_core_memory(ctx_id_str, "original")
    op = DeltaOperation(
        op_type=DeltaOpType.UPDATE_CORE_MEMORY,
        target_id=ctx_id_str,
        content="changed",
        source=DeltaSource.REFLECTOR,
    )
    batch = DeltaBatch(context_id=ctx_id_str, operations=[op])
    delta_engine = DeltaEngine(storage)
    applied = await delta_engine.apply_batch(batch)
    ctx = await storage.get_context(context_id)
    assert ctx.core_memory == "changed"

    # Roll back
    ok = await delta_engine.rollback_batch(applied.id)
    assert ok is True
    ctx = await storage.get_context(context_id)
    assert ctx.core_memory == "original"
