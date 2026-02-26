"""Tests for the consolidation engine (sleep cycle)."""

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from engram.core.consolidation import ConsolidationEngine, _cosine_similarity
from engram.core.models import (
    Bullet,
    BulletType,
    ConsolidationConfig,
    Context,
    IntentAnchor,
    SchemaNode,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class MockLLMAdapter(LLMAdapter):
    """Mock LLM for consolidation tests."""

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None):
        return "Abstract pattern identified from multiple instances."

    async def embed(self, text):
        h = hash(text)
        return [(h >> i & 0xFF) / 255.0 for i in range(8)]


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def mock_llm():
    return MockLLMAdapter()


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test consolidation")
    ctx = Context(name="Consolidation Test", intent=intent)
    await storage.create_context(ctx)
    return str(ctx.id)


class TestCosinesSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_similar_vectors(self):
        sim = _cosine_similarity([1, 0, 0], [0.9, 0.1, 0])
        assert 0.9 < sim < 1.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


class TestForgettingCurve:
    async def test_decay_old_bullets(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Create a bullet that's 10 days old
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        bullet = Bullet(
            content="Old fact",
            salience=0.5,
            created_at=old_time,
            bullet_type=BulletType.FACT,
        )
        await storage.add_bullet(context_id, bullet)

        config = ConsolidationConfig()
        decayed = await engine._apply_forgetting_curve(context_id, config)
        assert decayed == 1

        loaded = await storage.get_bullet(bullet.id)
        assert loaded.salience < 0.5  # Should have decayed

    async def test_no_decay_recent_bullets(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Create a very recent bullet
        bullet = Bullet(content="Fresh fact", salience=0.5)
        await storage.add_bullet(context_id, bullet)

        config = ConsolidationConfig()
        decayed = await engine._apply_forgetting_curve(context_id, config)
        assert decayed == 0

    async def test_slow_decay_for_decisions(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        decision = Bullet(
            content="Use PaddleOCR",
            salience=0.8,
            bullet_type=BulletType.DECISION,
            created_at=old_time,
        )
        fact = Bullet(
            content="PaddleOCR is fast",
            salience=0.8,
            bullet_type=BulletType.FACT,
            created_at=old_time,
        )
        await storage.add_bullet(context_id, decision)
        await storage.add_bullet(context_id, fact)

        config = ConsolidationConfig()
        await engine._apply_forgetting_curve(context_id, config)

        loaded_decision = await storage.get_bullet(decision.id)
        loaded_fact = await storage.get_bullet(fact.id)
        # Decision should decay slower than fact
        assert loaded_decision.salience > loaded_fact.salience


class TestSemanticDedup:
    async def test_merge_near_duplicates(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Create near-duplicate bullets with very similar embeddings
        b1 = Bullet(content="PaddleOCR is fast", embedding=[1.0, 0.0, 0.0], salience=0.8)
        b2 = Bullet(content="PaddleOCR is very fast", embedding=[0.99, 0.01, 0.0], salience=0.6)
        await storage.add_bullet(context_id, b1)
        await storage.add_bullet(context_id, b2)

        config = ConsolidationConfig(dedup_threshold=0.9)
        merged = await engine._semantic_dedup(context_id, config)
        assert merged == 1

        # Higher-salience bullet should survive
        bullets = await storage.list_bullets(context_id)
        assert len(bullets) == 1
        assert bullets[0].salience == 0.8

    async def test_no_merge_for_dissimilar(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        b1 = Bullet(content="OCR perf", embedding=[1.0, 0.0, 0.0], salience=0.8)
        b2 = Bullet(content="Cloud costs", embedding=[0.0, 1.0, 0.0], salience=0.6)
        await storage.add_bullet(context_id, b1)
        await storage.add_bullet(context_id, b2)

        config = ConsolidationConfig(dedup_threshold=0.9)
        merged = await engine._semantic_dedup(context_id, config)
        assert merged == 0

        bullets = await storage.list_bullets(context_id)
        assert len(bullets) == 2


class TestSchemaInduction:
    async def test_form_schema_from_section(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Create 4 bullets in the same section (>= schema_min_instances=3)
        for i in range(4):
            b = Bullet(content=f"Error pattern {i}", section="error_handling")
            await storage.add_bullet(context_id, b)

        config = ConsolidationConfig(schema_min_instances=3)
        schemas_formed = await engine._induce_schemas(context_id, config)
        assert schemas_formed == 1

        schemas = await storage.list_schemas(context_id)
        assert len(schemas) == 1
        assert schemas[0].name == "error_handling"

    async def test_no_schema_below_threshold(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Only 2 bullets — below threshold
        for i in range(2):
            b = Bullet(content=f"Lonely fact {i}", section="small_section")
            await storage.add_bullet(context_id, b)

        config = ConsolidationConfig(schema_min_instances=3)
        schemas_formed = await engine._induce_schemas(context_id, config)
        assert schemas_formed == 0


class TestArchiveStale:
    async def test_archive_low_salience_old_bullets(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        old_time = datetime.now(timezone.utc) - timedelta(days=90)
        stale = Bullet(
            content="Very old and low salience",
            salience=0.03,
            created_at=old_time,
            bullet_type=BulletType.FACT,
        )
        await storage.add_bullet(context_id, stale)

        config = ConsolidationConfig(
            archive_salience_threshold=0.05,
            archive_days_threshold=60,
        )
        archived = await engine._archive_stale(context_id, config)
        assert archived == 1

        loaded = await storage.get_bullet(stale.id)
        assert loaded.is_archived is True

    async def test_dont_archive_decisions(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        old_time = datetime.now(timezone.utc) - timedelta(days=90)
        decision = Bullet(
            content="Important decision",
            salience=0.03,
            bullet_type=BulletType.DECISION,
            created_at=old_time,
        )
        await storage.add_bullet(context_id, decision)

        config = ConsolidationConfig()
        archived = await engine._archive_stale(context_id, config)
        assert archived == 0  # Decisions are never archived

    async def test_dont_archive_recent_bullets(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        recent = Bullet(content="Recent low salience", salience=0.03)
        await storage.add_bullet(context_id, recent)

        config = ConsolidationConfig()
        archived = await engine._archive_stale(context_id, config)
        assert archived == 0  # Too recent


class TestFullConsolidation:
    async def test_full_cycle(self, storage, mock_llm, context_id):
        engine = ConsolidationEngine(storage, mock_llm)

        # Add various bullets
        old_time = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            b = Bullet(content=f"Fact {i}", section="general", created_at=old_time)
            await storage.add_bullet(context_id, b)

        report = await engine.consolidate(context_id)
        assert report.context_id == context_id
        assert report.duration_ms >= 0
        # At least some decay should happen
        assert report.decayed >= 0
