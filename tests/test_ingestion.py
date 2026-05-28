"""Tests for the ingestion engine (v0.2 — Reflector → Curator → Delta pipeline)."""

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from engram.core.ingestion import CuratorEngine, IngestionEngine, ReflectorEngine
from engram.core.models import (
    Bullet,
    BulletType,
    ConceptNode,
    ConceptType,
    ContentType,
    Context,
    DeltaBatch,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
    MaterializationRecord,
    Reflection,
    ReflectionInsight,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter that returns canned responses for Reflector."""

    def __init__(self):
        self.complete_response = json.dumps({
            "new_insights": [
                {
                    "content": "PaddleOCR is 40% faster than Textract",
                    "insight_type": "fact",
                    "suggested_section": "ocr_tools",
                    "evidence": "Performance benchmark data",
                    "novelty": 0.8,
                },
                {
                    "content": "Use PaddleOCR for field detection",
                    "insight_type": "decision",
                    "suggested_section": "architecture",
                    "evidence": "Based on performance comparison",
                    "novelty": 0.9,
                },
            ],
            "strategies_that_worked": ["Benchmark before choosing"],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.85,
        })

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        return self.complete_response

    async def embed(self, text):
        # Return a deterministic embedding based on text hash
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
    intent = IntentAnchor(objective="Test ingestion")
    ctx = Context(name="Ingestion Test", intent=intent)
    await storage.create_context(ctx)
    return ctx.id


class TestReflectorEngine:
    async def test_reflect_produces_insights(self, mock_llm):
        reflector = ReflectorEngine(mock_llm)
        reflection = await reflector.reflect(
            raw_input="PaddleOCR is 40% faster than Textract. We decided to use PaddleOCR.",
        )
        assert len(reflection.new_insights) == 2
        assert reflection.confidence == pytest.approx(0.85)
        assert reflection.new_insights[0].content == "PaddleOCR is 40% faster than Textract"
        assert reflection.new_insights[1].insight_type == "decision"

    async def test_reflect_with_feedback(self, mock_llm):
        reflector = ReflectorEngine(mock_llm)
        feedback = ExecutionFeedback(
            outcome=FeedbackOutcome.SUCCESS,
            metrics={"accuracy": 0.95},
        )
        reflection = await reflector.reflect(
            raw_input="Task completed successfully.",
            feedback=feedback,
        )
        assert len(reflection.new_insights) >= 1

    async def test_reflect_with_strategies(self, mock_llm):
        reflector = ReflectorEngine(mock_llm)
        reflection = await reflector.reflect(raw_input="Benchmark test")
        assert "Benchmark before choosing" in reflection.strategies_that_worked


class TestCuratorEngine:
    async def test_curate_produces_delta_operations(self, storage, mock_llm, context_id):
        curator = CuratorEngine(storage, mock_llm)
        reflection = Reflection(
            new_insights=[
                ReflectionInsight(
                    content="PaddleOCR is fast",
                    insight_type="fact",
                    suggested_section="ocr",
                    novelty=0.8,
                ),
            ],
            strategies_that_worked=["Benchmark first"],
            failure_modes=["Timeout on large files"],
            prediction_errors=["Expected 1s, got 5s"],
        )
        batch = await curator.curate(str(context_id), reflection, agent_id="test")

        assert len(batch.operations) >= 3  # insight + strategy + failure + prediction error
        # Check that the different types are present
        op_types = [op.bullet_type for op in batch.operations]
        assert "fact" in op_types
        assert "strategy" in op_types
        assert "warning" in op_types
        assert "exception" in op_types

    async def test_curate_dedup_exact_match(self, storage, mock_llm, context_id):
        ctx_id_str = str(context_id)
        # Pre-insert a bullet that matches
        existing = Bullet(content="PaddleOCR is fast", section="ocr", bullet_type=BulletType.FACT)
        await storage.add_bullet(ctx_id_str, existing)

        curator = CuratorEngine(storage, mock_llm)
        reflection = Reflection(
            new_insights=[
                ReflectionInsight(content="PaddleOCR is fast", insight_type="fact"),
            ],
        )
        batch = await curator.curate(ctx_id_str, reflection, agent_id="test")

        # The exact duplicate should be skipped
        add_ops = [op for op in batch.operations if op.content == "PaddleOCR is fast"]
        assert len(add_ops) == 0


class TestIngestionEngine:
    async def test_commit_produces_delta_batch(self, storage, mock_llm, context_id):
        engine = IngestionEngine(storage, mock_llm)
        batch = await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="PaddleOCR is 40% faster than Textract. We decided to use PaddleOCR.",
        )

        assert isinstance(batch, DeltaBatch)
        assert batch.bullets_added >= 1

        # Check that bullets were actually stored
        bullets = await storage.list_bullets(str(context_id))
        assert len(bullets) >= 1

    async def test_commit_creates_activity(self, storage, mock_llm, context_id):
        engine = IngestionEngine(storage, mock_llm)
        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Test content",
        )
        activities = await storage.list_activities(context_id)
        assert len(activities) == 1
        assert activities[0].agent_id == "test-agent"
        assert activities[0].delta_batch_id is not None

    async def test_commit_saves_delta_batch(self, storage, mock_llm, context_id):
        engine = IngestionEngine(storage, mock_llm)
        batch = await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Test content",
        )

        loaded_batch = await storage.get_delta_batch(batch.id)
        assert loaded_batch is not None

    async def test_add_bullet_directly(self, storage, mock_llm, context_id):
        engine = IngestionEngine(storage, mock_llm)
        bullet, batch = await engine.add_bullet_directly(
            context_id=str(context_id),
            content="Direct bullet addition",
            section="testing",
            bullet_type=BulletType.FACT,
            salience=0.7,
        )
        assert bullet.content == "Direct bullet addition"
        assert batch.bullets_added == 1

    async def test_record_decision(self, storage, mock_llm, context_id):
        engine = IngestionEngine(storage, mock_llm)
        decision_id, batch = await engine.record_decision(
            context_id=context_id,
            decision="Use PaddleOCR",
            rationale="Faster and cheaper",
            alternatives=["Textract", "Mistral Vision"],
            agent_id="architect",
        )
        assert decision_id is not None
        assert batch.bullets_added >= 3  # decision + 2 alternatives

        bullets = await storage.list_bullets(str(context_id))
        assert len(bullets) >= 3

        activities = await storage.list_activities(context_id)
        assert len(activities) == 1
        assert activities[0].agent_id == "architect"

    async def test_reconsolidation_success(self, storage, mock_llm, context_id):
        """Test that reconsolidation updates bullet stats on success."""
        ctx_id_str = str(context_id)

        # Set up bullets and a materialization record
        b1 = Bullet(content="Fact 1", salience=0.5, hit_count=0, recall_count=0)
        b2 = Bullet(content="Fact 2", salience=0.5, hit_count=0, recall_count=0)
        await storage.add_bullet(ctx_id_str, b1)
        await storage.add_bullet(ctx_id_str, b2)

        mat_record = MaterializationRecord(
            context_id=ctx_id_str,
            bullets_included=[b1.id, b2.id],
            token_count=200,
        )
        await storage.save_materialization(mat_record)

        # Commit with feedback referencing the materialization
        engine = IngestionEngine(storage, mock_llm)
        await engine.commit(
            context_id=context_id,
            agent_id="test",
            content="Task completed successfully.",
            feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
            materialization_id=mat_record.id,
        )

        # Check that bullets were updated
        b1_updated = await storage.get_bullet(b1.id)
        assert b1_updated.hit_count == 1
        assert b1_updated.recall_count == 1
        assert b1_updated.salience > 0.5  # Should have been boosted

    async def test_reconsolidation_failure(self, storage, mock_llm, context_id):
        """Test that reconsolidation penalizes bullets on failure."""
        ctx_id_str = str(context_id)

        b1 = Bullet(content="Bad advice", salience=0.5, hit_count=0, recall_count=0)
        await storage.add_bullet(ctx_id_str, b1)

        mat_record = MaterializationRecord(
            context_id=ctx_id_str,
            bullets_included=[b1.id],
            token_count=100,
        )
        await storage.save_materialization(mat_record)

        engine = IngestionEngine(storage, mock_llm)
        await engine.commit(
            context_id=context_id,
            agent_id="test",
            content="Task failed because of bad advice.",
            feedback=ExecutionFeedback(outcome=FeedbackOutcome.FAILURE),
            materialization_id=mat_record.id,
        )

        b1_updated = await storage.get_bullet(b1.id)
        assert b1_updated.miss_count == 1
        assert b1_updated.salience < 0.5  # Should have been penalized

    async def test_legacy_add_concept_directly(self, storage, mock_llm, context_id):
        """Backward compat: add_concept_directly still works."""
        engine = IngestionEngine(storage, mock_llm)
        concept = ConceptNode(
            type=ConceptType.ENTITY,
            content="Client X Corp",
            salience=0.6,
        )
        added, edge_ids = await engine.add_concept_directly(context_id, concept)
        assert added.id == concept.id
        assert added.embedding is not None
