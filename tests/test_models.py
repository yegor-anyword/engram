"""Tests for Pydantic data models (v0.2 — brain-inspired architecture)."""

import uuid
from datetime import datetime

import pytest

from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    BulletType,
    ConceptEdge,
    ConceptNode,
    ConceptType,
    ConsolidationConfig,
    Context,
    CreateContextRequest,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    EdgeType,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
    IntentInput,
    IntentStatus,
    MaterializeRequest,
    Reflection,
    ReflectionInsight,
    SchemaNode,
    SourceType,
)


class TestBullet:
    def test_create_minimal(self):
        bullet = Bullet(content="PaddleOCR is fast")
        assert bullet.content == "PaddleOCR is fast"
        assert bullet.bullet_type == BulletType.FACT
        assert bullet.salience == 0.5
        assert bullet.hit_count == 0
        assert bullet.miss_count == 0
        assert bullet.recall_count == 0
        assert bullet.is_active is True
        assert bullet.is_archived is False

    def test_create_full(self):
        bullet = Bullet(
            content="Use retry with backoff for flaky APIs",
            section="api_patterns",
            bullet_type=BulletType.STRATEGY,
            source_type=SourceType.REFLECTION,
            salience=0.8,
            confidence=0.9,
            hit_count=5,
            miss_count=1,
            recall_count=6,
        )
        assert bullet.section == "api_patterns"
        assert bullet.bullet_type == BulletType.STRATEGY
        assert bullet.hit_count == 5

    def test_effective_salience_no_history(self):
        bullet = Bullet(content="New fact", salience=0.5, recall_count=0)
        assert bullet.effective_salience == 0.5

    def test_effective_salience_with_good_history(self):
        bullet = Bullet(
            content="Useful fact",
            salience=0.8,
            hit_count=8,
            miss_count=2,
            recall_count=10,
        )
        # hit_rate = 0.8, effective = 0.8 * (0.5 + 0.8) = 1.04 → capped at 1.0
        assert bullet.effective_salience == 1.0

    def test_effective_salience_with_poor_history(self):
        bullet = Bullet(
            content="Mostly wrong fact",
            salience=0.8,
            hit_count=1,
            miss_count=9,
            recall_count=10,
        )
        # hit_rate = 0.1, effective = 0.8 * (0.5 + 0.1) = 0.48
        assert bullet.effective_salience == pytest.approx(0.48, abs=0.01)

    def test_hit_rate(self):
        bullet = Bullet(content="test", hit_count=3, recall_count=6)
        assert bullet.hit_rate == 0.5

    def test_hit_rate_zero_recalls(self):
        bullet = Bullet(content="test")
        assert bullet.hit_rate == 0.0

    def test_serialization(self):
        bullet = Bullet(content="Serialize me", section="test")
        data = bullet.model_dump(mode="json")
        assert data["content"] == "Serialize me"
        restored = Bullet.model_validate(data)
        assert restored.id == bullet.id

    def test_salience_bounds(self):
        with pytest.raises(Exception):
            Bullet(content="test", salience=1.5)
        with pytest.raises(Exception):
            Bullet(content="test", salience=-0.1)


class TestSchemaNode:
    def test_create(self):
        schema = SchemaNode(
            name="api_error_handling",
            description="Pattern for handling API errors with retries",
            instance_count=5,
            confidence=0.7,
        )
        assert schema.name == "api_error_handling"
        assert schema.instance_count == 5

    def test_serialization(self):
        schema = SchemaNode(name="test_schema", description="test")
        data = schema.model_dump(mode="json")
        restored = SchemaNode.model_validate(data)
        assert restored.name == schema.name


class TestDeltaOperation:
    def test_create(self):
        op = DeltaOperation(
            op_type=DeltaOpType.ADD_BULLET,
            target_id="abc123",
            section="general",
            content="New fact",
            bullet_type="fact",
            source=DeltaSource.CURATOR,
        )
        assert op.op_type == DeltaOpType.ADD_BULLET
        assert op.target_id == "abc123"

    def test_delta_batch(self):
        batch = DeltaBatch(
            context_id="ctx-1",
            operations=[
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="Fact 1"),
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="Fact 2"),
            ],
        )
        assert len(batch.operations) == 2
        assert batch.context_id == "ctx-1"


class TestExecutionFeedback:
    def test_create(self):
        feedback = ExecutionFeedback(
            outcome=FeedbackOutcome.SUCCESS,
            metrics={"accuracy": 0.95},
            user_accepted=True,
        )
        assert feedback.outcome == FeedbackOutcome.SUCCESS
        assert feedback.metrics["accuracy"] == 0.95


class TestReflection:
    def test_create(self):
        reflection = Reflection(
            new_insights=[
                ReflectionInsight(
                    content="API retries improve reliability",
                    insight_type="strategy",
                    suggested_section="api_patterns",
                )
            ],
            strategies_that_worked=["Use exponential backoff"],
            failure_modes=["Timeout on large payloads"],
            prediction_errors=["Expected 1s response, got 5s"],
        )
        assert len(reflection.new_insights) == 1
        assert len(reflection.failure_modes) == 1


class TestConsolidationConfig:
    def test_defaults(self):
        config = ConsolidationConfig()
        assert config.fast_decay_rate == 0.97
        assert config.slow_decay_rate == 0.995
        assert config.dedup_threshold == 0.92
        assert config.schema_min_instances == 3

    def test_custom(self):
        config = ConsolidationConfig(
            fast_decay_rate=0.95,
            dedup_threshold=0.85,
        )
        assert config.fast_decay_rate == 0.95
        assert config.dedup_threshold == 0.85


# Legacy model tests (backward compat)

class TestConceptNode:
    def test_create_minimal(self):
        node = ConceptNode(type=ConceptType.FACT, content="PaddleOCR is fast")
        assert node.type == ConceptType.FACT
        assert node.content == "PaddleOCR is fast"
        assert node.confidence == 0.8
        assert node.salience == 0.5
        assert node.is_valid is True
        assert isinstance(node.id, uuid.UUID)

    def test_serialization(self):
        node = ConceptNode(type=ConceptType.ENTITY, content="Client X")
        data = node.model_dump(mode="json")
        assert data["type"] == "entity"
        restored = ConceptNode.model_validate(data)
        assert restored.id == node.id

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ConceptNode(type=ConceptType.FACT, content="test", confidence=1.5)


class TestConceptEdge:
    def test_create(self):
        n1, n2 = uuid.uuid4(), uuid.uuid4()
        edge = ConceptEdge(
            from_node=n1, to_node=n2, type=EdgeType.SUPPORTS, weight=0.8,
        )
        assert edge.from_node == n1
        assert edge.type == EdgeType.SUPPORTS

    def test_all_edge_types(self):
        for et in EdgeType:
            edge = ConceptEdge(from_node=uuid.uuid4(), to_node=uuid.uuid4(), type=et)
            assert edge.type == et


class TestIntentAnchor:
    def test_create(self):
        intent = IntentAnchor(
            objective="Build a PDF extractor",
            success_criteria=["95% accuracy"],
        )
        assert intent.status == IntentStatus.ACTIVE


class TestContext:
    def test_create(self):
        intent = IntentAnchor(objective="Test project")
        ctx = Context(name="My Project", intent=intent)
        assert ctx.name == "My Project"
        assert ctx.total_bullets == 0
        assert ctx.schema_count == 0


class TestActivity:
    def test_create(self):
        activity = Activity(
            agent_id="claude",
            action_type=ActionType.DECISION_MADE,
            summary="Chose PaddleOCR",
            delta_batch_id="batch-123",
        )
        assert activity.delta_batch_id == "batch-123"


class TestRequestModels:
    def test_create_context_request(self):
        req = CreateContextRequest(
            name="Test",
            intent=IntentInput(objective="Build something"),
        )
        assert req.name == "Test"

    def test_materialize_request_defaults(self):
        req = MaterializeRequest()
        assert req.token_budget == 4000
        assert req.target_model == "claude"
        assert req.include_schemas is True
