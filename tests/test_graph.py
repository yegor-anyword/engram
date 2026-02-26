"""Tests for the storage layer, concept graph engine, and v0.2 bullet operations."""

import uuid

import pytest

from engram.core.graph import ConceptGraph
from engram.core.models import (
    Bullet,
    BulletType,
    ConceptEdge,
    ConceptNode,
    ConceptType,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    EdgeType,
    IntentAnchor,
    MaterializationRecord,
    SchemaNode,
    SourceType,
)
from engram.storage.sqlite import SQLiteBackend


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test project")
    ctx = Context(name="Test Context", intent=intent)
    await storage.create_context(ctx)
    return ctx.id


# ── Legacy Concept Storage Tests ──────────────────────────────────────────


class TestSQLiteStorage:
    async def test_create_and_get_context(self, storage):
        intent = IntentAnchor(
            objective="Build PDF extractor",
            success_criteria=["95% accuracy"],
        )
        ctx = Context(name="PDF Project", intent=intent, owner="alice")
        created = await storage.create_context(ctx)
        assert created.id == ctx.id

        loaded = await storage.get_context(ctx.id)
        assert loaded is not None
        assert loaded.name == "PDF Project"
        assert loaded.intent.objective == "Build PDF extractor"

    async def test_list_contexts(self, storage):
        for i in range(3):
            intent = IntentAnchor(objective=f"Task {i}")
            ctx = Context(name=f"Context {i}", intent=intent, owner="bob")
            await storage.create_context(ctx)

        all_contexts = await storage.list_contexts()
        assert len(all_contexts) == 3

        bob_contexts = await storage.list_contexts(owner="bob")
        assert len(bob_contexts) == 3

        nobody_contexts = await storage.list_contexts(owner="nobody")
        assert len(nobody_contexts) == 0

    async def test_context_not_found(self, storage):
        result = await storage.get_context(uuid.uuid4())
        assert result is None

    async def test_add_and_get_concept(self, storage, context_id):
        concept = ConceptNode(
            type=ConceptType.FACT,
            content="PaddleOCR is 40% faster than Textract",
            salience=0.8,
            domain_tags=["ocr", "performance"],
        )
        await storage.add_concept(context_id, concept)

        loaded = await storage.get_concept(concept.id)
        assert loaded is not None
        assert loaded.content == "PaddleOCR is 40% faster than Textract"
        assert loaded.domain_tags == ["ocr", "performance"]

    async def test_list_concepts_filters(self, storage, context_id):
        for ctype in [ConceptType.FACT, ConceptType.DECISION, ConceptType.FACT]:
            c = ConceptNode(type=ctype, content=f"Content for {ctype.value}")
            await storage.add_concept(context_id, c)

        all_concepts = await storage.list_concepts(context_id)
        assert len(all_concepts) == 3

        facts = await storage.list_concepts(context_id, type_filter="fact")
        assert len(facts) == 2

    async def test_add_and_get_edges(self, storage, context_id):
        c1 = ConceptNode(type=ConceptType.DECISION, content="Use PaddleOCR")
        c2 = ConceptNode(type=ConceptType.FACT, content="PaddleOCR is fast")
        await storage.add_concept(context_id, c1)
        await storage.add_concept(context_id, c2)

        edge = ConceptEdge(
            from_node=c2.id, to_node=c1.id, type=EdgeType.SUPPORTS, weight=0.9,
        )
        await storage.add_edge(context_id, edge)

        edges = await storage.get_edges(context_id)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS

    async def test_count_concepts(self, storage, context_id):
        assert await storage.count_concepts(context_id) == 0
        await storage.add_concept(
            context_id, ConceptNode(type=ConceptType.FACT, content="A")
        )
        assert await storage.count_concepts(context_id) == 1


# ── v0.2 Bullet Storage Tests ────────────────────────────────────────────


class TestBulletStorage:
    async def test_add_and_get_bullet(self, storage, context_id):
        ctx_id_str = str(context_id)
        bullet = Bullet(
            content="PaddleOCR outperforms Textract",
            section="ocr_tools",
            bullet_type=BulletType.FACT,
            salience=0.8,
        )
        await storage.add_bullet(ctx_id_str, bullet)

        loaded = await storage.get_bullet(bullet.id)
        assert loaded is not None
        assert loaded.content == "PaddleOCR outperforms Textract"
        assert loaded.section == "ocr_tools"
        assert loaded.salience == pytest.approx(0.8)

    async def test_list_bullets(self, storage, context_id):
        ctx_id_str = str(context_id)
        for i in range(5):
            b = Bullet(content=f"Fact {i}", section="general")
            await storage.add_bullet(ctx_id_str, b)

        bullets = await storage.list_bullets(ctx_id_str)
        assert len(bullets) == 5

    async def test_list_bullets_by_section(self, storage, context_id):
        ctx_id_str = str(context_id)
        for i in range(3):
            await storage.add_bullet(ctx_id_str, Bullet(content=f"OCR fact {i}", section="ocr"))
        for i in range(2):
            await storage.add_bullet(ctx_id_str, Bullet(content=f"API fact {i}", section="api"))

        ocr_bullets = await storage.list_bullets(ctx_id_str, section="ocr")
        assert len(ocr_bullets) == 3

        api_bullets = await storage.list_bullets(ctx_id_str, section="api")
        assert len(api_bullets) == 2

    async def test_list_bullets_by_type(self, storage, context_id):
        ctx_id_str = str(context_id)
        await storage.add_bullet(ctx_id_str, Bullet(content="Use retry", bullet_type=BulletType.STRATEGY))
        await storage.add_bullet(ctx_id_str, Bullet(content="API is fast", bullet_type=BulletType.FACT))
        await storage.add_bullet(ctx_id_str, Bullet(content="Watch for timeouts", bullet_type=BulletType.WARNING))

        strategies = await storage.list_bullets(ctx_id_str, bullet_type="strategy")
        assert len(strategies) == 1
        assert strategies[0].bullet_type == BulletType.STRATEGY

    async def test_update_bullet(self, storage, context_id):
        ctx_id_str = str(context_id)
        bullet = Bullet(content="Original", salience=0.5)
        await storage.add_bullet(ctx_id_str, bullet)

        bullet.content = "Updated content"
        bullet.salience = 0.9
        bullet.hit_count = 3
        await storage.update_bullet(bullet)

        loaded = await storage.get_bullet(bullet.id)
        assert loaded.content == "Updated content"
        assert loaded.salience == pytest.approx(0.9)
        assert loaded.hit_count == 3

    async def test_remove_bullet(self, storage, context_id):
        ctx_id_str = str(context_id)
        bullet = Bullet(content="Will be removed")
        await storage.add_bullet(ctx_id_str, bullet)

        await storage.remove_bullet(bullet.id)

        # Should not appear in default listing
        bullets = await storage.list_bullets(ctx_id_str)
        assert len(bullets) == 0

    async def test_count_bullets(self, storage, context_id):
        ctx_id_str = str(context_id)
        assert await storage.count_bullets(ctx_id_str) == 0

        await storage.add_bullet(ctx_id_str, Bullet(content="A"))
        await storage.add_bullet(ctx_id_str, Bullet(content="B"))
        assert await storage.count_bullets(ctx_id_str) == 2

    async def test_find_similar_bullets(self, storage, context_id):
        ctx_id_str = str(context_id)
        b1 = Bullet(content="OCR perf", embedding=[1.0, 0.0, 0.0])
        b2 = Bullet(content="Cloud costs", embedding=[0.0, 1.0, 0.0])
        b3 = Bullet(content="OCR accuracy", embedding=[0.9, 0.1, 0.0])
        for b in [b1, b2, b3]:
            await storage.add_bullet(ctx_id_str, b)

        results = await storage.find_similar_bullets(
            ctx_id_str, [1.0, 0.0, 0.0], limit=5, threshold=0.5
        )
        assert len(results) >= 2
        assert results[0][0].id == b1.id
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    async def test_archived_bullets(self, storage, context_id):
        ctx_id_str = str(context_id)
        b = Bullet(content="Archive me")
        await storage.add_bullet(ctx_id_str, b)

        b.is_archived = True
        await storage.update_bullet(b)

        # Default listing excludes archived
        assert len(await storage.list_bullets(ctx_id_str)) == 0

        # Include archived
        assert len(await storage.list_bullets(ctx_id_str, include_archived=True)) == 1


# ── Schema Storage Tests ─────────────────────────────────────────────────


class TestSchemaStorage:
    async def test_add_and_get_schema(self, storage, context_id):
        ctx_id_str = str(context_id)
        schema = SchemaNode(
            name="error_handling",
            description="Pattern for retry with backoff",
            instance_count=5,
            confidence=0.8,
        )
        await storage.add_schema(ctx_id_str, schema)

        loaded = await storage.get_schema(schema.id)
        assert loaded is not None
        assert loaded.name == "error_handling"
        assert loaded.instance_count == 5

    async def test_list_schemas(self, storage, context_id):
        ctx_id_str = str(context_id)
        for name in ["patterns_a", "patterns_b"]:
            await storage.add_schema(
                ctx_id_str, SchemaNode(name=name, description=f"Desc for {name}")
            )

        schemas = await storage.list_schemas(ctx_id_str)
        assert len(schemas) == 2

    async def test_update_schema(self, storage, context_id):
        ctx_id_str = str(context_id)
        schema = SchemaNode(name="test", description="Original")
        await storage.add_schema(ctx_id_str, schema)

        schema.description = "Updated"
        schema.instance_count = 10
        await storage.update_schema(schema)

        loaded = await storage.get_schema(schema.id)
        assert loaded.description == "Updated"
        assert loaded.instance_count == 10


# ── Delta Batch Storage Tests ─────────────────────────────────────────────


class TestDeltaBatchStorage:
    async def test_save_and_get_delta_batch(self, storage, context_id):
        ctx_id_str = str(context_id)
        batch = DeltaBatch(
            context_id=ctx_id_str,
            operations=[
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="Fact 1"),
            ],
            bullets_added=1,
        )
        await storage.save_delta_batch(batch)

        loaded = await storage.get_delta_batch(batch.id)
        assert loaded is not None
        assert loaded.bullets_added == 1
        assert len(loaded.operations) == 1

    async def test_list_delta_batches(self, storage, context_id):
        ctx_id_str = str(context_id)
        for i in range(3):
            batch = DeltaBatch(
                context_id=ctx_id_str,
                operations=[DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content=f"Fact {i}")],
            )
            await storage.save_delta_batch(batch)

        batches = await storage.list_delta_batches(ctx_id_str)
        assert len(batches) == 3


# ── Materialization Record Storage Tests ──────────────────────────────────


class TestMaterializationStorage:
    async def test_save_and_get(self, storage, context_id):
        ctx_id_str = str(context_id)
        record = MaterializationRecord(
            context_id=ctx_id_str,
            bullets_included=["b1", "b2", "b3"],
            token_count=500,
            target_model="claude",
            query="OCR tools",
        )
        await storage.save_materialization(record)

        loaded = await storage.get_materialization(record.id)
        assert loaded is not None
        assert loaded.bullets_included == ["b1", "b2", "b3"]
        assert loaded.token_count == 500


# ── Concept Graph Tests ──────────────────────────────────────────────────


class TestConceptGraph:
    async def test_invalidate_concepts(self, storage, context_id):
        graph = ConceptGraph(storage)
        c1 = ConceptNode(type=ConceptType.FACT, content="Stale fact")
        await storage.add_concept(context_id, c1)

        invalidated = await graph.invalidate_concepts(
            context_id, [c1.id], "Requirements changed"
        )
        assert len(invalidated) == 1

        reloaded = await storage.get_concept(c1.id)
        assert reloaded.is_valid is False

    async def test_get_neighborhood(self, storage, context_id):
        graph = ConceptGraph(storage)
        c1 = ConceptNode(type=ConceptType.DECISION, content="Main decision")
        c2 = ConceptNode(type=ConceptType.FACT, content="Supporting fact")
        c3 = ConceptNode(type=ConceptType.ENTITY, content="Related entity")
        for c in [c1, c2, c3]:
            await storage.add_concept(context_id, c)

        edge1 = ConceptEdge(from_node=c2.id, to_node=c1.id, type=EdgeType.SUPPORTS)
        edge2 = ConceptEdge(from_node=c3.id, to_node=c1.id, type=EdgeType.RELATED_TO)
        await storage.add_edge(context_id, edge1)
        await storage.add_edge(context_id, edge2)

        neighbors = await graph.get_concept_neighborhood(context_id, c1.id, depth=1)
        assert len(neighbors) == 3
