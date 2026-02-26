"""Tests for the materialization engine and renderers."""

import uuid

import pytest

from engram.core.models import (
    ConceptNode,
    ConceptType,
    IntentAnchor,
)
from engram.renderers.claude import ClaudeRenderer
from engram.renderers.generic import GenericRenderer
from engram.renderers.gpt import GPTRenderer


@pytest.fixture
def sample_intent():
    return IntentAnchor(
        objective="Build PDF form field extractor",
        success_criteria=["95% accuracy", "Under 30s per doc"],
        constraints=["AWS infrastructure only"],
    )


@pytest.fixture
def sample_concepts():
    return [
        ConceptNode(
            type=ConceptType.DECISION,
            content="Use PaddleOCR for field detection over Textract",
            salience=0.9,
            domain_tags=["ocr", "architecture"],
        ),
        ConceptNode(
            type=ConceptType.FACT,
            content="PaddleOCR is 40% faster than Textract on complex layouts",
            salience=0.8,
            domain_tags=["ocr", "performance"],
        ),
        ConceptNode(
            type=ConceptType.CONSTRAINT,
            content="Processing must complete in under 30 seconds per document",
            salience=0.7,
        ),
        ConceptNode(
            type=ConceptType.ENTITY,
            content="Client X requires HIPAA-compliant processing",
            salience=0.6,
        ),
        ConceptNode(
            type=ConceptType.GOAL,
            content="Achieve 95% extraction accuracy across all form types",
            salience=0.85,
        ),
    ]


class TestClaudeRenderer:
    def test_render_with_intent(self, sample_intent, sample_concepts):
        renderer = ClaudeRenderer()
        result = renderer.render(sample_concepts, sample_intent, token_budget=4000)
        assert "<context>" in result
        assert "<intent>" in result
        assert "<objective>" in result
        assert "Build PDF form field extractor" in result
        assert "<key_decisions>" in result
        assert "<relevant_facts>" in result
        assert "</context>" in result

    def test_render_without_intent(self, sample_concepts):
        renderer = ClaudeRenderer()
        result = renderer.render(sample_concepts, None, token_budget=4000)
        assert "<context>" in result
        assert "<intent>" not in result

    def test_render_respects_budget(self, sample_concepts):
        renderer = ClaudeRenderer()
        # Very small budget
        result = renderer.render(sample_concepts, None, token_budget=50)
        tokens = renderer.estimate_tokens(result)
        assert tokens <= 100  # Allow some overhead

    def test_estimate_tokens(self):
        renderer = ClaudeRenderer()
        assert renderer.estimate_tokens("hello world") > 0
        assert renderer.estimate_tokens("a" * 400) == pytest.approx(100, abs=5)


class TestGPTRenderer:
    def test_render_with_intent(self, sample_intent, sample_concepts):
        renderer = GPTRenderer()
        result = renderer.render(sample_concepts, sample_intent, token_budget=4000)
        assert "# Project Context" in result
        assert "**Objective:**" in result
        assert "## Key Decisions" in result
        assert "## Known Facts" in result

    def test_render_uses_markdown(self, sample_concepts):
        renderer = GPTRenderer()
        result = renderer.render(sample_concepts, None, token_budget=4000)
        # Should use markdown bullet points
        assert "- " in result
        assert "##" in result


class TestGenericRenderer:
    def test_render(self, sample_intent, sample_concepts):
        renderer = GenericRenderer()
        result = renderer.render(sample_concepts, sample_intent, token_budget=4000)
        assert "OBJECTIVE:" in result
        assert "[DECISION]" in result
        assert "[FACT]" in result

    def test_render_respects_budget(self, sample_concepts):
        renderer = GenericRenderer()
        result = renderer.render(sample_concepts, None, token_budget=30)
        tokens = renderer.estimate_tokens(result)
        # Should be within budget (with small overhead)
        assert tokens <= 60
