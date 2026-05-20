"""Tests for Phase 3a: MMR diversity re-ranking in materialization.

MMR formula (in engram.core.materialization._mmr_order):
  score(b) = λ * relevance(b) - (1 - λ) * max_sim(b, already_picked)

Coverage:
1. λ ≈ 1.0 reproduces greedy-by-relevance ordering.
2. λ < 1.0 demotes near-duplicates of an already-picked high-scoring bullet,
   so a moderately-relevant DIFFERENT bullet jumps ahead in the order.
3. Bullets without embeddings still produce a stable ordering (no crash; the
   redundancy term is treated as 0 for those candidates).
4. End-to-end: MaterializationEngine.materialize with mmr_lambda=0.5 packs a
   more diverse set than with mmr_lambda=1.0 when there are many duplicates.
"""

from __future__ import annotations

import json

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import IngestionEngine
from engram.core.materialization import MaterializationEngine, _mmr_order
from engram.core.models import (
    Bullet,
    BulletType,
    ContentType,
    Context,
    IntentAnchor,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class StubLLM(LLMAdapter):
    def __init__(self, response: dict | None = None):
        self.response = response or {
            "new_insights": [],
            "strategies_that_worked": [],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.7,
        }

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        return json.dumps(self.response)

    async def embed(self, text):
        vec = [0.0] * 64
        for word in text.lower().split():
            vec[hash(word) % 64] += 1.0
        import math
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "mmr.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="MMR", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)
    return ctx.id


# ── 1. λ ≈ 1.0 collapses to greedy-by-relevance ──────────────────────────────

def test_mmr_lambda_one_is_greedy_by_relevance():
    bullets = [
        Bullet(id="a", content="alpha", embedding=[1.0, 0.0]),
        Bullet(id="b", content="beta", embedding=[0.0, 1.0]),
        Bullet(id="c", content="gamma", embedding=[0.7, 0.7]),
    ]
    relevance = {"a": 0.1, "b": 0.9, "c": 0.5}
    order = _mmr_order(bullets, relevance, lambda_=1.0)
    assert [b.id for b in order] == ["b", "c", "a"]


# ── 2. λ < 1.0 prefers diversity over near-duplicates ────────────────────────

def test_mmr_lambda_half_demotes_near_duplicates():
    # b1 and b2 share the embedding (perfect cosine 1.0) — duplicates.
    # b3 is orthogonal — diverse.
    bullets = [
        Bullet(id="b1", content="first", embedding=[1.0, 0.0]),
        Bullet(id="b2", content="duplicate of b1", embedding=[1.0, 0.0]),
        Bullet(id="b3", content="diverse", embedding=[0.0, 1.0]),
    ]
    # b1 wins on raw relevance; b2 is nearly tied; b3 is below them.
    relevance = {"b1": 0.95, "b2": 0.90, "b3": 0.60}

    # Greedy: b1, b2, b3
    greedy = _mmr_order(bullets, relevance, lambda_=1.0)
    assert [b.id for b in greedy] == ["b1", "b2", "b3"]

    # MMR λ=0.5: b1 first; for the second pick, b2 gets penalized by full
    # redundancy with b1 (cosine 1.0), so b3 jumps ahead.
    diverse = _mmr_order(bullets, relevance, lambda_=0.5)
    assert diverse[0].id == "b1"
    assert diverse[1].id == "b3"
    assert diverse[2].id == "b2"


# ── 3. Bullets without embeddings don't crash ────────────────────────────────

def test_mmr_handles_missing_embeddings():
    bullets = [
        Bullet(id="x", content="x", embedding=None),
        Bullet(id="y", content="y", embedding=[1.0, 0.0]),
        Bullet(id="z", content="z", embedding=None),
    ]
    relevance = {"x": 0.5, "y": 0.8, "z": 0.3}
    order = _mmr_order(bullets, relevance, lambda_=0.5)
    # All three should appear once.
    assert sorted(b.id for b in order) == ["x", "y", "z"]
    # Highest raw relevance still leads.
    assert order[0].id == "y"


# ── 4. End-to-end: materialization with low λ packs a more diverse set ───────

async def test_materialize_mmr_reduces_duplicate_packing(storage, context_id):
    """Seed several near-duplicate bullets that share all words with the query
    plus diverse bullets that share NO words with the query. Query relevance
    heavily favors the duplicates; only MMR breaks the tie in favor of variety.

    Budget is tightened so only a few bullets fit, making the difference observable.
    """
    llm = StubLLM()
    engine = IngestionEngine(storage, llm, ingestion_config=IngestionConfig())

    # 5 near-duplicates: all share words with the query.
    for i in range(5):
        await engine.add_bullet_directly(
            context_id=str(context_id),
            content=f"OCR PDFs benchmark variant {i}",
            bullet_type=BulletType.FACT,
            salience=0.9,
        )
    # 3 diverse bullets: share NO words with the query.
    for content in [
        "Budget limited to USD 5000",
        "Deadline Q3 release window",
        "Customer prefers Python tooling",
    ]:
        await engine.add_bullet_directly(
            context_id=str(context_id),
            content=content,
            bullet_type=BulletType.FACT,
            salience=0.5,
        )

    mat = MaterializationEngine(storage, llm)
    common_kwargs = dict(
        context_id=context_id,
        query="OCR PDFs benchmark",
        token_budget=2000,
        target_model="claude",
        include_worked_examples=False,
    )

    greedy = await mat.materialize(**common_kwargs, mmr_lambda=1.0)
    diverse = await mat.materialize(**common_kwargs, mmr_lambda=0.3)

    def first_n_bullet_ids(result: dict, n: int) -> list[str]:
        return result["bullets_included"][:n]

    # Look at the first 3 picks. With greedy, all 3 are OCR duplicates. With MMR,
    # at least one of the first 3 should be a diverse-topic bullet.
    async def ids_to_contents(ids: list[str]) -> list[str]:
        out = []
        for bid in ids:
            b = await storage.get_bullet(bid)
            if b:
                out.append(b.content)
        return out

    greedy_top3 = await ids_to_contents(first_n_bullet_ids(greedy, 3))
    mmr_top3 = await ids_to_contents(first_n_bullet_ids(diverse, 3))

    greedy_diverse_topics = [c for c in greedy_top3
                              if any(k in c for k in ("Budget", "Deadline", "Customer"))]
    mmr_diverse_topics = [c for c in mmr_top3
                           if any(k in c for k in ("Budget", "Deadline", "Customer"))]

    assert len(greedy_diverse_topics) == 0, f"greedy top-3 = {greedy_top3}"
    assert len(mmr_diverse_topics) >= 1, f"mmr top-3 = {mmr_top3}"
