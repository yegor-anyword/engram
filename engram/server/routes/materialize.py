"""Materialization API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from engram.core.models import MaterializeRequest, MaterializeResponse, RecallRequest

router = APIRouter()


@router.post("/{context_id}/materialize", response_model=MaterializeResponse)
async def materialize_context(
    context_id: uuid.UUID, req: MaterializeRequest, request: Request
) -> MaterializeResponse:
    """Materialize context — assemble relevant bullets/concepts and render for a target model.

    Returns a materialization_id that can be passed back in a subsequent commit
    for reconsolidation (tracking which bullets were useful).
    """
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    engine = request.app.state.materialization
    result = await engine.materialize(
        context_id=context_id,
        query=req.query,
        task=req.task,
        agent_role=req.agent_role,
        focus_domains=req.focus_domains,
        focus_sections=req.focus_sections,
        token_budget=req.token_budget,
        target_model=req.target_model,
        include_intent=req.include_intent,
        include_decisions=req.include_decisions,
        include_schemas=req.include_schemas,
        recency_weight=req.recency_weight,
        max_concept_age_days=req.max_concept_age_days,
        include_worked_examples=req.include_worked_examples,
        worked_example_threshold=req.worked_example_threshold,
        worked_example_limit=req.worked_example_limit,
        include_usage_stats=req.include_usage_stats,
        mmr_lambda=req.mmr_lambda,
    )

    return MaterializeResponse(
        materialization_id=result["materialization_id"],
        rendered_text=result["rendered_text"],
        concepts_included=result.get("concepts_included", []),
        bullets_included=result.get("bullets_included", []),
        schemas_included=result.get("schemas_included", []),
        token_count=result["token_count"],
        coverage_score=result["coverage_score"],
    )


@router.post("/{context_id}/recall")
async def recall_context(
    context_id: uuid.UUID, req: RecallRequest, request: Request
) -> dict:
    """Quick recall — simplified materialization that returns just the rendered text."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    engine = request.app.state.materialization
    text = await engine.recall(
        context_id=context_id,
        query=req.query,
        token_budget=req.token_budget,
        target_model=req.target_model,
    )
    return {"context": text}
