"""Python SDK client for the Engram API."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from engram.core.models import (
    AddBulletRequest,
    AddConceptRequest,
    BulletType,
    CommitRequest,
    ConsolidateRequest,
    ConsolidationConfig,
    ContentType,
    ConceptType,
    CreateContextRequest,
    EdgeType,
    ExecutionFeedback,
    IntentInput,
    InvalidateRequest,
    MaterializeRequest,
    RecallRequest,
    RecordDecisionRequest,
    RelationshipInput,
)


class ContextHandle:
    """Returned by create_context / get_context — a reference to a context."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: uuid.UUID = uuid.UUID(data["id"])
        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.owner: str = data.get("owner", "default")
        self.raw: dict[str, Any] = data


class Engram:
    """Async-first SDK client for the Engram context database."""

    def __init__(
        self,
        url: str = "http://localhost:5820",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Engram:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Context Management ─────────────────────────────────────────────

    async def create_context(
        self,
        name: str,
        intent: dict[str, Any],
        description: str = "",
        owner: str = "default",
    ) -> ContextHandle:
        """Create a new context with an intent anchor."""
        req = CreateContextRequest(
            name=name,
            description=description,
            owner=owner,
            intent=IntentInput(**intent),
        )
        resp = await self._client.post(
            "/contexts", json=req.model_dump(mode="json")
        )
        resp.raise_for_status()
        return ContextHandle(resp.json())

    async def list_contexts(
        self,
        owner: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List available contexts."""
        params: dict[str, str] = {}
        if owner:
            params["owner"] = owner
        if status:
            params["status"] = status
        resp = await self._client.get("/contexts", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_context(self, context_id: uuid.UUID | str) -> ContextHandle:
        """Get a context by ID."""
        resp = await self._client.get(f"/contexts/{context_id}")
        resp.raise_for_status()
        return ContextHandle(resp.json())

    # ── Ingestion ──────────────────────────────────────────────────────

    async def commit(
        self,
        context_id: uuid.UUID | str,
        agent_id: str,
        content: str,
        content_type: str = "conversation",
        session_id: uuid.UUID | str | None = None,
        feedback: dict[str, Any] | None = None,
        materialization_id: str | None = None,
        source_model: str | None = None,
    ) -> dict[str, Any]:
        """Commit raw content — triggers Reflector → Curator → Delta pipeline.

        Pass feedback + materialization_id to trigger reconsolidation.
        v0.4: source_model identifies what model generated the raw text.
        """
        fb = ExecutionFeedback(**feedback) if feedback else None
        req = CommitRequest(
            agent_id=agent_id,
            content=content,
            content_type=ContentType(content_type),
            session_id=uuid.UUID(str(session_id)) if session_id else None,
            feedback=fb,
            materialization_id=materialization_id,
            source_model=source_model,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/commit",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def add_concept(
        self,
        context_id: uuid.UUID | str,
        concept: dict[str, Any],
        relationships: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Directly add a concept (no LLM extraction). Legacy — prefer add_bullet."""
        rels = None
        if relationships:
            rels = [
                RelationshipInput(
                    type=EdgeType(r["type"]),
                    target_content=r["target_content"],
                    rationale=r.get("rationale"),
                )
                for r in relationships
            ]
        req = AddConceptRequest(
            type=ConceptType(concept["type"]),
            content=concept["content"],
            salience=concept.get("salience", 0.5),
            confidence=concept.get("confidence", 0.8),
            domain_tags=concept.get("domain_tags", []),
            relationships=rels or [],
            agent_id=concept.get("agent_id", "sdk"),
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/concepts",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def record_decision(
        self,
        context_id: uuid.UUID | str,
        decision: str,
        rationale: str,
        alternatives_considered: list[str] | None = None,
        agent_id: str = "sdk",
        domain_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a decision explicitly."""
        req = RecordDecisionRequest(
            decision=decision,
            rationale=rationale,
            alternatives_considered=alternatives_considered or [],
            agent_id=agent_id,
            domain_tags=domain_tags or [],
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/decisions",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def invalidate(
        self,
        context_id: uuid.UUID | str,
        concept_ids: list[uuid.UUID | str] | None = None,
        bullet_ids: list[str] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Mark concepts or bullets as invalid."""
        req = InvalidateRequest(
            concept_ids=[uuid.UUID(str(cid)) for cid in (concept_ids or [])],
            bullet_ids=bullet_ids or [],
            reason=reason,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/invalidate",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    # ── Bullets (v0.2) ────────────────────────────────────────────────

    async def add_bullet(
        self,
        context_id: uuid.UUID | str,
        content: str,
        section: str = "general",
        bullet_type: str = "fact",
        salience: float = 0.5,
        confidence: float = 0.5,
        agent_id: str = "sdk",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a bullet directly (bypass Reflector pipeline)."""
        req = AddBulletRequest(
            content=content,
            section=section,
            bullet_type=BulletType(bullet_type),
            salience=salience,
            confidence=confidence,
            agent_id=agent_id,
            session_id=session_id,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/bullets",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def list_bullets(
        self,
        context_id: uuid.UUID | str,
        section: str | None = None,
        bullet_type: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List bullets in a context."""
        params: dict[str, Any] = {
            "include_archived": include_archived,
            "limit": limit,
        }
        if section:
            params["section"] = section
        if bullet_type:
            params["bullet_type"] = bullet_type
        resp = await self._client.get(
            f"/contexts/{context_id}/bullets", params=params
        )
        resp.raise_for_status()
        return resp.json()

    async def get_bullet(
        self,
        context_id: uuid.UUID | str,
        bullet_id: str,
    ) -> dict[str, Any]:
        """Get a single bullet by ID."""
        resp = await self._client.get(
            f"/contexts/{context_id}/bullets/{bullet_id}"
        )
        resp.raise_for_status()
        return resp.json()

    # ── Deltas (v0.2) ─────────────────────────────────────────────────

    async def list_deltas(
        self,
        context_id: uuid.UUID | str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List delta batches (audit trail)."""
        resp = await self._client.get(
            f"/contexts/{context_id}/deltas",
            params={"limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        return resp.json()

    async def rollback_delta(
        self,
        context_id: uuid.UUID | str,
        delta_id: str,
    ) -> dict[str, Any]:
        """Roll back a delta batch."""
        resp = await self._client.post(
            f"/contexts/{context_id}/deltas/{delta_id}/rollback"
        )
        resp.raise_for_status()
        return resp.json()

    # ── Schemas (v0.2) ────────────────────────────────────────────────

    async def list_schemas(
        self,
        context_id: uuid.UUID | str,
    ) -> list[dict[str, Any]]:
        """List schemas (abstract patterns) in a context."""
        resp = await self._client.get(f"/contexts/{context_id}/schemas")
        resp.raise_for_status()
        return resp.json()

    # ── Consolidation (v0.2) ──────────────────────────────────────────

    async def consolidate(
        self,
        context_id: uuid.UUID | str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the consolidation engine (sleep cycle) on a context."""
        req = ConsolidateRequest(
            config=ConsolidationConfig(**config) if config else None,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/consolidate",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    # ── Health (v0.2) ─────────────────────────────────────────────────

    async def get_health(
        self,
        context_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Get health metrics for a context."""
        resp = await self._client.get(f"/contexts/{context_id}/health")
        resp.raise_for_status()
        return resp.json()

    # ── Materialization ────────────────────────────────────────────────

    async def materialize(
        self,
        context_id: uuid.UUID | str,
        query: str | None = None,
        task: str | None = None,
        agent_role: str | None = None,
        focus_domains: list[str] | None = None,
        focus_sections: list[str] | None = None,
        token_budget: int = 4000,
        target_model: str = "claude",
        include_intent: bool = True,
        include_decisions: bool = True,
        include_schemas: bool = True,
        recency_weight: float = 0.5,
        max_concept_age_days: int | None = None,
    ) -> dict[str, Any]:
        """Materialize context for an agent. Returns materialization_id for reconsolidation."""
        req = MaterializeRequest(
            query=query,
            task=task,
            agent_role=agent_role,
            focus_domains=focus_domains or [],
            focus_sections=focus_sections or [],
            token_budget=token_budget,
            target_model=target_model,
            include_intent=include_intent,
            include_decisions=include_decisions,
            include_schemas=include_schemas,
            recency_weight=recency_weight,
            max_concept_age_days=max_concept_age_days,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/materialize",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()

    async def recall(
        self,
        context_id: uuid.UUID | str,
        query: str,
        token_budget: int = 2000,
        target_model: str = "claude",
    ) -> str:
        """Quick recall — returns rendered context text."""
        req = RecallRequest(
            query=query,
            token_budget=token_budget,
            target_model=target_model,
        )
        resp = await self._client.post(
            f"/contexts/{context_id}/recall",
            json=req.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return resp.json()["context"]

    # ── Lifecycle (v0.3) ────────────────────────────────────────────────

    async def archive_bullet(
        self,
        context_id: uuid.UUID | str,
        bullet_id: str,
        reason: str = "manual",
    ) -> dict[str, Any]:
        """Archive a bullet."""
        resp = await self._client.post(
            f"/contexts/{context_id}/bullets/{bullet_id}/archive",
            json={"reason": reason},
        )
        resp.raise_for_status()
        return resp.json()

    async def restore_bullet(
        self,
        context_id: uuid.UUID | str,
        bullet_id: str,
    ) -> dict[str, Any]:
        """Restore an archived bullet."""
        resp = await self._client.post(
            f"/contexts/{context_id}/bullets/{bullet_id}/restore"
        )
        resp.raise_for_status()
        return resp.json()

    async def list_archived_bullets(
        self,
        context_id: uuid.UUID | str,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List archived bullets."""
        resp = await self._client.get(
            f"/contexts/{context_id}/archived-bullets",
            params={"offset": offset, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_lifecycle(
        self,
        context_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Get lifecycle status (capacity + config)."""
        resp = await self._client.get(f"/contexts/{context_id}/lifecycle")
        resp.raise_for_status()
        return resp.json()

    async def purge_context(
        self,
        context_id: uuid.UUID | str,
    ) -> dict[str, Any]:
        """Permanently delete all data for a context (GDPR erasure)."""
        resp = await self._client.delete(f"/contexts/{context_id}/purge")
        resp.raise_for_status()
        return resp.json()

    async def sync(
        self,
        context_id: uuid.UUID | str,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Get delta batches since a timestamp (polling alternative to SSE)."""
        params: dict[str, str] = {}
        if since:
            params["since"] = since
        resp = await self._client.post(
            f"/contexts/{context_id}/sync",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Re-extraction (v0.4) ────────────────────────────────────────────

    async def re_extract(
        self,
        context_id: uuid.UUID | str,
        reflector_model: str | None = None,
        since: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Re-extract bullets from raw history with a new Reflector model.

        Set dry_run=False to actually apply changes.
        """
        body: dict[str, Any] = {"dry_run": dry_run}
        if reflector_model:
            body["reflector_model"] = reflector_model
        if since:
            body["since"] = since
        resp = await self._client.post(
            f"/contexts/{context_id}/re-extract",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_ingestion_config(self) -> dict[str, Any]:
        """Get the server-level ingestion configuration."""
        resp = await self._client.get("/config/ingestion")
        resp.raise_for_status()
        return resp.json()

    async def update_ingestion_config(
        self, **kwargs: Any,
    ) -> dict[str, Any]:
        """Update server-level ingestion configuration.

        Pass keyword arguments for fields to update, e.g.:
            await engram.update_ingestion_config(reflector_model="claude-sonnet-4-20250514")
        """
        resp = await self._client.put("/config/ingestion", json=kwargs)
        resp.raise_for_status()
        return resp.json()

    # ── Activity ───────────────────────────────────────────────────────

    async def get_activity(
        self,
        context_id: uuid.UUID | str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get activity ledger."""
        resp = await self._client.get(
            f"/contexts/{context_id}/activity",
            params={"limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_concepts(
        self,
        context_id: uuid.UUID | str,
        include_invalid: bool = False,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List concepts in a context (legacy)."""
        params: dict[str, Any] = {"include_invalid": include_invalid}
        if type_filter:
            params["type_filter"] = type_filter
        resp = await self._client.get(
            f"/contexts/{context_id}/concepts",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()
