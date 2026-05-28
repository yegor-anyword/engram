"""FastAPI application factory and server entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from engram.core.concurrency import ContextLockManager
from engram.core.config import IngestionConfig, get_settings
from engram.core.consolidation import ConsolidationEngine
from engram.core.delta import DeltaEngine
from engram.core.events import EventBus
from engram.core.graph import ConceptGraph
from engram.core.ingestion import CuratorEngine, IngestionEngine, ReflectorEngine
from engram.core.materialization import MaterializationEngine
from engram.core.re_extraction import ReExtractionEngine
from engram.llm.adapter import LiteLLMAdapter
from engram.server.routes.bullets import router as bullets_router
from engram.server.routes.concepts import router as concepts_router
from engram.server.routes.contexts import router as contexts_router
from engram.server.routes.deltas import router as deltas_router
from engram.server.routes.lifecycle import config_router
from engram.server.routes.lifecycle import router as lifecycle_router
from engram.server.routes.lifecycle import user_router
from engram.server.routes.materialize import router as materialize_router
from engram.server.routes.schemas import router as schemas_router
from engram.storage.base import StorageBackend
from engram.storage.sqlite import SQLiteBackend

logger = logging.getLogger(__name__)


def _create_storage() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == "postgres":
        from engram.storage.postgres import PostgresBackend
        if not settings.postgres_dsn:
            raise ValueError(
                "ENGRAM_POSTGRES_DSN must be set when storage_backend is 'postgres'"
            )
        return PostgresBackend(dsn=settings.postgres_dsn)
    elif settings.storage_backend == "sqlite":
        return SQLiteBackend(db_path=settings.sqlite_path)
    else:
        raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


def _create_llm() -> LiteLLMAdapter:
    settings = get_settings()
    return LiteLLMAdapter(
        model=settings.llm_model,
        api_key=settings.llm_api_key or None,
        embedding_model=settings.embedding_model,
        embedding_api_key=settings.embedding_api_key or None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and teardown application resources."""
    storage = _create_storage()
    await storage.initialize()
    llm = _create_llm()

    # v0.3: Concurrency and event infrastructure
    lock_manager = ContextLockManager()
    event_bus = EventBus()

    # v0.4: Canonical ingestion config from env vars
    settings = get_settings()
    ingestion_config = IngestionConfig(
        reflector_model=settings.reflector_model,
        reflector_prompt_version=settings.reflector_prompt_version,
        max_reflection_rounds=settings.max_reflection_rounds,
        curator_dedup_threshold=settings.curator_dedup_threshold,
        curator_slow_path_model=settings.curator_slow_path_model,
        enable_validity_gate=settings.enable_validity_gate,
        validity_gate_model=settings.validity_gate_model,
        embedding_model=settings.embedding_model,
    )

    # Attach engines to app state
    ingestion_engine = IngestionEngine(
        storage, llm, lock_manager, event_bus, ingestion_config
    )
    app.state.storage = storage
    app.state.llm = llm
    app.state.lock_manager = lock_manager
    app.state.event_bus = event_bus
    app.state.ingestion_config = ingestion_config
    app.state.ingestion = ingestion_engine
    app.state.materialization = MaterializationEngine(storage, llm)
    app.state.graph = ConceptGraph(storage)
    app.state.delta_engine = DeltaEngine(storage)
    app.state.consolidation = ConsolidationEngine(storage, llm, lock_manager, event_bus)

    # v0.4: Re-extraction engine
    re_reflector = ReflectorEngine(llm, config=ingestion_config)
    re_curator = CuratorEngine(storage, llm)
    app.state.re_extraction = ReExtractionEngine(
        re_reflector, re_curator, storage, lock_manager, event_bus,
    )

    logger.info("Engram server started (v0.5)")
    yield

    await storage.close()
    logger.info("Engram server stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Engram",
        description="A portable, model-agnostic context database for AI agents",
        version="0.5.0",
        lifespan=lifespan,
    )

    app.include_router(contexts_router, prefix="/contexts", tags=["contexts"])
    app.include_router(concepts_router, prefix="/contexts", tags=["concepts"])
    app.include_router(bullets_router, prefix="/contexts", tags=["bullets"])
    app.include_router(deltas_router, prefix="/contexts", tags=["deltas"])
    app.include_router(schemas_router, prefix="/contexts", tags=["schemas"])
    app.include_router(materialize_router, prefix="/contexts", tags=["materialize"])
    app.include_router(lifecycle_router, prefix="/contexts", tags=["lifecycle"])
    app.include_router(config_router, tags=["config"])
    app.include_router(user_router, tags=["lifecycle"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.5.0"}

    return app


def main() -> None:
    """Entry point for `engram` CLI command."""
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
