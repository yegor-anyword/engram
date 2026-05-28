"""Configuration management via environment variables.

v0.4: Adds IngestionConfig for canonical Reflector model configuration.
"""

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class IngestionConfig(BaseModel):
    """Server-level configuration for the ingestion pipeline.

    CRITICAL: The Reflector model is configured HERE, not per-agent.
    This guarantees consistent bullet extraction quality regardless
    of which agent (Claude, GPT, Gemini, local model) committed
    the raw input.
    """

    # Canonical Reflector — same model processes ALL raw inputs
    reflector_model: str = "claude-haiku-4-5"
    reflector_prompt_version: str = "v1"
    max_reflection_rounds: int = 2

    # Curator configuration
    curator_dedup_threshold: float = 0.92
    curator_slow_path_model: str = "claude-haiku-4-5"

    # v0.5: Mem-α-inspired content-validity gate. When enabled, the Curator
    # runs a single batched LLM-judge call across all proposed ADD_BULLET ops
    # and drops any that fail validity (empty/trivial/malformed). One extra
    # LLM call per commit — opt-in.
    enable_validity_gate: bool = False
    validity_gate_model: str = "claude-haiku-4-5"

    # Embedding model — must be consistent within a context
    embedding_model: str = "text-embedding-3-small"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "ENGRAM_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 5820
    log_level: str = "info"

    storage_backend: str = "sqlite"
    sqlite_path: str = "./engram.db"

    # PostgreSQL settings (used when storage_backend = "postgres")
    postgres_dsn: str = ""

    llm_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""

    # v0.4: Ingestion pipeline configuration
    reflector_model: str = "claude-haiku-4-5"
    reflector_prompt_version: str = "v1"
    max_reflection_rounds: int = 2
    curator_dedup_threshold: float = 0.92
    curator_slow_path_model: str = "claude-haiku-4-5"

    # v0.5: validity gate (off by default — see IngestionConfig)
    enable_validity_gate: bool = False
    validity_gate_model: str = "claude-haiku-4-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
