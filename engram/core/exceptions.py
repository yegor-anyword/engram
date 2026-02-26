"""Custom exception classes for Engram."""


class EngramError(Exception):
    """Base exception for all Engram errors."""


class ContextNotFoundError(EngramError):
    """Raised when a context ID does not exist."""

    def __init__(self, context_id: str) -> None:
        super().__init__(f"Context not found: {context_id}")
        self.context_id = context_id


class ConceptNotFoundError(EngramError):
    """Raised when a concept ID does not exist."""

    def __init__(self, concept_id: str) -> None:
        super().__init__(f"Concept not found: {concept_id}")
        self.concept_id = concept_id


class StorageError(EngramError):
    """Raised when a storage operation fails."""


class IngestionError(EngramError):
    """Raised when concept extraction fails."""


class MaterializationError(EngramError):
    """Raised when context materialization fails."""


class LLMAdapterError(EngramError):
    """Raised when an LLM call fails."""


class CapacityExceededError(EngramError):
    """Raised when a context's active bullet limit is reached."""

    def __init__(self, context_id: str, current: int, maximum: int) -> None:
        super().__init__(
            f"Context {context_id} at capacity: {current}/{maximum} active bullets"
        )
        self.context_id = context_id
        self.current = current
        self.maximum = maximum


class ConcurrencyError(EngramError):
    """Raised when a context lock cannot be acquired within timeout."""

    def __init__(self, context_id: str, timeout: float) -> None:
        super().__init__(
            f"Failed to acquire lock for context {context_id} within {timeout}s"
        )
        self.context_id = context_id
        self.timeout = timeout
