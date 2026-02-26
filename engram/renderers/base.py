"""Abstract base renderer for model-specific context formatting."""

from __future__ import annotations

import abc

from engram.core.models import ConceptNode, IntentAnchor


class ContextRenderer(abc.ABC):
    """Renders selected concepts into a prompt-ready string for a specific LLM."""

    @abc.abstractmethod
    def render(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
    ) -> str:
        """Render concepts into a formatted context string."""

    @abc.abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string."""
