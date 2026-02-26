"""Generic plain-text renderer as a fallback for any model."""

from __future__ import annotations

from engram.core.models import ConceptNode, ConceptType, IntentAnchor
from engram.renderers.base import ContextRenderer


class GenericRenderer(ContextRenderer):
    """Fallback renderer — outputs plain structured text."""

    def render(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
    ) -> str:
        sections: list[str] = []

        if intent:
            sections.append(f"OBJECTIVE: {intent.objective}")
            if intent.success_criteria:
                sections.append(
                    "SUCCESS CRITERIA: " + "; ".join(intent.success_criteria)
                )
            if intent.constraints:
                sections.append("CONSTRAINTS: " + "; ".join(intent.constraints))
            sections.append("")

        current_tokens = self.estimate_tokens("\n".join(sections))

        for concept in concepts:
            line = f"[{concept.type.value.upper()}] {concept.content}"
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > token_budget:
                break
            sections.append(line)
            current_tokens += line_tokens

        return "\n".join(sections)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4 + 1
