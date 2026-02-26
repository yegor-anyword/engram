"""OpenAI function calling tool definitions for Engram integration.

v0.3: Added archive_bullet and get_lifecycle tools.
v0.4: Added re_extract_context and get_ingestion_config tools.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from engram.sdk.client import Engram

# OpenAI-compatible tool definitions
ENGRAM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recall_context",
            "description": (
                "Recall relevant context from the Engram knowledge graph. "
                "Returns context formatted for your model. "
                "Save the returned materialization_id for reconsolidation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context to recall from",
                    },
                    "query": {
                        "type": "string",
                        "description": "What information do you need?",
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Maximum tokens for the response (default: 2000)",
                    },
                },
                "required": ["context_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_context",
            "description": (
                "Save conversation content to the Engram knowledge graph. "
                "The system extracts insights via Reflector → Curator pipeline. "
                "Pass materialization_id + outcome for reconsolidation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context to save to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to save",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Identifier for this agent",
                    },
                    "materialization_id": {
                        "type": "string",
                        "description": "ID from a previous recall — enables reconsolidation",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failure", "partial", "unknown"],
                        "description": "How the recalled context performed",
                    },
                },
                "required": ["context_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_bullet",
            "description": (
                "Save a single knowledge bullet directly (bypass LLM pipeline). "
                "Use for atomic, well-formed insights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context",
                    },
                    "content": {
                        "type": "string",
                        "description": "The bullet content — one atomic insight",
                    },
                    "section": {
                        "type": "string",
                        "description": "Section grouping (e.g., 'architecture', 'debugging')",
                    },
                    "bullet_type": {
                        "type": "string",
                        "enum": ["strategy", "warning", "fact", "procedure", "exception", "principle", "decision"],
                        "description": "Type of bullet",
                    },
                },
                "required": ["context_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_decision",
            "description": "Record a decision with rationale to the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context",
                    },
                    "decision": {
                        "type": "string",
                        "description": "What was decided",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this decision was made",
                    },
                    "alternatives_considered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other options that were considered",
                    },
                },
                "required": ["context_id", "decision", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contexts",
            "description": "List available Engram contexts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Filter by owner (optional)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health",
            "description": "Get health metrics for a context — bullet stats, staleness, schemas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context",
                    },
                },
                "required": ["context_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_bullet",
            "description": "Archive a bullet — moves it to cold storage. Can be restored later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context",
                    },
                    "bullet_id": {
                        "type": "string",
                        "description": "ID of the bullet to archive",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for archiving",
                    },
                },
                "required": ["context_id", "bullet_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lifecycle",
            "description": "Get lifecycle status — capacity metrics and configuration for a context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context",
                    },
                },
                "required": ["context_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "re_extract_context",
            "description": (
                "Re-extract bullets from raw input history with a new Reflector model. "
                "Like upgrading a compiler and recompiling from source. "
                "Use dry_run=True to preview changes first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "UUID of the context to re-extract",
                    },
                    "reflector_model": {
                        "type": "string",
                        "description": "New Reflector model to use",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, preview changes without applying (default: true)",
                    },
                },
                "required": ["context_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ingestion_config",
            "description": "Get the server-level ingestion configuration (canonical Reflector model, etc.).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def get_engram_tools() -> list[dict[str, Any]]:
    """Return the tool definitions for use with OpenAI's function calling API."""
    return ENGRAM_TOOLS


class EngramToolHandler:
    """Handles tool calls from an OpenAI agent and routes them to Engram."""

    def __init__(self, engram: Engram, default_agent_id: str = "openai-agent") -> None:
        self.engram = engram
        self.default_agent_id = default_agent_id

    async def handle_tool_call(
        self, function_name: str, arguments: str | dict[str, Any]
    ) -> str:
        """Process a tool call and return the result as a string."""
        if isinstance(arguments, str):
            args = json.loads(arguments)
        else:
            args = arguments

        match function_name:
            case "recall_context":
                result = await self.engram.materialize(
                    context_id=args["context_id"],
                    query=args["query"],
                    token_budget=args.get("token_budget", 2000),
                    target_model="gpt-4o",
                )
                return json.dumps({
                    "context": result.get("rendered_text", ""),
                    "materialization_id": result.get("materialization_id"),
                    "bullets_included": len(result.get("bullets_included", [])),
                })

            case "save_context":
                feedback = None
                if args.get("outcome"):
                    feedback = {"outcome": args["outcome"]}
                result = await self.engram.commit(
                    context_id=args["context_id"],
                    agent_id=args.get("agent_id", self.default_agent_id),
                    content=args["content"],
                    feedback=feedback,
                    materialization_id=args.get("materialization_id"),
                )
                return json.dumps(result)

            case "save_bullet":
                result = await self.engram.add_bullet(
                    context_id=args["context_id"],
                    content=args["content"],
                    section=args.get("section", "general"),
                    bullet_type=args.get("bullet_type", "fact"),
                    agent_id=self.default_agent_id,
                )
                return json.dumps(result)

            case "record_decision":
                result = await self.engram.record_decision(
                    context_id=args["context_id"],
                    decision=args["decision"],
                    rationale=args["rationale"],
                    alternatives_considered=args.get("alternatives_considered", []),
                    agent_id=self.default_agent_id,
                )
                return json.dumps(result)

            case "list_contexts":
                contexts = await self.engram.list_contexts(
                    owner=args.get("owner"),
                )
                return json.dumps(contexts)

            case "get_health":
                health = await self.engram.get_health(
                    context_id=args["context_id"],
                )
                return json.dumps(health)

            case "archive_bullet":
                result = await self.engram.archive_bullet(
                    context_id=args["context_id"],
                    bullet_id=args["bullet_id"],
                    reason=args.get("reason", "manual"),
                )
                return json.dumps(result)

            case "get_lifecycle":
                lifecycle = await self.engram.get_lifecycle(
                    context_id=args["context_id"],
                )
                return json.dumps(lifecycle)

            case "re_extract_context":
                result = await self.engram.re_extract(
                    context_id=args["context_id"],
                    reflector_model=args.get("reflector_model"),
                    dry_run=args.get("dry_run", True),
                )
                return json.dumps(result)

            case "get_ingestion_config":
                config = await self.engram.get_ingestion_config()
                return json.dumps(config)

            case _:
                return json.dumps({"error": f"Unknown function: {function_name}"})
