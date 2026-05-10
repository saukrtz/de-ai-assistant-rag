"""
app/agents/orchestrator.py
───────────────────────────
LangGraph-based Agent Orchestrator.

Logical flow:
  1. Receive a user message.
  2. Use an intent-routing system prompt to decide which tool(s) to invoke.
  3. Call Groq llama-3.1-8b-instant with tool definitions.
  4. Execute tool(s) and feed results back into the conversation.
  5. Maintain in-session conversation history.
  6. Return the final assistant response.

Tool routing map:
  - Pipeline question  → pipeline_qa()
  - Catalogue/PII      → search_tables() / get_pii_tables()
  - Lineage            → get_lineage()
  - Health / SLO       → get_pipeline_status() / get_recent_failures()
  - Quality check      → run_quality_check()
"""

import json
import logging
from groq import Groq
from openai import OpenAI   # Ollama uses OpenAI-compatible API

from app.config import settings
from app.agents.tools.pipeline_qa import pipeline_qa
from app.agents.tools.catalog_explorer import (
    search_tables,
    get_lineage,
    get_pii_tables,
    get_table_by_name,
)
from app.agents.tools.health_monitor import (
    get_pipeline_status,
    get_recent_failures,
    calculate_slo_adherence,
    get_failure_rate,
)
from app.agents.tools.quality_checker import run_quality_check

logger = logging.getLogger(__name__)

# ── LLM client factory (Groq or Ollama) ──────────────────────────────────────
def _make_client():
    if settings.llm_backend == "ollama":
        # Ollama exposes OpenAI-compatible API — no auth needed
        return OpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",   # required by openai SDK but ignored by Ollama
        ), settings.ollama_model
    else:
        return Groq(api_key=settings.groq_api_key), settings.groq_model

_client, _model = _make_client()

SYSTEM_PROMPT = """You are a senior Data Engineering Assistant.
You have access to the following tools to help answer questions:

- pipeline_qa: Answer questions about pipeline documentation and code.
- search_tables: Find tables by name, column, PII tag, or owner.
- get_lineage: Show upstream/downstream table lineage.
- get_pii_tables: List all tables containing PII data.
- get_pipeline_status: Show current pipeline run statuses.
- get_recent_failures: Show pipelines that failed recently.
- calculate_slo_adherence: Show SLO compliance per pipeline.
- run_quality_check: Trigger an on-demand data quality scan for a table.

Use the tools to provide accurate, grounded answers.
Always explain which tool you used and why."""

# ── Tool definitions for Groq function-calling ────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pipeline_qa",
            "description": "Answer a question about pipeline documentation using RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The pipeline question to answer."}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "Search the data catalogue by table name, column, PII tag, or owner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lineage",
            "description": "Get upstream or downstream lineage for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["upstream", "downstream", "both"]},
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pii_tables",
            "description": "List all tables with PII columns.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": "Get the current status of all pipelines.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_failures",
            "description": "Get pipelines that failed in the last N hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "default": 24}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_slo_adherence",
            "description": "Calculate SLO adherence for all pipelines.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_quality_check",
            "description": "Trigger an on-demand data quality check for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table to check."}
                },
                "required": ["table_name"],
            },
        },
    },
]

# ── Tool dispatch map ─────────────────────────────────────────────────────────
_TOOL_MAP = {
    "pipeline_qa": pipeline_qa,
    "search_tables": search_tables,
    "get_lineage": get_lineage,
    "get_pii_tables": get_pii_tables,
    "get_pipeline_status": get_pipeline_status,
    "get_recent_failures": get_recent_failures,
    "calculate_slo_adherence": calculate_slo_adherence,
    "run_quality_check": run_quality_check,
}


class Orchestrator:
    """Stateful orchestrator maintaining in-session conversation history."""

    def __init__(self):
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_calls_log: list[str] = []

    def chat(self, user_message: str) -> dict:
        """
        Process one user turn.

        Returns:
            dict with 'response' (str), 'tool_used' (str|None),
            'tool_result' (any), 'sources' (list[str]).
        """
        self.history.append({"role": "user", "content": user_message})

        # ── Trim history: Ollama = 20 turns (local, unlimited); Groq = 6 turns (rate-limited)
        system_msg = self.history[0]
        recent = self.history[1:]
        max_turns = 40 if settings.llm_backend == "ollama" else 12  # messages not turns
        if len(recent) > max_turns:
            recent = recent[-max_turns:]
        trimmed_history = [system_msg] + recent

        # Ollama local = no rate limits; Groq = keep tokens small
        max_tok = 2048 if settings.llm_backend == "ollama" else 512

        response = _client.chat.completions.create(
            model=_model,
            messages=trimmed_history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=max_tok,
        )

        msg = response.choices[0].message
        tool_used = None
        tool_result = None
        sources = []

        # ── Handle tool calls ─────────────────────────────────────────────────
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]  # Execute first tool
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_used = tool_name

            logger.info(f"Tool called: {tool_name}({tool_args})")
            self.tool_calls_log.append(tool_name)

            fn = _TOOL_MAP.get(tool_name)
            if fn:
                tool_result = fn(**tool_args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            # Extract sources if pipeline_qa was called
            if tool_name == "pipeline_qa" and isinstance(tool_result, dict):
                sources = tool_result.get("sources", [])

            # Feed tool result back for final response
            self.history.append(msg)
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, default=str),
                }
            )

            # Re-trim after appending tool result
            recent2 = self.history[1:]
            max_turns2 = 42 if settings.llm_backend == "ollama" else 14
            if len(recent2) > max_turns2:
                recent2 = recent2[-max_turns2:]
            trimmed2 = [self.history[0]] + recent2

            final = _client.chat.completions.create(
                model=_model,
                messages=trimmed2,
                temperature=0.3,
                max_tokens=max_tok,
            )
            final_content = final.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": final_content})
        else:
            final_content = msg.content.strip() if msg.content else ""
            self.history.append({"role": "assistant", "content": final_content})

        return {
            "response": final_content,
            "tool_used": tool_used,
            "tool_result": tool_result,
            "sources": sources,
        }

    def reset(self):
        """Reset conversation history (new session)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_calls_log = []
