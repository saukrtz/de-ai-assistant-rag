"""
app/rag/corrective_rag.py
──────────────────────────
Corrective RAG (CRAG) pipeline.

Full logical flow:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Query                                                          │
  │   │                                                             │
  │   ▼                                                             │
  │  [1] Hybrid Retrieve  (Dense + BM25 via RRF)                   │
  │   │                                                             │
  │   ▼                                                             │
  │  [2] Grade each chunk (LLM relevance grader)                   │
  │   │                                                             │
  │   ├─ ALL IRRELEVANT ────► [3a] Rewrite query + re-retrieve     │
  │   │                            → grade again                   │
  │   │                            → return best available         │
  │   │                                                             │
  │   ├─ SOME AMBIGUOUS ───► [3b] Keep RELEVANT, discard rest      │
  │   │                            (optionally supplement with      │
  │   │                             re-retrieved on rewritten q)   │
  │   │                                                             │
  │   └─ HAS RELEVANT ─────► [4] Format context for LLM           │
  │                                                                 │
  │   ▼                                                             │
  │  Return: { context, sources, retrieval_trace }                  │
  └─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.rag.hybrid_retriever import hybrid_retrieve, format_context
from app.rag.grader import grade_hits

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CRAGResult:
    """Output of the CRAG pipeline."""
    context: str                         # Formatted text for the LLM prompt
    sources: list[str]                   # Deduplicated source file names
    graded_hits: list[dict]              # All hits with their grade attached
    rewritten_query: Optional[str] = None  # Set if query was rewritten
    fallback_used: bool = False          # True if no relevant docs were found


# ── Query rewriter ────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
Rewrite the following question to improve document retrieval.
Make it more specific and use different keywords.
Return ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:"""


def _rewrite_query(question: str) -> str:
    """Ask the LLM to rewrite the query for better retrieval coverage."""
    try:
        from app.agents.orchestrator import _client, _model
        resp = _client.chat.completions.create(
            model=_model,
            messages=[{
                "role": "user",
                "content": _REWRITE_PROMPT.format(question=question),
            }],
            max_tokens=80,
            temperature=0.3,
        )
        rewritten = resp.choices[0].message.content.strip()
        logger.info("Query rewritten: %r → %r", question, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed: %s — using original", exc)
        return question


# ── Main CRAG pipeline ────────────────────────────────────────────────────────

def corrective_retrieve(
    query: str,
    top_k: int = 5,
    grade_threshold_relevant: int = 1,   # min RELEVANT hits before accepting
    enable_rewrite: bool = True,
) -> CRAGResult:
    """
    Run the full Hybrid + Corrective RAG pipeline.

    Args:
        query:                    User question.
        top_k:                    Number of docs to retrieve and grade.
        grade_threshold_relevant: How many RELEVANT hits are needed before
                                  skipping the correction step.
        enable_rewrite:           Whether to rewrite the query when all docs
                                  are irrelevant.

    Returns:
        CRAGResult with context, sources, graded_hits.
    """
    rewritten_query: Optional[str] = None
    fallback_used = False

    # ── Step 1: Hybrid retrieve ───────────────────────────────────────────────
    hits = hybrid_retrieve(query, top_k=top_k)
    logger.info("CRAG step 1 — hybrid retrieved %d hits", len(hits))

    if not hits:
        logger.warning("CRAG: no documents in vector store")
        return CRAGResult(
            context="No relevant documentation found.",
            sources=[],
            graded_hits=[],
            fallback_used=True,
        )

    # ── Step 2: Grade each hit ────────────────────────────────────────────────
    graded = grade_hits(query, hits)
    relevant  = [h for h in graded if h["grade"] == "RELEVANT"]
    ambiguous = [h for h in graded if h["grade"] == "AMBIGUOUS"]
    irrelevant = [h for h in graded if h["grade"] == "IRRELEVANT"]

    logger.info(
        "CRAG step 2 — grades: %d relevant, %d ambiguous, %d irrelevant",
        len(relevant), len(ambiguous), len(irrelevant),
    )

    # ── Step 3: Corrective actions ────────────────────────────────────────────
    if len(relevant) < grade_threshold_relevant and enable_rewrite:
        # 3a: All/mostly irrelevant → rewrite and re-retrieve
        rewritten_query = _rewrite_query(query)
        new_hits = hybrid_retrieve(rewritten_query, top_k=top_k)
        new_graded = grade_hits(rewritten_query, new_hits)
        new_relevant = [h for h in new_graded if h["grade"] == "RELEVANT"]

        if new_relevant:
            logger.info(
                "CRAG step 3a — re-retrieve found %d relevant hits", len(new_relevant)
            )
            # Merge: prefer new relevant, supplement with original relevant
            final_hits = new_relevant + relevant
        else:
            # Last resort: use ambiguous from both rounds
            logger.warning("CRAG step 3a — still no relevant hits after rewrite")
            final_hits = new_graded + ambiguous
            fallback_used = True

        graded = new_graded  # update trace with new grading
    else:
        # 3b: We have enough relevant hits — keep them (discard IRRELEVANT)
        final_hits = relevant if relevant else ambiguous

    # Deduplicate by content prefix
    seen: set[str] = set()
    deduped: list[dict] = []
    for h in final_hits:
        key = h["content"][:120]
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    final_hits = deduped[:top_k]

    # ── Step 4: Format output ─────────────────────────────────────────────────
    context = format_context(final_hits) if final_hits else "No relevant documentation found."
    sources = list({h["source"].split("/")[-1] for h in final_hits})

    return CRAGResult(
        context=context,
        sources=sources,
        graded_hits=graded,
        rewritten_query=rewritten_query,
        fallback_used=fallback_used,
    )
