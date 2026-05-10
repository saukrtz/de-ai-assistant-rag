"""
app/agents/tools/pipeline_qa.py
────────────────────────────────
Pipeline Q&A Tool — answers questions about pipeline documentation.

Logical flow:
  1. Accept a natural-language question about DE pipelines.
  2. Retrieve the top-K relevant chunks via the hybrid retriever.
  3. Build a RAG prompt and call the Groq llama-3.1-8b-instant model.
  4. Return the answer plus source citations.
"""

import logging
from groq import Groq

from app.config import settings
from app.rag.retriever import retrieve, format_context

logger = logging.getLogger(__name__)

_client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are a senior Data Engineering assistant specialising in
pipeline architecture, ETL design, and operational runbooks.
Answer questions using ONLY the context provided below.
If the context doesn't contain the answer, say so honestly.
Always cite the source document in your answer using [source: filename]."""


def pipeline_qa(question: str) -> dict:
    """
    Answer a pipeline-related question using RAG.

    Args:
        question: Natural-language question about the DE pipeline.

    Returns:
        dict with 'answer' (str) and 'sources' (list[str]).
    """
    hits = retrieve(question)
    context = format_context(hits)
    sources = list({h["source"].split("/")[-1] for h in hits})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()
    logger.info(f"Pipeline Q&A answered. Sources: {sources}")

    return {"answer": answer, "sources": sources}
