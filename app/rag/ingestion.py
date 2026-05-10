"""
app/rag/ingestion.py
─────────────────────
Document ingestion pipeline.

Logical flow:
  1. Walk `data/pipeline_docs/` and load every Markdown file.
  2. Split each document into overlapping chunks using
     RecursiveCharacterTextSplitter (chunk_size=500, overlap=50).
  3. Embed each chunk with sentence-transformers (all-MiniLM-L6-v2).
  4. Upsert chunks into ChromaDB with source + section metadata.

Run from project root:
    python -m app.rag.ingestion
"""

import os
import hashlib
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.vectorstore import (
    get_chroma_client,
    get_or_create_collection,
    upsert_documents,
)

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ── Embedding model (loaded once) ──────────────────────────────────────────────
_embedder = SentenceTransformer(settings.embedding_model)

# ── Text splitter ─────────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _stable_id(source: str, chunk_index: int) -> str:
    """Generate a stable, deterministic chunk ID."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_markdown_files(docs_dir: Path) -> list[dict]:
    """
    Recursively load all .md files from docs_dir.
    Returns a list of {'source': str, 'content': str} dicts.
    """
    docs = []
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": str(path), "content": text})
        logger.info(f"Loaded: {path} ({len(text)} chars)")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split documents into overlapping chunks.
    Returns list of {'chunk_id', 'source', 'content'} dicts.
    """
    chunks = []
    for doc in docs:
        pieces = _splitter.split_text(doc["content"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": _stable_id(doc["source"], i),
                    "source": doc["source"],
                    "content": piece,
                }
            )
    logger.info(f"Total chunks produced: {len(chunks)}")
    return chunks


def embed_and_ingest(chunks: list[dict]) -> int:
    """
    Embed chunks and upsert into ChromaDB.
    Returns the number of chunks ingested.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    texts = [c["content"] for c in chunks]
    embeddings = _embedder.encode(texts, show_progress_bar=True).tolist()

    upsert_documents(
        collection=collection,
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    logger.info(f"Ingested {len(chunks)} chunks into ChromaDB.")
    return len(chunks)


def run_ingestion(docs_dir: Path | None = None) -> int:
    """End-to-end ingestion: load → chunk → embed → upsert."""
    docs_dir = docs_dir or settings.pipeline_docs_dir
    docs = load_markdown_files(docs_dir)
    if not docs:
        logger.warning(f"No Markdown files found in {docs_dir}")
        return 0
    chunks = chunk_documents(docs)
    return embed_and_ingest(chunks)


if __name__ == "__main__":
    total = run_ingestion()
    print(f"✅ Ingestion complete — {total} chunks stored.")
