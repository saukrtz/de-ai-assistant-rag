"""
app/config.py
─────────────
Centralised configuration using Pydantic Settings v2.
All environment variables are validated here and
exposed as a single `settings` singleton used across the project.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings — loaded from .env (Pydantic v2 style)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Groq / LLM ────────────────────────────────────────────────────────────
    # Field names must match env var names (case-insensitive)
    groq_api_key: str = "not-set"   # Optional when using Ollama
    groq_model: str = "llama-3.1-8b-instant"

    # ── LLM Backend: 'groq' or 'ollama' ──────────────────────────────────────
    llm_backend: str = "groq"               # switch to 'ollama' for local
    ollama_base_url: str = "http://localhost:11434"  # default Ollama port
    ollama_model: str = "llama3.1:8b"

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "pipeline_docs"

    # ── Data paths ────────────────────────────────────────────────────────────
    pipeline_docs_dir: Path = Path("data/pipeline_docs")
    catalogue_dir: Path = Path("data/catalogue")
    health_dir: Path = Path("data/health")

    # ── RAG settings ──────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "DE-AI Assistant"
    log_level: str = "INFO"


# Singleton
settings = Settings()
