"""
Application configuration loaded from environment variables via pydantic-settings.
All secrets are injected at runtime — never hard-coded here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    openrouter_api_key: str

    # ── Model identifiers ─────────────────────────────────────────────────────
    # OpenRouter routes this to Gemini 2.5 Flash
    chat_model: str = "google/gemini-2.5-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Local sentence-transformers model — no API key required
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Re-ranker ─────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5       # chunks fed to LLM after re-ranking
    retrieval_top_k: int = 15     # initial FAISS retrieval count

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Storage ───────────────────────────────────────────────────────────────
    faiss_index_path: str = "./faiss_index"
    eval_log_path: str = "./logs/ragas_eval.jsonl"

    # ── Server ────────────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings: # singleton pattern
    """Cached singleton — settings object is created once per process."""
    return Settings()