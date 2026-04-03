from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "OmniMind API"
    debug: bool = False
    # Set True on first local run to create tables (replace with Alembic migrations later).
    bootstrap_schema: bool = False
    database_url: str = "postgresql://omnimind:omnimind@localhost:5432/omnimind"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" | "json"

    # Model gateway (optional — without OPENAI_API_KEY, chat uses placeholder replies)
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # e.g. Azure or compatible proxies
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Truncate embeddings (OpenAI 3 models support reduced dims); None = API default
    openai_embedding_dimensions: int | None = None
    memory_top_k: int = 6
    chat_history_limit: int = 40


@lru_cache
def get_settings() -> Settings:
    return Settings()
