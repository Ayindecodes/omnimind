import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


def _client() -> OpenAI | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def embed_text(text: str) -> list[float] | None:
    """Return embedding vector or None if no API key / failure."""
    settings = get_settings()
    client = _client()
    if client is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        params: dict = {
            "model": settings.openai_embedding_model,
            "input": text,
        }
        if settings.openai_embedding_dimensions is not None:
            params["dimensions"] = settings.openai_embedding_dimensions
        resp = client.embeddings.create(**params)
        vec = resp.data[0].embedding
        return list(vec)
    except Exception:
        logger.exception("embeddings.create failed")
        return None
