import logging
from typing import Any

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


def chat_complete(messages: list[dict[str, Any]]) -> str | None:
    """Return assistant text or None if unavailable / error."""
    settings = get_settings()
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.7,
        )
        choice = resp.choices[0].message
        if choice.content:
            return choice.content.strip()
        return None
    except Exception:
        logger.exception("chat.completions failed")
        return None
