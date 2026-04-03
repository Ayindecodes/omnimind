from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Message, MessageEmbedding


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def retrieve_similar_messages(
    db: Session,
    conversation_id: uuid.UUID,
    query_embedding: list[float],
    *,
    exclude_message_ids: set[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[tuple[Message, float]]:
    """
    Cosine similarity over stored message embeddings in one conversation.
    Returns (message, score) sorted by score descending.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.memory_top_k
    exclude_message_ids = exclude_message_ids or set()

    q = np.array(query_embedding, dtype=np.float64)
    q = _normalize(q)

    stmt = (
        select(Message, MessageEmbedding)
        .join(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .where(Message.conversation_id == conversation_id)
    )
    rows = db.execute(stmt).all()

    scored: list[tuple[Message, float]] = []
    for msg, emb in rows:
        if msg.id in exclude_message_ids:
            continue
        vec = np.array(emb.vector, dtype=np.float64)
        vec = _normalize(vec)
        score = float(np.dot(q, vec))
        scored.append((msg, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
