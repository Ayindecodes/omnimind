import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Conversation, DecisionRecord, Message, MessageEmbedding
from app.services import embeddings as embedding_service
from app.services import llm as llm_service
from app.services.decision_engine import evaluate_turn
from app.services.memory_retrieval import retrieve_similar_messages

logger = logging.getLogger(__name__)


def _persist_embedding(db: Session, message: Message, vec: list[float] | None) -> None:
    if vec is None:
        return
    settings = get_settings()
    row = MessageEmbedding(
        message_id=message.id,
        model=settings.openai_embedding_model,
        dimensions=len(vec),
        vector=vec,
    )
    db.add(row)


def _recent_messages(db: Session, conversation_id: uuid.UUID, limit: int) -> list[Message]:
    lim = max(1, limit)
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(lim)
    )
    rows = list(db.scalars(stmt).all())
    rows.reverse()
    return rows


def _build_llm_messages(
    memory_snippets: list[tuple[Message, float]],
    recent: list[Message],
) -> list[dict]:
    lines: list[str] = []
    for m, score in memory_snippets:
        excerpt = m.content.replace("\n", " ").strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        lines.append(f"- ({m.role}, relevance={score:.3f}): {excerpt}")
    memory_block = "\n".join(lines) if lines else "(none)"

    system = (
        "You are OmniMind, a concise and accurate assistant.\n\n"
        "MEMORY SNIPPETS (semantically related prior messages in this conversation):\n"
        f"{memory_block}\n\n"
        "If snippets conflict with the recent dialogue, trust the recent messages."
    )
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in recent:
        if msg.role not in ("user", "assistant", "system"):
            continue
        out.append({"role": msg.role, "content": msg.content})
    return out


def create_conversation(db: Session, title: str | None) -> Conversation:
    conv = Conversation(title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    logger.info("conversation created id=%s", conv.id)
    return conv


def get_conversation(db: Session, conversation_id: uuid.UUID) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def append_user_message_and_reply(
    db: Session, conversation_id: uuid.UUID, user_content: str
) -> tuple[Message, Message]:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError("conversation_not_found")

    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=user_content.strip(),
    )
    db.add(user_msg)
    db.flush()

    user_vec = embedding_service.embed_text(user_content)
    _persist_embedding(db, user_msg, user_vec)
    db.commit()
    db.refresh(user_msg)

    memory_pairs: list[tuple[Message, float]] = []
    if user_vec is not None:
        memory_pairs = retrieve_similar_messages(
            db,
            conversation_id,
            user_vec,
            exclude_message_ids={user_msg.id},
        )

    settings = get_settings()
    recent = _recent_messages(db, conversation_id, settings.chat_history_limit)
    llm_messages = _build_llm_messages(memory_pairs, recent)

    raw_reply = llm_service.chat_complete(llm_messages)
    llm_succeeded = raw_reply is not None
    if raw_reply is None:
        assistant_text = (
            f"[OmniMind] No LLM configured or generation failed. "
            f"Heard: {user_content[:200]}{'…' if len(user_content) > 200 else ''}"
        )
    else:
        assistant_text = raw_reply

    turn = evaluate_turn(
        user_content=user_content,
        memory_hit_count=len(memory_pairs),
        llm_succeeded=llm_succeeded,
        user_embedding_ok=user_vec is not None,
    )

    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_text,
    )
    db.add(assistant_msg)
    db.flush()

    asst_vec = embedding_service.embed_text(assistant_text)
    _persist_embedding(db, assistant_msg, asst_vec)

    db.add(
        DecisionRecord(
            conversation_id=conversation_id,
            user_message_id=user_msg.id,
            assistant_message_id=assistant_msg.id,
            action=turn.action,
            confidence=turn.confidence,
            rules_fired=list(turn.rules_fired),
            context=dict(turn.context),
        )
    )
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    logger.info(
        "chat turn conversation_id=%s user_message_id=%s assistant_message_id=%s",
        conversation_id,
        user_msg.id,
        assistant_msg.id,
    )
    return user_msg, assistant_msg


def list_messages(db: Session, conversation_id: uuid.UUID) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list(db.scalars(stmt).all())
