import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.chat import ChatMessageCreate, ChatMessageRead, ConversationCreate, ConversationRead
from app.services import chat as chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
def create_conversation(
    body: ConversationCreate,
    db: Session = Depends(get_db),
) -> ConversationRead:
    conv = chat_service.create_conversation(db, body.title)
    return ConversationRead.model_validate(conv)


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ConversationRead:
    conv = chat_service.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return ConversationRead.model_validate(conv)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=list[ChatMessageRead],
    status_code=status.HTTP_201_CREATED,
)
def post_message(
    conversation_id: uuid.UUID,
    body: ChatMessageCreate,
    db: Session = Depends(get_db),
) -> list[ChatMessageRead]:
    try:
        user_msg, assistant_msg = chat_service.append_user_message_and_reply(
            db, conversation_id, body.content
        )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return [
        ChatMessageRead.model_validate(user_msg),
        ChatMessageRead.model_validate(assistant_msg),
    ]


@router.get("/conversations/{conversation_id}/messages", response_model=list[ChatMessageRead])
def list_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[ChatMessageRead]:
    conv = chat_service.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    rows = chat_service.list_messages(db, conversation_id)
    return [ChatMessageRead.model_validate(m) for m in rows]
