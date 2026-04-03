import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Conversation, DecisionRecord
from app.schemas.decision import DecisionRead

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.get("/conversations/{conversation_id}", response_model=list[DecisionRead])
def list_decisions_for_conversation(
    conversation_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[DecisionRead]:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    stmt = (
        select(DecisionRecord)
        .where(DecisionRecord.conversation_id == conversation_id)
        .order_by(DecisionRecord.created_at.desc())
        .limit(limit)
    )
    rows = list(db.scalars(stmt).all())
    return [DecisionRead.model_validate(r) for r in rows]
