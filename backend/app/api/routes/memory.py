from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Conversation
from app.schemas.memory import MemoryHit, MemorySearchRequest
from app.services import embeddings as embedding_service
from app.services.memory_retrieval import retrieve_similar_messages

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=list[MemoryHit])
def memory_search(
    body: MemorySearchRequest,
    db: Session = Depends(get_db),
) -> list[MemoryHit]:
    conv = db.get(Conversation, body.conversation_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    qvec = embedding_service.embed_text(body.query)
    if qvec is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings unavailable (set OPENAI_API_KEY or check provider)",
        )

    pairs = retrieve_similar_messages(
        db,
        body.conversation_id,
        qvec,
        exclude_message_ids=None,
        top_k=body.top_k,
    )
    return [
        MemoryHit(
            message_id=m.id,
            role=m.role,
            content=m.content,
            score=score,
            created_at=m.created_at,
        )
        for m, score in pairs
    ]
