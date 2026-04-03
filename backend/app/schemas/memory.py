import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MemorySearchRequest(BaseModel):
    conversation_id: uuid.UUID
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class MemoryHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: uuid.UUID
    role: str
    content: str
    score: float
    created_at: datetime
