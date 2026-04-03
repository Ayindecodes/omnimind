import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DecisionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID | None
    assistant_message_id: uuid.UUID | None
    action: str
    confidence: float
    rules_fired: list[Any]
    context: dict[str, Any]
    created_at: datetime
