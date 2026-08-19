from pydantic import BaseModel
from datetime import datetime


class QueryRequest(BaseModel):
    question: str
    conversation_id: int | None = None


class ConversationTitleUpdate(BaseModel):
    title: str


class ConversationSummary(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True