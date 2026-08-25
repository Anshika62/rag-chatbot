from pydantic import BaseModel
from datetime import datetime


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    # Optional: scope the knowledge-base search to a single
    # uploaded document (e.g. "ask about this PDF"). When not
    # provided, search runs across the whole conversation's
    # knowledge base as before.
    document_id: str | None = None


class ConversationTitleUpdate(BaseModel):
    title: str


class ConversationSummary(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    images: list[dict] | None = None
    created_at: datetime

    class Config:
        from_attributes = True