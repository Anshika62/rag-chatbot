from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None