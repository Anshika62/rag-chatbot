from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    conversation_id: int | None = None