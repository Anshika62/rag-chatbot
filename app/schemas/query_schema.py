from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_id: str | None = None
    is_new_conv: bool = False