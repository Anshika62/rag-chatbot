from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_id: str | None = None
    attachment_document_ids: list[str] | None = None