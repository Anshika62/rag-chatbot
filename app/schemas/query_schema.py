from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    is_new_conv: bool = False
    document_id: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    full_address: Optional[str] = None


    location: Optional[dict] = None
    coordinates: Optional[dict] = None

    model_config = {"extra": "ignore"}