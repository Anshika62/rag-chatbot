from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    is_new_conv: bool = False
    document_id: Optional[str] = None

    # Location is supplied only when user shares/updates location.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: str | None = None

    # NEW: conversation.py's _normalize_request_location() reads
    # request.full_address, request.location, and request.coordinates,
    # but only request.address was declared here. That caused
    # AttributeError whenever those code paths were hit. Added only
    # to fix that; nothing else changed.
    full_address: Optional[str] = None
    location: Optional[dict] = None
    coordinates: Optional[dict] = None

    model_config = {"extra": "ignore"}