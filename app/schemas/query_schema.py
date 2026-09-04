from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_id: str | None = None
    is_new_conv: bool = False

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="User's current latitude, if already known.",
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description="User's current longitude, if already known.",
    )

    address: str | None = Field(
        default=None,
        description="Optional human-readable address for the "
        "given latitude/longitude.",
    )