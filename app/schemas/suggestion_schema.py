from pydantic import BaseModel, Field


class SuggestionsResponse(BaseModel):
    suggestions: list[str] = Field(
        description="Exactly 3 concise and relevant follow-up questions"
    )