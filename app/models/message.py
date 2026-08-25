import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id"),
        nullable=False
    )

    role = Column(
        String,
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    # JSON-encoded list of image references
    # (document_id, parent_document_id, filename, url) attached to
    # this message, e.g. images surfaced by the knowledge-base tool.
    # Stored so they survive a page refresh instead of only living
    # in the live SSE stream / frontend state.
    images = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    conversation = relationship(
        "Conversation",
        back_populates="messages"
    )