import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    user_id = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False
    )

    title = Column(
        String,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Conversation belongs to one user
    user = relationship(
        "User",
        back_populates="conversations"
    )

    # Conversation has many messages
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

    # Conversation has many documents
    documents = relationship(
        "Document",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )