import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Float,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Conversation(Base):

    __tablename__ = "conversations"

    # ============================================================
    # PRIMARY KEY
    # ============================================================

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    # ============================================================
    # USER
    # ============================================================

    user_id = Column(
        String(36),
        ForeignKey("users.id"),
        nullable=False,
    )

    # ============================================================
    # CONVERSATION INFO
    # ============================================================

    title = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # ============================================================
    # CONVERSATION LOCATION
    # ============================================================
    #
    # Location is persisted per conversation.
    #
    # First location request:
    #     latitude/longitude received
    #     -> saved here
    #
    # Later requests:
    #     no coordinates
    #     -> existing coordinates loaded from here
    #
    # ============================================================

    latitude = Column(
        Float,
        nullable=True,
    )

    longitude = Column(
        Float,
        nullable=True,
    )

    # ============================================================
    # RELATIONSHIPS
    # ============================================================

    # Conversation belongs to one user
    user = relationship(
        "User",
        back_populates="conversations",
    )

    # Conversation has many messages
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    # Conversation has many documents
    documents = relationship(
        "Document",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )