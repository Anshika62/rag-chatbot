from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    file_name = Column(
        String,
        nullable=False
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False
    )

    # Document belongs to one conversation
    conversation = relationship(
        "Conversation",
        back_populates="documents"
    )

    # Document has many chunks
    chunks = relationship(
        "Docs_chunks",
        back_populates="document",
        cascade="all, delete-orphan"
    )