import uuid

from sqlalchemy import Column, String, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Docs_chunks(Base):
    __tablename__ = "docs_chunks"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    doc_id = Column(
        String(36),
        ForeignKey("documents.id"),
        nullable=False
    )
    chunk_text = Column(Text, nullable=False)
    document = relationship(
        "Document",
        back_populates="chunks"
    )