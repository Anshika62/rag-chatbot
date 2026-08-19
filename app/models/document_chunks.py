from sqlalchemy import Column, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class Docs_chunks(Base):
    __tablename__ = "docs_chunks"

    id = Column(Integer, primary_key=True, index=True)

    doc_id = Column(Integer,
        ForeignKey("documents.id"),
        nullable=False
    )
    chunk_text = Column(Text, nullable=False)
    document = relationship(
        "Document",
        back_populates="chunks"
    )