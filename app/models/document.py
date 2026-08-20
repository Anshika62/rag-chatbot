import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    BigInteger,
    ForeignKey,
    DateTime,
    Enum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    UPLOADING = "uploading"    
    PROCESSING = "processing"  
    READY = "ready"            
    FAILED = "failed"          


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    parent_id = Column(
        Integer,
        ForeignKey("documents.id"),
        nullable=True,   
        index=True,
    )

    is_folder = Column(Boolean, nullable=False, default=False)

    file_name = Column(String, nullable=False)

    gcs_path = Column(String(1024), nullable=True, unique=True)

    mime_type = Column(String(128), nullable=True)
    size_bytes = Column(BigInteger, nullable=True)

    status = Column(
        Enum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.UPLOADING,
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=True,   
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)  

    parent = relationship(
        "Document",
        remote_side=[id],
        back_populates="children",
    )
    children = relationship(
        "Document",
        back_populates="parent",
        cascade="all, delete-orphan",
    )


    conversation = relationship(
        "Conversation",
        back_populates="documents",
    )

    chunks = relationship(
        "Docs_chunks",
        back_populates="document",
        cascade="all, delete-orphan",
    )