from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    file_name: str


# ---- Folder create ----
class FolderCreate(BaseModel):
    file_name: str
    parent_id: Optional[str] = None


# ---- Document metadata update ----
class DocumentUpdate(BaseModel):
    file_name: Optional[str] = None
    mime_type: Optional[str] = None


# ---- Response: single document/folder ----
class DocumentOut(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    parent_id: Optional[str]
    is_folder: bool
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    status: DocumentStatus
    conversation_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ---- Response: listing ----
class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int
    page: int
    page_size: int
    total_pages: int