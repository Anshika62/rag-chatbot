
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    file_name: str


# ---- Folder create ----
class FolderCreate(BaseModel):
    file_name: str                     
    parent_id: Optional[int] = None    


class DocumentRename(BaseModel):
    file_name: str


# ---- Response: single document/folder ----
class DocumentOut(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    parent_id: Optional[int]
    is_folder: bool
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    status: DocumentStatus
    conversation_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


# ---- Response: listing 
class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int