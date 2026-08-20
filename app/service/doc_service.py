from typing import Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.repository import document_repo


def generate_gcs_path(user_id: int, document_id: int, original_filename: str) -> str:
    safe_filename = original_filename.replace("/", "_").strip()
    return f"users/{user_id}/documents/{document_id}/{safe_filename}"


def create_folder_service(
    db: Session, file_name: str, parent_id: Optional[int], user_id: int
) -> Optional[Document]:

    if not parent_id or parent_id <= 0:
        parent_id = None

    if parent_id:
        parent = document_repo.get_owned_folder_by_id(db, parent_id, user_id)
        if not parent:
            return None

    return document_repo.create_folder(db, file_name, parent_id, user_id)


def upload_document_service(
    db: Session,
    file: UploadFile,
    parent_id: Optional[int],
    conversation_id: Optional[int],
    user_id: int,
) -> Optional[Document]:
    
    if not parent_id or parent_id <= 0:
        parent_id = None
    if not conversation_id or conversation_id <= 0:
        conversation_id = None

    if parent_id:
        parent = document_repo.get_owned_folder_by_id(db, parent_id, user_id)
        if not parent:
            return None


    doc = document_repo.create_file(
        db,
        file_name=file.filename,
        parent_id=parent_id,
        user_id=user_id,
        mime_type=file.content_type,
        conversation_id=conversation_id,
    )

   
    gcs_path = generate_gcs_path(user_id, doc.id, file.filename)

    
    contents = file.file.read()

    
    doc = document_repo.update_file_storage_info(
        db,
        doc,
        gcs_path=gcs_path,
        size_bytes=len(contents),
        status=DocumentStatus.PROCESSING,  
    )
    return doc

def rename_document_service(
    db: Session, doc_id: int, user_id: int, new_name: str
) -> Optional[Document]:
    doc = document_repo.get_owned_document_by_id(db, doc_id, user_id)
    if not doc:
        return None
    return document_repo.rename_document(db, doc, new_name)


def delete_document_service(db: Session, doc_id: int, user_id: int) -> bool:
    doc = document_repo.get_owned_document_by_id(db, doc_id, user_id)
    if not doc:
        return False
    _delete_recursive(db, doc)
    return True


def _delete_recursive(db: Session, doc: Document) -> None:
    if doc.is_folder:
        children = document_repo.get_children(db, doc.id)
        for child in children:
            _delete_recursive(db, child)
    else:
        # TODO: GCS delete -> gcs_client.delete(doc.gcs_path)
        # TODO: Vector DB delete -> vector_store.delete(document_id=doc.id)
        pass

    document_repo.delete_document_row(db, doc)