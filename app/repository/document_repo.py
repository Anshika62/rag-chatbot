from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.repository import document_repo
from app.core.database import SessionLocal
from app.models.document import Document, DocumentStatus
from app.models.document_chunks import Docs_chunks


def create_chunks(
    db: Session,
    doc_id: str,
    chunks: list[str],
):
    try:
        chunk_objects = [
            Docs_chunks(
                doc_id=doc_id,
                chunk_text=chunk,
            )
            for chunk in chunks
        ]

        db.add_all(chunk_objects)
        db.commit()

        return chunk_objects

    except SQLAlchemyError:
        db.rollback()
        raise


def get_document_by_id(
    db: Session,
    doc_id: str,
):
    try:
        return (
            db.query(Document)
            .filter(Document.id == doc_id)
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_chunks_by_document_id(
    db: Session,
    doc_id: str,
):
    try:
        return (
            db.query(Docs_chunks)
            .filter(Docs_chunks.doc_id == doc_id)
            .all()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# FOLDER / FILE OPERATIONS
# ============================================================


def create_folder(
    db: Session,
    file_name: str,
    parent_id: Optional[str],
    user_id: str,
):
    try:
        folder = Document(
            file_name=file_name,
            parent_id=parent_id,
            is_folder=True,
            user_id=user_id,
        )

        db.add(folder)
        db.commit()
        db.refresh(folder)

        return folder

    except SQLAlchemyError:
        db.rollback()
        raise


def create_file(
    db: Session,
    file_name: str,
    parent_id: Optional[str],
    user_id: str,
    mime_type: Optional[str],
    conversation_id: Optional[str] = None,
):
    try:
        doc = Document(
            file_name=file_name,
            parent_id=parent_id,
            is_folder=False,
            mime_type=mime_type,
            status=DocumentStatus.UPLOADING,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)

        return doc

    except SQLAlchemyError:
        db.rollback()
        raise


def update_file_storage_info(
    db: Session,
    doc: Document,
    gcs_path: str,
    size_bytes: int,
    status: DocumentStatus,
):
    try:
        doc.gcs_path = gcs_path
        doc.size_bytes = size_bytes
        doc.status = status

        db.commit()
        db.refresh(doc)

        return doc

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# OWNERSHIP
# ============================================================


def get_owned_document_by_id(
    db: Session,
    doc_id: str,
    user_id: str,
):
    """
    Fetch document only if it belongs to the current user.
    """

    try:
        return (
            db.query(Document)
            .filter(
                Document.id == doc_id,
                Document.user_id == user_id,
            )
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_owned_folder_by_id(
    db: Session,
    folder_id: str,
    user_id: str,
):
    try:
        return (
            db.query(Document)
            .filter(
                Document.id == folder_id,
                Document.is_folder == True,
                Document.user_id == user_id,
            )
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# LIST DOCUMENTS
# ============================================================


def list_documents_by_parent(
    db: Session,
    parent_id: Optional[str],
    user_id: str,
    skip: int = 0,
    limit: int = 20,
):
    """
    Returns (items, total_count) for the given folder.

    Folders are listed before files,
    both alphabetically.
    """

    try:
        base_query = (
            db.query(Document)
            .filter(
                Document.parent_id == parent_id,
                Document.user_id == user_id,
            )
        )

        total = base_query.count()

        items = (
            base_query
            .order_by(
                Document.is_folder.desc(),
                Document.file_name.asc(),
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

        return items, total

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# CHILDREN
# ============================================================


def get_children(
    db: Session,
    parent_id: str,
):
    """
    Parent ownership is already verified by caller.
    """

    try:
        return (
            db.query(Document)
            .filter(Document.parent_id == parent_id)
            .all()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# DOCUMENT METADATA UPDATE
# ============================================================


def update_document(
    db: Session,
    doc: Document,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
):
    """
    Update document metadata.

    Only fields that are not None are updated.
    Existing values remain unchanged otherwise.

    Note:
    status is intentionally not updated here because
    document status is controlled by the backend/RAG
    processing flow.
    """

    try:
        if file_name is not None:
            doc.file_name = file_name

        if mime_type is not None:
            doc.mime_type = mime_type

        db.commit()
        db.refresh(doc)

        return doc

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# DELETE
# ============================================================


def delete_document_row(
    db: Session,
    doc: Document,
):
    try:
        db.delete(doc)
        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise