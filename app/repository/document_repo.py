from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

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


def delete_chunks_by_document_id(
    db: Session,
    doc_id: str,
):
    try:
        db.query(Docs_chunks).filter(
            Docs_chunks.doc_id == str(doc_id)
        ).delete(
            synchronize_session=False
        )

        db.commit()

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
            conversation_id=None,
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
    page_number: Optional[int] = None,
):
    """
    page_number:
        1-based page/slide number for a document extracted from a
        paginated parent (a PDF page image, a PPTX slide image).
        None for top-level documents and standalone uploads.
    """
    try:
        doc = Document(
            file_name=file_name,
            parent_id=parent_id,
            is_folder=False,
            mime_type=mime_type,
            status=DocumentStatus.UPLOADING,
            conversation_id=conversation_id,
            user_id=user_id,
            page_number=page_number,
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


def update_document_status(
    db: Session,
    doc: Document,
    status: DocumentStatus,
):
    """
    Update only the status field of a document.

    Added specifically for the streamed-upload flow so the
    document can be marked PROCESSING as soon as the file is
    saved and RAG indexing begins (before gcs_path/size_bytes
    are known, which update_file_storage_info requires).
    """
    try:
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
    Fetch a document only if it belongs to the current user.

    This is the basic ownership check and does not apply
    conversation-scope rules. Prefer get_accessible_document_by_id
    for any lookup that should also respect conversation scoping
    (i.e. almost every LLM-tool-triggered lookup).
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


def get_accessible_document_by_id(
    db: Session,
    doc_id: str,
    user_id: str,
    conversation_id: Optional[str] = None,
):
    """
    Fetch a document that the current user is allowed to access.

    Access rules:

    1. The document must belong to the current user.
    2. Global documents (conversation_id IS NULL) are accessible
       from every conversation.
    3. Conversation documents are accessible only from their
       owning conversation.

    If conversation_id is None, only global documents are returned.
    """

    try:
        query = (
            db.query(Document)
            .filter(
                Document.id == str(doc_id),
                Document.user_id == str(user_id),
            )
        )

        if conversation_id is None:
            query = query.filter(
                Document.conversation_id.is_(None)
            )
        else:
            query = query.filter(
                (
                    Document.conversation_id.is_(None)
                )
                |
                (
                    Document.conversation_id
                    == str(conversation_id)
                )
            )

        return query.first()

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
                Document.conversation_id.is_(None),
            )
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


# ============================================================
# LIST GLOBAL DOCUMENTS
# ============================================================


def list_documents_by_parent(
    db: Session,
    parent_id: Optional[str],
    user_id: str,
    skip: int = 0,
    limit: int = 20,
):
    """
    Returns global documents for the given folder.

    The Document Folder represents the user's global
    document space, therefore only documents with:

        conversation_id IS NULL

    are returned.

    Folders are listed before files,
    both alphabetically.
    """

    try:
        base_query = (
            db.query(Document)
            .filter(
                Document.parent_id == parent_id,
                Document.user_id == user_id,
                Document.conversation_id.is_(None),
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
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    page_number: Optional[int] = None,
):
    """
    Return children belonging to the specified parent.

    If conversation_id is provided:
        - global children are allowed
        - children belonging to the current conversation are allowed

    If conversation_id is None:
        - only global children are returned

    If user_id is provided:
        - only children owned by that user are returned.

    If page_number is provided:
        - only children with that exact page_number are returned.
    """

    try:
        query = (
            db.query(Document)
            .filter(
                Document.parent_id == parent_id,
            )
        )

        if user_id is not None:
            query = query.filter(
                Document.user_id == str(user_id),
            )

        if page_number is not None:
            query = query.filter(
                Document.page_number == page_number,
            )

        if conversation_id is None:
            query = query.filter(
                Document.conversation_id.is_(None)
            )
        else:
            query = query.filter(
                (
                    Document.conversation_id.is_(None)
                )
                |
                (
                    Document.conversation_id
                    == str(conversation_id)
                )
            )

        return query.all()

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
        # Delete all DB chunks belonging to this document
        # before deleting the document row.
        delete_chunks_by_document_id(
            db=db,
            doc_id=doc.id,
        )

        db.delete(doc)
        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise

