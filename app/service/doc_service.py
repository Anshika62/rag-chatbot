import os
import logging
from typing import Optional

import pdfplumber
from fastapi import UploadFile
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document, DocumentStatus
from app.repository import document_repo
from app.repository.conversation_repo import (
    create_conversation,
    get_conversation,
)
from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)


logger = logging.getLogger(__name__)

UPLOAD_DIR = "Uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# GCS PATH
# ============================================================


def generate_gcs_path(
    user_id: str,
    document_id: str,
    original_filename: str,
) -> str:
    safe_filename = original_filename.replace(
        "/",
        "_",
    ).strip()

    return (
        f"users/{user_id}/documents/"
        f"{document_id}/{safe_filename}"
    )


# ============================================================
# CREATE FOLDER
# ============================================================


def create_folder_service(
    db: Session,
    file_name: str,
    parent_id: Optional[str],
    user_id: str,
) -> Optional[Document]:

    if not parent_id:
        parent_id = None

    if parent_id:

        parent = document_repo.get_owned_folder_by_id(
            db,
            parent_id,
            user_id,
        )

        if not parent:
            return None

    return document_repo.create_folder(
        db,
        file_name,
        parent_id,
        user_id,
    )


# ============================================================
# GET / CREATE CONVERSATION FOR DOCUMENT
# ============================================================


def _get_or_create_document_conversation(
    db: Session,
    conversation_id: Optional[str],
    user_id: str,
) -> str:
    """
    Resolve the conversation that will own the uploaded document.

    Rules:
    1. If conversation_id is not provided:
       create a new conversation for the current user.

    2. If conversation_id is provided:
       verify that the conversation belongs to the current user.

    3. A user can never attach a document to another user's
       conversation.
    """

    # --------------------------------------------------------
    # No conversation_id supplied
    # --------------------------------------------------------

    if not conversation_id:

        conversation = create_conversation(
            db=db,
            user_id=user_id,
            title="Document Upload",
        )

        if not conversation:
            raise RuntimeError(
                "Failed to create conversation for document upload"
            )

        logger.info(
            "Created new conversation=%s for document upload "
            "user_id=%s",
            conversation.id,
            user_id,
        )

        return str(conversation.id)

    # --------------------------------------------------------
    # conversation_id supplied
    # --------------------------------------------------------

    conversation = get_conversation(
        db=db,
        conversation_id=str(conversation_id),
        user_id=user_id,
    )

    if not conversation:
        raise PermissionError(
            "Conversation not found or does not belong "
            "to the current user"
        )

    return str(conversation.id)


# ============================================================
# UPLOAD DOCUMENT
# ============================================================


def upload_document_service(
    db: Session,
    file: UploadFile,
    parent_id: Optional[str],
    conversation_id: Optional[str],
    user_id: str,
) -> Optional[Document]:

    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    if not parent_id:
        parent_id = None

    if not conversation_id:
        conversation_id = None

    # --------------------------------------------------------
    # Validate parent folder ownership
    # --------------------------------------------------------

    if parent_id:

        parent = document_repo.get_owned_folder_by_id(
            db,
            parent_id,
            user_id,
        )

        if not parent:
            return None

    # --------------------------------------------------------
    # Resolve conversation
    #
    # If no conversation_id:
    #     create one automatically for current user.
    #
    # If conversation_id:
    #     verify ownership.
    # --------------------------------------------------------

    try:

        conversation_id = _get_or_create_document_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

    except PermissionError as exc:

        logger.warning(
            "Unauthorized conversation access attempt: "
            "user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )

        raise exc

    # --------------------------------------------------------
    # 1. Create document row
    # --------------------------------------------------------

    doc = document_repo.create_file(
        db=db,
        file_name=file.filename,
        parent_id=parent_id,
        user_id=user_id,
        mime_type=file.content_type,
        conversation_id=conversation_id,
    )

    # --------------------------------------------------------
    # 2. Read bytes and save locally
    # --------------------------------------------------------

    contents = file.file.read()

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{doc.id}_{file.filename}",
    )

    with open(file_path, "wb") as f:
        f.write(contents)

    # --------------------------------------------------------
    # GCS path
    # --------------------------------------------------------

    gcs_path = generate_gcs_path(
        user_id=user_id,
        document_id=doc.id,
        original_filename=file.filename,
    )

    # --------------------------------------------------------
    # 3. RAG processing
    # --------------------------------------------------------

    try:

        _process_for_rag(
            db=db,
            doc=doc,
            file_path=file_path,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        doc = document_repo.update_file_storage_info(
            db=db,
            doc=doc,
            gcs_path=gcs_path,
            size_bytes=len(contents),
            status=DocumentStatus.READY,
        )

        logger.info(
            "Document upload and RAG processing completed: "
            "document_id=%s user_id=%s conversation_id=%s",
            doc.id,
            user_id,
            conversation_id,
        )

    except Exception:

        logger.exception(
            "RAG processing failed for document_id=%s "
            "user_id=%s conversation_id=%s",
            doc.id,
            user_id,
            conversation_id,
        )

        doc = document_repo.update_file_storage_info(
            db=db,
            doc=doc,
            gcs_path=gcs_path,
            size_bytes=len(contents),
            status=DocumentStatus.FAILED,
        )

    return doc


# ============================================================
# RAG PROCESSING
# ============================================================


def _process_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: str,
    user_id: str,
) -> None:
    """
    Extract text, create chunks, generate embeddings,
    store vectors in Qdrant and chunks in PostgreSQL.

    Qdrant payload contains:
        user_id
        conversation_id
        filename
        chunk_index
        text
        type=document
    """

    # --------------------------------------------------------
    # Extract PDF text
    # --------------------------------------------------------

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            text += page.extract_text() or ""

    if not text.strip():

        logger.warning(
            "No extractable text for document_id=%s",
            doc.id,
        )

        return

    # --------------------------------------------------------
    # Text splitting
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_text(text)

    if not chunks:

        logger.warning(
            "No chunks generated for document_id=%s",
            doc.id,
        )

        return

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_manager.generate_embedding(
        chunks
    )

    # --------------------------------------------------------
    # Store vectors in Qdrant
    # --------------------------------------------------------

    vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        filename=doc.file_name,
        conversation_id=str(conversation_id),
        user_id=str(user_id),
    )

    # --------------------------------------------------------
    # Store chunks in PostgreSQL
    # --------------------------------------------------------

    document_repo.create_chunks(
        db=db,
        doc_id=doc.id,
        chunks=chunks,
    )

    logger.info(
        "RAG indexing completed: "
        "document_id=%s chunks=%s user_id=%s "
        "conversation_id=%s",
        doc.id,
        len(chunks),
        user_id,
        conversation_id,
    )


# ============================================================
# DOCUMENT METADATA UPDATE
# ============================================================


def update_document_service(
    db: Session,
    doc: Document,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Document:
    """
    Update document metadata.

    The document has already been fetched and ownership-validated
    by get_current_document() dependency.
    """

    if file_name is None and mime_type is None:
        return doc

    return document_repo.update_document(
        db=db,
        doc=doc,
        file_name=file_name,
        mime_type=mime_type,
    )


# ============================================================
# DELETE
# ============================================================


def delete_document_service(
    db: Session,
    doc: Document,
) -> bool:
    """
    Delete a document or folder tree.

    Ownership has already been validated by
    get_current_document() dependency.
    """

    _delete_recursive(
        db=db,
        doc=doc,
    )

    return True


def _delete_recursive(
    db: Session,
    doc: Document,
) -> None:

    if doc.is_folder:

        children = document_repo.get_children(
            db,
            doc.id,
        )

        for child in children:

            _delete_recursive(
                db=db,
                doc=child,
            )

    else:

        # TODO:
        # GCS delete
        # gcs_client.delete(doc.gcs_path)

        # TODO:
        # Qdrant vector delete
        # vector_store.delete(document_id=doc.id)

        pass

    document_repo.delete_document_row(
        db,
        doc,
    )