import os
import logging
from typing import Optional

import pdfplumber
from fastapi import UploadFile
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document, DocumentStatus
from app.repository import document_repo
from app.service.rag_clients import embedding_manager, vector_store


logger = logging.getLogger(__name__)

UPLOAD_DIR = "Uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def generate_gcs_path(
    user_id: str,
    document_id: str,
    original_filename: str,
) -> str:
    safe_filename = original_filename.replace("/", "_").strip()

    return (
        f"users/{user_id}/documents/"
        f"{document_id}/{safe_filename}"
    )


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
# UPLOAD DOCUMENT
# ============================================================


def upload_document_service(
    db: Session,
    file: UploadFile,
    parent_id: Optional[str],
    conversation_id: Optional[str],
    user_id: str,
) -> Optional[Document]:

    if not parent_id:
        parent_id = None

    if not conversation_id:
        conversation_id = None

    if parent_id:
        parent = document_repo.get_owned_folder_by_id(
            db,
            parent_id,
            user_id,
        )

        if not parent:
            return None

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

    gcs_path = generate_gcs_path(
        user_id,
        doc.id,
        file.filename,
    )

    # --------------------------------------------------------
    # 3. RAG processing
    # --------------------------------------------------------

    if conversation_id is not None:

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

        except Exception:
            logger.exception(
                "RAG processing failed for document_id=%s",
                doc.id,
            )

            doc = document_repo.update_file_storage_info(
                db=db,
                doc=doc,
                gcs_path=gcs_path,
                size_bytes=len(contents),
                status=DocumentStatus.FAILED,
            )

    else:
        # No conversation attached -> no RAG indexing
        doc = document_repo.update_file_storage_info(
            db=db,
            doc=doc,
            gcs_path=gcs_path,
            size_bytes=len(contents),
            status=DocumentStatus.READY,
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
    Extract text, chunk it, generate embeddings,
    store vectors in Qdrant and chunks in PostgreSQL.
    """

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    if not text.strip():
        logger.warning(
            "No extractable text for document_id=%s, "
            "skipping RAG indexing",
            doc.id,
        )
        return

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
        conversation_id=conversation_id,
        user_id=user_id,
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
        "RAG indexing completed: document_id=%s, chunks=%s",
        doc.id,
        len(chunks),
    )


# ============================================================
# DOCUMENT METADATA UPDATE
# ============================================================


def update_document_service(
    db: Session,
    doc_id: str,
    user_id: str,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Optional[Document]:
    """
    Update document metadata for the current user's document.

    Only supplied fields are changed.
    """

    doc = document_repo.get_owned_document_by_id(
        db,
        doc_id,
        user_id,
    )

    if not doc:
        return None

    # Nothing to update
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
    doc_id: str,
    user_id: str,
) -> bool:

    doc = document_repo.get_owned_document_by_id(
        db,
        doc_id,
        user_id,
    )

    if not doc:
        return False

    _delete_recursive(
        db,
        doc,
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
                db,
                child,
            )

    else:
        # TODO:
        # GCS delete -> gcs_client.delete(doc.gcs_path)

        # TODO:
        # Vector DB delete ->
        # vector_store.delete(document_id=doc.id)

        pass

    document_repo.delete_document_row(
        db,
        doc,
    )