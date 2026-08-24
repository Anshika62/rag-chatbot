
import os
import logging
from typing import Optional
from PIL import Image
import fitz  # PyMuPDF
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
from app.service.tools.image_tool import generate_image_caption


logger = logging.getLogger(__name__)

UPLOAD_DIR = "Uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
}


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

    doc = document_repo.create_file(
        db=db,
        file_name=file.filename,
        parent_id=parent_id,
        user_id=user_id,
        mime_type=file.content_type,
        conversation_id=conversation_id,
    )

    contents = file.file.read()

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{doc.id}_{file.filename}",
    )

    with open(file_path, "wb") as f:
        f.write(contents)

    gcs_path = generate_gcs_path(
        user_id=user_id,
        document_id=doc.id,
        original_filename=file.filename,
    )

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
# RELIABLE FILE-TYPE DETECTION
#
# Client-supplied Content-Type header can be wrong/missing
# depending on the uploading tool (Swagger, curl, browsers).
# So we sniff actual file bytes first, and only fall back to
# mime_type / filename extension if sniffing is inconclusive.
# ============================================================

def _sniff_file_type(file_path: str) -> str:
    """
    Returns "image", "pdf", or "unknown" based on the file's
    actual content — PDF is checked via header bytes, and any
    image format (jpg/png/webp/gif/avif/heic/bmp/etc.) is
    checked by attempting to open it with Pillow. This avoids
    maintaining a manual list of image magic-byte signatures.
    """

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)

        if header.startswith(b"%PDF"):
            return "pdf"

    except Exception:
        return "unknown"

    try:
        with Image.open(file_path) as img:
            img.verify()

        return "image"

    except Exception:
        return "unknown"

def _detect_file_type(
    file_path: str,
    mime_type: Optional[str],
    file_name: str,
) -> str:
    """
    Determine whether the uploaded file should be processed
    as an "image" or a "pdf". Tries, in order:

    1. Binary signature sniffing (most reliable)
    2. Content-Type header sent by the client
    3. File extension
    """

    file_type = _sniff_file_type(file_path)

    if file_type != "unknown":
        return file_type

    if mime_type in IMAGE_MIME_TYPES:
        return "image"

    if mime_type == "application/pdf":
        return "pdf"

    extension = os.path.splitext(file_name or "")[1].lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension == ".pdf":
        return "pdf"

    return "unknown"


# ============================================================
# RAG PROCESSING (ROUTER)
# ============================================================


def _process_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: str,
    user_id: str,
) -> None:
    """
    Route to the correct RAG-indexing pipeline based on the
    file's actual detected type (not just the client-supplied
    mime_type).

    - Standalone image  -> caption via Gemini vision, index caption
    - PDF                -> extract text (unchanged) + extract
                             embedded images, caption them, index
                             text chunks + image captions together
    """

    file_type = _detect_file_type(
        file_path=file_path,
        mime_type=doc.mime_type,
        file_name=doc.file_name,
    )

    if file_type == "image":

        _process_image_for_rag(
            db=db,
            doc=doc,
            file_path=file_path,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return

    if file_type == "pdf":

        _process_pdf_for_rag(
            db=db,
            doc=doc,
            file_path=file_path,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return

    logger.warning(
        "Unsupported file type for RAG processing: "
        "document_id=%s mime_type=%s file_name=%s",
        doc.id,
        doc.mime_type,
        doc.file_name,
    )


# ============================================================
# STANDALONE IMAGE PROCESSING
# ============================================================


def _process_image_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: str,
    user_id: str,
) -> None:

    try:
        caption = generate_image_caption(file_path)

    except Exception:

        logger.exception(
            "Image captioning failed for document_id=%s",
            doc.id,
        )

        return

    if not caption or not caption.strip():

        logger.warning(
            "Empty caption generated for document_id=%s",
            doc.id,
        )

        return

    chunks = [caption.strip()]

    embeddings = embedding_manager.generate_embedding(
        chunks
    )

    vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        filename=doc.file_name,
        conversation_id=str(conversation_id),
        user_id=str(user_id),
    )

    document_repo.create_chunks(
        db=db,
        doc_id=doc.id,
        chunks=chunks,
    )

    logger.info(
        "Image RAG indexing completed: "
        "document_id=%s user_id=%s conversation_id=%s",
        doc.id,
        user_id,
        conversation_id,
    )


# ============================================================
# PDF PROCESSING (TEXT + EMBEDDED IMAGES)
# ============================================================


def _extract_pdf_images(
    file_path: str,
    doc_id: str,
) -> list[str]:
    """
    Extract embedded raster images from a PDF using PyMuPDF
    and save them as temporary files. Returns the saved paths.
    """

    saved_paths = []

    pdf = fitz.open(file_path)

    try:

        for page_index in range(len(pdf)):

            page = pdf[page_index]

            for image_index, img in enumerate(
                page.get_images(full=True)
            ):

                xref = img[0]

                try:
                    base_image = pdf.extract_image(xref)

                except Exception:

                    logger.warning(
                        "Unable to extract image xref=%s "
                        "page=%s document_id=%s",
                        xref,
                        page_index,
                        doc_id,
                    )

                    continue

                image_bytes = base_image["image"]
                extension = base_image["ext"]

                filename = (
                    f"{doc_id}_p{page_index}_{image_index}."
                    f"{extension}"
                )

                image_path = os.path.join(
                    UPLOAD_DIR,
                    filename,
                )

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                saved_paths.append(image_path)

    finally:
        pdf.close()

    return saved_paths


def _caption_pdf_images(
    image_paths: list[str],
    doc_id: str,
) -> list[str]:
    """
    Caption each extracted PDF image and clean up the
    temporary image files afterward.
    """

    captions = []

    for image_path in image_paths:

        try:
            caption = generate_image_caption(image_path)

            if caption and caption.strip():
                captions.append(
                    f"[Image content]: {caption.strip()}"
                )

        except Exception:

            logger.exception(
                "Captioning failed for image=%s document_id=%s",
                image_path,
                doc_id,
            )

        finally:

            try:
                if os.path.exists(image_path):
                    os.remove(image_path)

            except Exception:

                logger.warning(
                    "Unable to remove temporary PDF image: %s",
                    image_path,
                )

    return captions


def _process_pdf_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: str,
    user_id: str,
) -> None:
    """
    Extract text + embedded images, chunk everything,
    generate embeddings, and store vectors in Qdrant and
    chunks in PostgreSQL.
    """

    # --------------------------------------------------------
    # Extract PDF text
    # --------------------------------------------------------

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            text += page.extract_text() or ""

    # --------------------------------------------------------
    # Text splitting
    # --------------------------------------------------------

    text_chunks = []

    if text.strip():

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

        text_chunks = text_splitter.split_text(text)

    else:

        logger.warning(
            "No extractable text for document_id=%s",
            doc.id,
        )

    # --------------------------------------------------------
    # Extract + caption embedded images
    # --------------------------------------------------------

    image_captions = []

    try:

        extracted_image_paths = _extract_pdf_images(
            file_path=file_path,
            doc_id=doc.id,
        )

        image_captions = _caption_pdf_images(
            image_paths=extracted_image_paths,
            doc_id=doc.id,
        )

    except Exception:

        logger.exception(
            "PDF image extraction failed for document_id=%s",
            doc.id,
        )

    # --------------------------------------------------------
    # Combine text chunks + image captions
    # --------------------------------------------------------

    chunks = text_chunks + image_captions

    if not chunks:

        logger.warning(
            "No chunks (text or image) generated for "
            "document_id=%s",
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
        "document_id=%s text_chunks=%s image_chunks=%s "
        "user_id=%s conversation_id=%s",
        doc.id,
        len(text_chunks),
        len(image_captions),
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