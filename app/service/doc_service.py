import csv
import io
import os
import logging
from typing import Optional

from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import openpyxl
import docx  # python-docx

from fastapi import UploadFile
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document, DocumentStatus
from app.repository import document_repo
from app.repository.conversation_repo import get_conversation

from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)

from app.service.tools.image_tool import (
    generate_image_caption,
    ImageCaptionQuotaExceededError,
)


logger = logging.getLogger(__name__)

UPLOAD_DIR = "Uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# Gemini free tier allows only a small number of image-captioning
# requests PER DAY. Cap how many embedded images a single PDF
# will caption.
MAX_PDF_IMAGES_TO_CAPTION = int(
    os.getenv("MAX_PDF_IMAGES_TO_CAPTION", "15")
)


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


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
}

CSV_EXTENSIONS = {
    ".csv",
}

EXCEL_EXTENSIONS = {
    ".xlsx",
    ".xlsm",
}

DOCX_EXTENSIONS = {
    ".docx",
}

DOCX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
# VALIDATE DOCUMENT CONVERSATION
# ============================================================


def _validate_document_conversation(
    db: Session,
    conversation_id: Optional[str],
    user_id: str,
) -> Optional[str]:
    """
    Validate the conversation that will own the uploaded document.

    Rules:

    1. conversation_id is None:
       The document is a GLOBAL document.

    2. conversation_id is provided:
       The conversation must belong to the current user.

    3. A document can never be attached to another user's
       conversation.
    """

    # No conversation means GLOBAL document.
    if not conversation_id:
        return None

    conversation = get_conversation(
        db=db,
        conversation_id=str(conversation_id),
        user_id=str(user_id),
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

    logger.info(
    "UPLOAD DEBUG | file=%s | parent_id=%r | conversation_id=%r | user_id=%r",
    file.filename,
    parent_id,
    conversation_id,
    user_id,
)

    if not parent_id:
        parent_id = None

    if not conversation_id:
        conversation_id = None

    # --------------------------------------------------------
    # Global folder validation
    # --------------------------------------------------------

    if parent_id:

        parent = document_repo.get_owned_folder_by_id(
            db,
            parent_id,
            user_id,
        )

        if not parent:
            return None

        # A parent folder belongs to the global document space.
        #
        # Therefore a conversation document should not be
        # inserted inside the global document folder.
        if conversation_id:
            raise ValueError(
                "Conversation documents cannot be uploaded "
                "inside the global document folder"
            )

    # --------------------------------------------------------
    # Conversation validation
    # --------------------------------------------------------

    conversation_id = _validate_document_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    # --------------------------------------------------------
    # Create document record
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
    # Save local upload
    # --------------------------------------------------------

    contents = file.file.read()

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{doc.id}_{file.filename}",
    )

    with open(file_path, "wb") as f:
        f.write(contents)

    # --------------------------------------------------------
    # Generate GCS path
    # --------------------------------------------------------

    gcs_path = generate_gcs_path(
        user_id=user_id,
        document_id=doc.id,
        original_filename=file.filename,
    )

    try:

        # ----------------------------------------------------
        # RAG processing
        # ----------------------------------------------------

        _process_for_rag(
            db=db,
            doc=doc,
            file_path=file_path,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        # ----------------------------------------------------
        # Mark document READY
        # ----------------------------------------------------

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
# ============================================================


def _sniff_file_type(file_path: str) -> str:
    """
    Returns "image", "pdf", or "unknown" based on the file's
    actual content.
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
    Determine which RAG pipeline should process the uploaded file.
    """

    file_type = _sniff_file_type(file_path)

    if file_type != "unknown":
        return file_type

    if mime_type in IMAGE_MIME_TYPES:
        return "image"

    if mime_type == "application/pdf":
        return "pdf"

    if mime_type in DOCX_MIME_TYPES:
        return "docx"

    extension = os.path.splitext(file_name or "")[1].lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension == ".pdf":
        return "pdf"

    if extension in CSV_EXTENSIONS:
        return "csv"

    if extension in EXCEL_EXTENSIONS:
        return "excel"

    if extension in DOCX_EXTENSIONS:
        return "docx"

    if extension in TEXT_EXTENSIONS:
        return "text"

    return "unknown"


# ============================================================
# RAG PROCESSING (ROUTER)
# ============================================================


def _process_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: Optional[str],
    user_id: str,
) -> None:

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

    if file_type in ("csv", "excel", "text", "docx"):

        _process_tabular_or_text_for_rag(
            db=db,
            doc=doc,
            file_path=file_path,
            file_type=file_type,
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
    conversation_id: Optional[str],
    user_id: str,
) -> None:

    try:

        caption = generate_image_caption(file_path)

    except ImageCaptionQuotaExceededError:

        logger.warning(
            "Image-captioning quota exhausted for document_id=%s "
            "— document uploaded but not caption-indexed yet.",
            doc.id,
        )

        return

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

    embeddings = embedding_manager.generate_embedding(chunks)

    vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        filename=doc.file_name,
        conversation_id=conversation_id,
        user_id=str(user_id),
        document_id=str(doc.id),
        content_type="image",
        parent_document_id=str(doc.id),
    )

    document_repo.create_chunks(
        db=db,
        doc_id=doc.id,
        chunks=chunks,
    )

    logger.info(
        "Image RAG indexing completed: document_id=%s "
        "user_id=%s conversation_id=%s",
        doc.id,
        user_id,
        conversation_id,
    )


# ============================================================
# SHARED TEXT CHUNKING + INDEXING
# ============================================================


def _chunk_and_index_text(
    db: Session,
    doc: Document,
    text: str,
    conversation_id: Optional[str],
    user_id: str,
) -> int:

    if not text or not text.strip():

        logger.warning(
            "No extractable text for document_id=%s",
            doc.id,
        )

        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_text(text)

    if not chunks:
        return 0

    embeddings = embedding_manager.generate_embedding(chunks)

    vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        filename=doc.file_name,
        conversation_id=conversation_id,
        user_id=str(user_id),
        document_id=str(doc.id),
        content_type="text",
        parent_document_id=str(doc.id),
    )

    document_repo.create_chunks(
        db=db,
        doc_id=doc.id,
        chunks=chunks,
    )

    return len(chunks)


# ============================================================
# CSV / EXCEL / MARKDOWN / TEXT PROCESSING
# ============================================================


def _extract_csv_text(file_path: str) -> str:

    lines = []

    with open(
        file_path,
        newline="",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        reader = csv.reader(f)

        try:
            header = next(reader)

        except StopIteration:
            return ""

        for row in reader:

            row_text = ", ".join(
                f"{col}: {val}"
                for col, val in zip(header, row)
            )

            lines.append(row_text)

    return "\n".join(lines)


def _extract_excel_text(file_path: str) -> str:

    workbook = openpyxl.load_workbook(
        file_path,
        read_only=True,
        data_only=True,
    )

    lines = []

    try:

        for sheet in workbook.worksheets:

            rows = sheet.iter_rows(values_only=True)

            try:
                header = next(rows)

            except StopIteration:
                continue

            header = [
                str(h) if h is not None else ""
                for h in header
            ]

            lines.append(
                f"Sheet: {sheet.title}"
            )

            for row in rows:

                row_text = ", ".join(
                    f"{col}: {val}"
                    for col, val in zip(header, row)
                    if val is not None
                )

                if row_text:
                    lines.append(row_text)

    finally:
        workbook.close()

    return "\n".join(lines)


def _extract_plain_text(file_path: str) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        return f.read()


def _extract_docx_text(file_path: str) -> str:

    document = docx.Document(file_path)

    lines = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            lines.append(
                paragraph.text.strip()
            )

    for table in document.tables:

        for row in table.rows:

            row_text = ", ".join(
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            )

            if row_text:
                lines.append(row_text)

    return "\n".join(lines)


def _process_tabular_or_text_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    file_type: str,
    conversation_id: Optional[str],
    user_id: str,
) -> None:

    try:

        if file_type == "csv":

            text = _extract_csv_text(file_path)

        elif file_type == "excel":

            text = _extract_excel_text(file_path)

        elif file_type == "docx":

            text = _extract_docx_text(file_path)

        else:

            text = _extract_plain_text(file_path)

    except Exception:

        logger.exception(
            "Content extraction failed for document_id=%s "
            "file_type=%s",
            doc.id,
            file_type,
        )

        return

    chunk_count = _chunk_and_index_text(
        db=db,
        doc=doc,
        text=text,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    logger.info(
        "RAG indexing completed: document_id=%s "
        "file_type=%s text_chunks=%s user_id=%s "
        "conversation_id=%s",
        doc.id,
        file_type,
        chunk_count,
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


# ============================================================
# PDF IMAGE PERSISTENCE
# ============================================================


def _persist_pdf_image(
    db: Session,
    parent_doc: Document,
    image_bytes: bytes,
    extension: str,
    page_index: int,
    image_index: int,
    conversation_id: Optional[str],
    user_id: str,
) -> Document:

    mime_ext = (
        "jpeg"
        if extension.lower() in ("jpg", "jpeg")
        else extension.lower()
    )

    filename = (
        f"{parent_doc.file_name}_p"
        f"{page_index}_{image_index}.{extension}"
    )

    image_doc = document_repo.create_file(
        db=db,
        file_name=filename,
        parent_id=parent_doc.id,
        user_id=user_id,
        mime_type=f"image/{mime_ext}",
        conversation_id=conversation_id,
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{image_doc.id}_{filename}",
    )

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    image_doc = document_repo.update_file_storage_info(
        db=db,
        doc=image_doc,
        gcs_path=generate_gcs_path(
            user_id,
            image_doc.id,
            filename,
        ),
        size_bytes=len(image_bytes),
        status=DocumentStatus.READY,
    )

    return image_doc


def _extract_and_persist_pdf_images(
    db: Session,
    parent_doc: Document,
    file_path: str,
    conversation_id: Optional[str],
    user_id: str,
) -> list[tuple[Document, str]]:

    results = []

    pdf = fitz.open(file_path)

    try:

        for page_index in range(len(pdf)):

            page = pdf[page_index]

            for image_index, img in enumerate(
                page.get_images(full=True)
            ):

                xref = img[0]

                try:

                    base_image = pdf.extract_image(
                        xref
                    )

                except Exception:

                    logger.warning(
                        "Unable to extract image xref=%s "
                        "page=%s document_id=%s",
                        xref,
                        page_index,
                        parent_doc.id,
                    )

                    continue

                image_doc = _persist_pdf_image(
                    db=db,
                    parent_doc=parent_doc,
                    image_bytes=base_image["image"],
                    extension=base_image["ext"],
                    page_index=page_index,
                    image_index=image_index,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                image_path = os.path.join(
                    UPLOAD_DIR,
                    f"{image_doc.id}_{image_doc.file_name}",
                )

                results.append(
                    (image_doc, image_path)
                )

    finally:
        pdf.close()

    return results


# ============================================================
# PDF RAG PROCESSING
# ============================================================


def _process_pdf_for_rag(
    db: Session,
    doc: Document,
    file_path: str,
    conversation_id: Optional[str],
    user_id: str,
) -> None:

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:
            text += page.extract_text() or ""

    text_chunks = []

    if text.strip():

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )

        text_chunks = text_splitter.split_text(text)

    else:

        logger.warning(
            "No extractable text for document_id=%s",
            doc.id,
        )

    # --------------------------------------------------------
    # PDF text chunks
    # --------------------------------------------------------

    if text_chunks:

        embeddings = embedding_manager.generate_embedding(
            text_chunks
        )

        vector_store.add_documents(
            chunks=text_chunks,
            embeddings=embeddings,
            filename=doc.file_name,
            conversation_id=conversation_id,
            user_id=str(user_id),
            document_id=str(doc.id),
            content_type="text",
            parent_document_id=str(doc.id),
        )

        document_repo.create_chunks(
            db=db,
            doc_id=doc.id,
            chunks=text_chunks,
        )

    # --------------------------------------------------------
    # PDF embedded images
    # --------------------------------------------------------

    image_pairs = []

    try:

        image_pairs = _extract_and_persist_pdf_images(
            db=db,
            parent_doc=doc,
            file_path=file_path,
            conversation_id=conversation_id,
            user_id=user_id,
        )

    except Exception:

        logger.exception(
            "PDF image extraction failed for document_id=%s",
            doc.id,
        )

    image_count = 0
    skipped_count = 0

    images_to_caption = image_pairs[
        :MAX_PDF_IMAGES_TO_CAPTION
    ]

    skipped_count += max(
        0,
        len(image_pairs) - MAX_PDF_IMAGES_TO_CAPTION,
    )

    if skipped_count:

        logger.warning(
            "PDF has %s embedded images; only captioning "
            "the first %s for document_id=%s.",
            len(image_pairs),
            MAX_PDF_IMAGES_TO_CAPTION,
            doc.id,
        )

    for image_doc, image_path in images_to_caption:

        try:

            caption = generate_image_caption(
                image_path
            )

            if not caption or not caption.strip():
                continue

            chunks = [caption.strip()]

            embeddings = (
                embedding_manager.generate_embedding(
                    chunks
                )
            )

            vector_store.add_documents(
                chunks=chunks,
                embeddings=embeddings,
                filename=image_doc.file_name,
                conversation_id=conversation_id,
                user_id=str(user_id),
                document_id=str(image_doc.id),
                content_type="image",
                parent_document_id=str(doc.id),
            )

            document_repo.create_chunks(
                db=db,
                doc_id=image_doc.id,
                chunks=chunks,
            )

            image_count += 1

        except ImageCaptionQuotaExceededError:

            remaining = (
                len(images_to_caption)
                - image_count
                - skipped_count
            )

            skipped_count += max(
                0,
                remaining,
            )

            logger.warning(
                "Image-captioning quota exhausted while "
                "processing document_id=%s",
                doc.id,
            )

            break

        except Exception:

            logger.exception(
                "Captioning failed for image_document_id=%s "
                "parent=%s",
                image_doc.id,
                doc.id,
            )

    logger.info(
        "RAG indexing completed: document_id=%s "
        "text_chunks=%s image_chunks=%s "
        "image_chunks_skipped=%s user_id=%s "
        "conversation_id=%s",
        doc.id,
        len(text_chunks),
        image_count,
        skipped_count,
        user_id,
        conversation_id,
    )
