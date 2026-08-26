import logging
from typing import Any

from langchain_core.tools import tool

from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)
import os

from PIL import Image
import pytesseract

from app.service.tools.image_tool import (
    generate_image_caption,
    ImageCaptionQuotaExceededError,
)

from app.repository import document_repo
from app.core.database import SessionLocal


logger = logging.getLogger(__name__)


def _extract_text_from_image(image_path: str) -> str:
    """
    Same OCR approach used at upload time in doc_service.py's
    PDF image pipeline, reused here so the live-caption fallback
    produces the same combined caption+OCR content instead of
    caption-only text.
    """

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            ocr_text = pytesseract.image_to_string(
                image,
                config="--psm 6",
            )

        return ocr_text.strip()

    except Exception:

        logger.exception(
            "OCR failed for image: %s",
            image_path,
        )

        return ""


def _retrieve_document_images(
    db,
    parent_doc,
    user_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:

    image_docs = []

    if (
        not parent_doc.is_folder
        and parent_doc.mime_type
        and parent_doc.mime_type.startswith("image/")
    ):
        image_docs.append(parent_doc)

    children = document_repo.get_children(
        db=db,
        parent_id=parent_doc.id,
    )

    image_docs.extend(
        doc
        for doc in children
        if (
            not doc.is_folder
            and doc.mime_type
            and doc.mime_type.startswith("image/")
        )
    )

    logger.info(
        "IMAGE RETRIEVAL: document_id=%s image_count=%s",
        parent_doc.id,
        len(image_docs),
    )

    documents: list[dict[str, Any]] = []

    for image_doc in image_docs:
        is_standalone = image_doc.id == parent_doc.id

        stored_chunks = document_repo.get_chunks_by_document_id(
            db=db,
            doc_id=image_doc.id,
        )

        caption_text = (
            " ".join(
                chunk.chunk_text
                for chunk in stored_chunks
                if chunk.chunk_text
            ).strip()
            if stored_chunks
            else ""
        )

        if not caption_text:
            image_path = os.path.join(
                "Uploads",
                f"{image_doc.id}_{image_doc.file_name}",
            )

            try:
                live_caption = generate_image_caption(image_path)
                live_caption = (
                    live_caption.strip()
                    if live_caption
                    else ""
                )

                live_ocr_text = _extract_text_from_image(
                    image_path
                )

                searchable_parts = []

                if live_caption:
                    searchable_parts.append(
                        f"Image description:\n{live_caption}"
                    )

                if live_ocr_text:
                    searchable_parts.append(
                        f"Text extracted from image:\n{live_ocr_text}"
                    )

                combined_text = "\n\n".join(
                    searchable_parts
                ).strip()

                if combined_text:
                    caption_text = combined_text

                    document_repo.create_chunks(
                        db=db,
                        doc_id=image_doc.id,
                        chunks=[caption_text],
                    )

                    caption_embedding = embedding_manager.generate_embedding(
                        [caption_text]
                    )

                    vector_store.add_documents(
                        chunks=[caption_text],
                        embeddings=caption_embedding,
                        filename=image_doc.file_name,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        document_id=str(image_doc.id),
                        content_type="image",
                        parent_document_id=str(parent_doc.id),
                    )

            except ImageCaptionQuotaExceededError:
                caption_text = (
                    "Image captioning quota is currently "
                    "exhausted. Please try again later."
                )

            except Exception:
                logger.exception(
                    "Live caption/OCR generation failed for "
                    "document_id=%s",
                    image_doc.id,
                )

        if not caption_text:
            caption_text = (
                "No AI-generated description is available "
                "for this image yet."
            )

        documents.append(
            {
                "filename": image_doc.file_name,
                "chunk_index": None,
                "text": (
                    f"Image description: {caption_text}"
                    if is_standalone
                    else (
                        f"Image extracted from "
                        f"{parent_doc.file_name} — "
                        f"description: {caption_text}"
                    )
                ),
                "document_id": str(image_doc.id),
                "parent_document_id": str(parent_doc.id),
                "content_type": image_doc.mime_type,
                "gcs_path": image_doc.gcs_path,
            }
        )

    return documents


def create_search_knowledge_base_tool(
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
):
    """
    Create a knowledge-base search tool for the current
    authenticated user and conversation.

    user_id, conversation_id, and (optionally) document_id are
    injected by the application and are NOT exposed as LLM
    tool arguments.

    When document_id is provided, results are scoped to that
    uploaded document.

    For image-related queries on a specific PDF, the tool
    directly retrieves image child documents instead of relying
    on semantic/vector search.
    """

    user_id = str(user_id)
    conversation_id = str(conversation_id)
    document_id = str(document_id) if document_id else None

    @tool
    def search_knowledge_base(
        query: str,
        limit: int = 4,
        content_type: str = "any",
    ) -> list[dict[str, Any]]:
        """
        Search uploaded documents and indexed knowledge base
        for relevant information.

        Args:
            query: The search text describing what to look for.
            limit: Max number of results to return.
            content_type: Set to "image" whenever the user is
                asking about an image, picture, photo, figure,
                diagram, chart, illustration, screenshot, or
                asking what something "looks like" / what is
                "shown"/"visible" in an uploaded file — in ANY
                language or phrasing, including indirect ones
                like "iska content kya hai" or "ismein kya hai".
                Set to "text" when the user is clearly asking
                about textual/written content only. Use "any"
                (default) only when genuinely unclear.
                When "image" is passed for a specific document,
                the extracted image(s) are returned directly
                instead of running semantic text search.
        """

        if not query or not query.strip():
            raise ValueError(
                "Knowledge-base search query cannot be empty."
            )

        if limit < 1:
            raise ValueError(
                "Search result limit must be at least 1."
            )

        limit = min(limit, 4)

        clean_query = query.strip()

        logger.info(
            "KB SEARCH START: query=%s, limit=%s, "
            "user_id=%s, conversation_id=%s, document_id=%s",
            clean_query,
            limit,
            user_id,
            conversation_id,
            document_id,
        )

        if document_id and content_type == "image":

            db = SessionLocal()

            try:
                parent_doc = document_repo.get_owned_document_by_id(
                    db=db,
                    doc_id=document_id,
                    user_id=user_id,
                )

                if not parent_doc:
                    return [
                        {
                            "filename": None,
                            "chunk_index": None,
                            "text": "Document not found.",
                        }
                    ]

                documents = _retrieve_document_images(
                    db=db,
                    parent_doc=parent_doc,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )

                if not documents:
                    return [
                        {
                            "filename": None,
                            "chunk_index": None,
                            "text": (
                                "No images were found in this PDF."
                            ),
                        }
                    ]

                return documents

            finally:
                db.close()

        query_embedding = embedding_manager.generate_embedding(
            [clean_query]
        )

        search_top_k = (
            max(limit * 5, 20)
            if document_id
            else limit
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            conversation_id=(
                None
                if document_id
                else conversation_id
            ),
            top_k=search_top_k,
        )

        logger.info(
            "KB SEARCH RAW: raw_count=%s sample_payload=%s",
            len(results),
            (
                results[0].payload
                if results
                else None
            ),
        )

        documents: list[dict[str, Any]] = []

        for point in results:

            payload = point.payload or {}

            text = payload.get("text", "")

            if not text:
                continue

            point_document_id = payload.get(
                "document_id"
            )

            point_parent_id = payload.get(
                "parent_document_id"
            )

            if document_id:
                if (
                    str(point_document_id)
                    != str(document_id)
                    and str(point_parent_id)
                    != str(document_id)
                ):
                    continue

            documents.append(
                {
                    "filename": payload.get(
                        "filename"
                    ),
                    "chunk_index": payload.get(
                        "chunk_index"
                    ),
                    "text": text,
                    "document_id": (
                        str(point_document_id)
                        if point_document_id
                        else None
                    ),
                    "parent_document_id": (
                        str(point_parent_id)
                        if point_parent_id
                        else None
                    ),
                    "content_type": payload.get(
                        "content_type"
                    ),
                }
            )

            if len(documents) >= limit:
                break

        logger.info(
            "KB SEARCH COMPLETE: count=%s, filenames=%s",
            len(documents),
            [
                item.get("filename")
                for item in documents
            ],
        )

        # ----------------------------------------------------
        # Fallback: if this search was scoped to a specific
        # document and semantic search found nothing (e.g. the
        # document is an image whose caption/embedding was never
        # created because the LLM did not send content_type=
        # "image"), check whether it's image-based and retrieve
        # it directly instead of reporting "not found".
        # ----------------------------------------------------

        if not documents and document_id:

            db = SessionLocal()

            try:
                parent_doc = document_repo.get_owned_document_by_id(
                    db=db,
                    doc_id=document_id,
                    user_id=user_id,
                )

                if parent_doc:

                    is_image_related = (
                        parent_doc.mime_type
                        and parent_doc.mime_type.startswith("image/")
                    ) or any(
                        child.mime_type
                        and child.mime_type.startswith("image/")
                        for child in document_repo.get_children(
                            db=db,
                            parent_id=parent_doc.id,
                        )
                    )

                    if is_image_related:

                        fallback_documents = _retrieve_document_images(
                            db=db,
                            parent_doc=parent_doc,
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )

                        if fallback_documents:

                            logger.info(
                                "KB SEARCH IMAGE FALLBACK USED: "
                                "document_id=%s count=%s",
                                document_id,
                                len(fallback_documents),
                            )

                            return fallback_documents

            finally:
                db.close()

        if not documents:
            return [
                {
                    "filename": None,
                    "chunk_index": None,
                    "text": (
                        "No relevant information was found "
                        "in the uploaded knowledge base."
                    ),
                }
            ]

        return documents

    return search_knowledge_base