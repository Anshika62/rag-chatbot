import logging
import os
from typing import Any, Optional

from PIL import Image
import pytesseract
from langchain_core.tools import tool

from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)

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
    conversation_id: str | None,
    page_number: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    page_number:
        If provided, only images belonging to that exact
        1-based page/slide are returned (§19 — direct metadata
        filter, no semantic search).
    """

    image_docs = []

    if (
        not parent_doc.is_folder
        and parent_doc.mime_type
        and parent_doc.mime_type.startswith("image/")
        and (
            page_number is None
            or parent_doc.page_number == page_number
        )
    ):
        image_docs.append(parent_doc)

    children = document_repo.get_children(
        db=db,
        parent_id=parent_doc.id,
        conversation_id=conversation_id,
        user_id=user_id,
        page_number=page_number,
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
        "IMAGE RETRIEVAL: document_id=%s page_number=%s "
        "image_count=%s",
        parent_doc.id,
        page_number,
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

                live_caption = generate_image_caption(
                    image_path
                )

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

                    caption_embedding = (
                        embedding_manager.generate_embedding(
                            [caption_text]
                        )
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
                        page_number=image_doc.page_number,
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
                "page_number": image_doc.page_number,
                "gcs_path": image_doc.gcs_path,
            }
        )

    return documents


def _is_image_query(query: str) -> bool:
    """
    Detect whether the user is asking for images,
    pictures, photos, figures, diagrams, charts, etc.

    This is intentionally a SECONDARY signal (§24) — the LLM's own
    `content_type` tool argument is the primary routing mechanism.
    This is only consulted as a fallback when the LLM passes
    content_type="any" and the raw query text is unambiguous, so a
    plain keyword match never overrides an explicit LLM decision.
    """

    image_keywords = [
        "image",
        "images",
        "picture",
        "pictures",
        "photo",
        "photos",
        "figure",
        "figures",
        "diagram",
        "diagrams",
        "chart",
        "charts",
        "illustration",
        "illustrations",
        "चित्र",
        "तस्वीर",
        "फोटो",
        "इमेज",
        "डायग्राम",
    ]

    query_lower = query.lower()

    return any(
        keyword in query_lower
        for keyword in image_keywords
    )


def create_search_knowledge_base_tool(
    user_id: str,
    conversation_id: str | None,
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

    For image-related queries on a specific PDF/PPTX, the tool
    directly retrieves image child documents instead of relying
    on semantic/vector search.
    """

    user_id = str(user_id)

    conversation_id = (
        str(conversation_id)
        if conversation_id is not None
        else None
    )

    document_id = (
        str(document_id)
        if document_id
        else None
    )

    def _search_knowledge_base_impl(
        query: str,
        limit: int = 4,
        content_type: str = "any",
        page_number: Optional[int] = None,
    ) -> list[dict[str, Any]]:

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

        # ========================================================
        # SECONDARY KEYWORD SIGNAL (§24)
        #
        # Only used to fill in an ambiguous "any" — never
        # overrides an explicit "text" or "image" decision made
        # by the LLM itself.
        # ========================================================

        effective_content_type = content_type

        if (
            content_type == "any"
            and _is_image_query(clean_query)
        ):
            effective_content_type = "image"

        logger.info(
            "KB SEARCH START: query=%s, limit=%s, "
            "content_type=%s (effective=%s), page_number=%s, "
            "user_id=%s, conversation_id=%s, document_id=%s",
            clean_query,
            limit,
            content_type,
            effective_content_type,
            page_number,
            user_id,
            conversation_id,
            document_id,
        )

        # ========================================================
        # DIRECT IMAGE RETRIEVAL
        #
        # Triggered by content_type="image" on a specific document,
        # optionally narrowed to a single page/slide via
        # page_number (§19). No semantic search involved.
        # ========================================================

        if document_id and effective_content_type == "image":

            db = SessionLocal()

            try:

                parent_doc = (
                    document_repo.get_accessible_document_by_id(
                        db=db,
                        doc_id=document_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                )

                if not parent_doc:

                    return [
                        {
                            "filename": None,
                            "chunk_index": None,
                            "text": (
                                "Document not found, or is not "
                                "accessible from this conversation."
                            ),
                        }
                    ]

                documents = _retrieve_document_images(
                    db=db,
                    parent_doc=parent_doc,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    page_number=page_number,
                )

                if not documents:

                    return [
                        {
                            "filename": None,
                            "chunk_index": None,
                            "text": (
                                f"No images were found on page "
                                f"{page_number} of this document."
                                if page_number is not None
                                else "No images were found in "
                                "this document."
                            ),
                        }
                    ]

                return documents

            except Exception:

                logger.exception(
                    "IMAGE RETRIEVAL FAILED: document_id=%s",
                    document_id,
                )

                return [
                    {
                        "filename": None,
                        "chunk_index": None,
                        "text": (
                            "Unable to retrieve images from this "
                            "document right now. Please try again."
                        ),
                    }
                ]

            finally:

                db.close()

        # ========================================================
        # SEMANTIC SEARCH
        #
        # document_id (and page_number, when set) are now applied
        # as Qdrant payload filters directly — not a client-side
        # post-filter over an artificially widened top_k window.
        # ========================================================

        query_embedding = (
            embedding_manager.generate_embedding(
                [clean_query]
            )
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            conversation_id=conversation_id,
            content_type=(
                effective_content_type
                if effective_content_type in ("text", "image")
                else None
            ),
            document_id=document_id,
            page_number=page_number,
            top_k=limit,
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

            text = payload.get(
                "text",
                "",
            )

            if not text:
                continue

            point_document_id = payload.get(
                "document_id"
            )

            point_parent_id = payload.get(
                "parent_document_id"
            )

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
                    "page_number": payload.get(
                        "page_number"
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

        # ========================================================
        # IMAGE FALLBACK
        #
        # If a document-scoped semantic search comes back empty,
        # but the document itself is image-related, fall back to
        # direct image retrieval rather than reporting "not found".
        # ========================================================

        if not documents and document_id:

            db = SessionLocal()

            try:

                parent_doc = (
                    document_repo.get_accessible_document_by_id(
                        db=db,
                        doc_id=document_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                )

                if parent_doc:

                    children = (
                        document_repo.get_children(
                            db=db,
                            parent_id=parent_doc.id,
                            conversation_id=conversation_id,
                            user_id=user_id,
                        )
                    )

                    is_image_related = (
                        (
                            parent_doc.mime_type
                            and parent_doc.mime_type.startswith(
                                "image/"
                            )
                        )
                        or any(
                            child.mime_type
                            and child.mime_type.startswith(
                                "image/"
                            )
                            for child in children
                        )
                    )

                    if is_image_related:

                        fallback_documents = (
                            _retrieve_document_images(
                                db=db,
                                parent_doc=parent_doc,
                                user_id=user_id,
                                conversation_id=conversation_id,
                                page_number=page_number,
                            )
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

        # ========================================================
        # NO RESULTS
        # ========================================================

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

    # ============================================================
    # PUBLIC TOOL WRAPPER
    # ============================================================

    @tool
    def search_knowledge_base(
        query: str,
        limit: int = 4,
        content_type: str = "any",
        page_number: Optional[int] = None,
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
                "shown"/"visible" in an uploaded file.

                Set to "text" when the user is clearly asking
                about textual/written content only.

                Use "any" when genuinely unclear.

                When "image" is passed for a specific document,
                the extracted image(s) are returned directly
                instead of running semantic text search.

            page_number: Set this to a specific 1-based page or
                slide number when the user names an exact page,
                e.g. "show me the image on page 4" or "what's on
                slide 4". Leave unset otherwise. Only meaningful
                together with a specific document (a document_id
                is already scoped for this conversation turn).
        """

        try:

            return _search_knowledge_base_impl(
                query=query,
                limit=limit,
                content_type=content_type,
                page_number=page_number,
            )

        except ValueError:

            raise

        except Exception:

            logger.exception(
                "KNOWLEDGE BASE SEARCH FAILED: query=%s "
                "user_id=%s conversation_id=%s document_id=%s",
                query,
                user_id,
                conversation_id,
                document_id,
            )

            return [
                {
                    "filename": None,
                    "chunk_index": None,
                    "text": (
                        "The knowledge base search is temporarily "
                        "unavailable. Please try again in a moment."
                    ),
                }
            ]

    return search_knowledge_base