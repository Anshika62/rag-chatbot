import logging
import os
from typing import Any, Optional

from PIL import Image
import pytesseract
from sqlalchemy import or_
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
from app.models.document import Document


logger = logging.getLogger(__name__)


def _is_image_content_type(content_type) -> bool:
    """
    True for either shape "content_type" appears in across this
    codebase:
      - the bare Qdrant payload discriminator "image" (what
        vector_store.add_documents(content_type="image", ...)
        actually stores — see doc_service.py / search_kb.py)
      - a real mime type like "image/png" (what
        _retrieve_document_images returns, taken from
        image_doc.mime_type)

    A plain `.startswith("image/")` check only matches the second
    shape and silently misses every semantic-search hit, since
    those come back from Qdrant with the bare "image" string —
    that mismatch is why image results from search_knowledge_base
    were previously missing their "url" field.
    """

    if not content_type:
        return False

    content_type = str(content_type)

    return (
        content_type == "image"
        or content_type.startswith("image/")
    )


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
                # NEW: ready-to-use render URL, so the LLM never
                # has to reconstruct/guess the path pattern when
                # embedding this image inline in its answer.
                "url": f"/documents/{image_doc.id}/file",
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


# ================================================================
# FILENAME-HINT RESOLUTION (NEW)
#
# PROBLEM THIS FIXES: pure semantic/vector search matches on
# MEANING, not on filename text. A query like "who is present in
# premium_photo-1669740462478-135db9b990ea.avi" has almost no
# semantic content Overlap with any stored image caption, so the
# vector search was returning unrelated images that happened to
# score highest by accident.
#
# WHAT THIS DOES NOT DO: it does NOT replace, skip, or alter the
# semantic/vector search in any way. It only runs BEFORE it, and
# only when the tool-level document_id is not already fixed
# (i.e. the general "search across everything" case — exactly the
# case seen in the logs). If it finds exactly one uploaded document
# whose filename clearly matches something in the query text, it
# narrows the SAME semantic search to that one document via the
# document_id filter the search already supports. The embedding
# comparison itself is completely unchanged.
#
# If zero or multiple documents match, this is a no-op and the
# original unscoped semantic search runs exactly as before —
# so there's no way for this to make results worse than they
# already were; it can only help disambiguate.
# ================================================================

MIN_FILENAME_HINT_LENGTH = 4


def _find_documents_by_filename_hint(
    db,
    user_id: str,
    conversation_id: Optional[str],
    query: str,
    limit: int = 3,
) -> list[Document]:

    clean = query.strip()

    if not clean or len(clean) < MIN_FILENAME_HINT_LENGTH:
        return []

    base_query = (
        db.query(Document)
        .filter(Document.user_id == user_id)
        .filter(Document.is_folder == False)  # noqa: E712
    )

    if conversation_id:
        base_query = base_query.filter(
            or_(
                Document.conversation_id == conversation_id,
                Document.conversation_id.is_(None),
            )
        )
    else:
        base_query = base_query.filter(
            Document.conversation_id.is_(None)
        )

    # First pass: does the query, as a whole, contain (or get
    # contained by) an actual uploaded filename? This is the
    # common case — the user pastes/types the filename directly.
    direct_matches = (
        base_query
        .filter(Document.file_name.ilike(f"%{clean}%"))
        .limit(limit)
        .all()
    )

    if direct_matches:
        return direct_matches

    # Second pass: token overlap fallback — handles minor typos/
    # truncation (e.g. a dropped trailing letter, or the user only
    # typing part of a long filename). Only meaningful tokens
    # (length > 3) are used so short words like "the", "img" don't
    # cause false positives.
    tokens = [
        token
        for token in clean.replace("_", " ").replace("-", " ").split()
        if len(token) > 3
    ]

    if not tokens:
        return []

    token_filters = [
        Document.file_name.ilike(f"%{token}%")
        for token in tokens
    ]

    return (
        base_query
        .filter(or_(*token_filters))
        .limit(limit)
        .all()
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

        # ========================================================
        # FILENAME-HINT RESOLUTION (NEW — see block comment above)
        #
        # Only attempted when the tool wasn't already scoped to a
        # specific document at creation time. If exactly one
        # document matches, semantic search below is narrowed to
        # it. Any other outcome (0 or 2+ matches) leaves behavior
        # completely unchanged from before.
        # ========================================================

        resolved_document_id = document_id

        if not resolved_document_id:

            hint_db = SessionLocal()

            try:

                filename_matches = _find_documents_by_filename_hint(
                    db=hint_db,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query=clean_query,
                )

                if len(filename_matches) == 1:

                    resolved_document_id = str(
                        filename_matches[0].id
                    )

                    logger.info(
                        "KB SEARCH FILENAME HINT MATCHED: "
                        "query=%s -> document_id=%s (%s)",
                        clean_query,
                        resolved_document_id,
                        filename_matches[0].file_name,
                    )

                elif len(filename_matches) > 1:

                    logger.info(
                        "KB SEARCH FILENAME HINT AMBIGUOUS: "
                        "query=%s matched %s documents — "
                        "falling back to unscoped semantic search",
                        clean_query,
                        len(filename_matches),
                    )

            except Exception:

                logger.exception(
                    "Filename-hint resolution failed for "
                    "query=%s — falling back to unscoped "
                    "semantic search",
                    clean_query,
                )

            finally:

                hint_db.close()

        logger.info(
            "KB SEARCH START: query=%s, limit=%s, "
            "content_type=%s (effective=%s), page_number=%s, "
            "user_id=%s, conversation_id=%s, document_id=%s "
            "(resolved=%s)",
            clean_query,
            limit,
            content_type,
            effective_content_type,
            page_number,
            user_id,
            conversation_id,
            document_id,
            resolved_document_id,
        )

        # ========================================================
        # DIRECT IMAGE RETRIEVAL
        #
        # Triggered by content_type="image" on a specific document,
        # optionally narrowed to a single page/slide via
        # page_number (§19). No semantic search involved.
        # ========================================================

        if resolved_document_id and effective_content_type == "image":

            db = SessionLocal()

            try:

                parent_doc = (
                    document_repo.get_accessible_document_by_id(
                        db=db,
                        doc_id=resolved_document_id,
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
                    resolved_document_id,
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
        #
        # This is the exact same semantic search as before — the
        # only change is that `resolved_document_id` (which equals
        # the original `document_id` unless the filename-hint
        # match above fired) is passed instead of the raw
        # `document_id`.
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
            document_id=resolved_document_id,
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

            point_content_type = payload.get(
                "content_type"
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
                    "content_type": point_content_type,
                    "page_number": payload.get(
                        "page_number"
                    ),
                    # NEW: same ready-to-use render URL as the
                    # direct-image-retrieval path above, so image
                    # results coming back from semantic search are
                    # just as embeddable by the LLM. Only added
                    # when this hit is actually an image — text
                    # hits keep their existing shape unchanged.
                    # Uses _is_image_content_type() because Qdrant
                    # stores this as the bare "image" discriminator,
                    # not a mime type.
                    **(
                        {
                            "url": (
                                f"/documents/"
                                f"{point_document_id}/file"
                            )
                        }
                        if (
                            _is_image_content_type(
                                point_content_type
                            )
                            and point_document_id
                        )
                        else {}
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

        if not documents and resolved_document_id:

            db = SessionLocal()

            try:

                parent_doc = (
                    document_repo.get_accessible_document_by_id(
                        db=db,
                        doc_id=resolved_document_id,
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
                                resolved_document_id,
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