import logging
from typing import Any

from langchain_core.tools import tool

from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)

from app.repository import document_repo
from app.core.database import SessionLocal


logger = logging.getLogger(__name__)


def _is_image_query(query: str) -> bool:
    """
    Detect whether the user is asking for images,
    pictures, photos, figures, diagrams, charts, etc.
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
    ) -> list[dict[str, Any]]:
        """
        Search uploaded documents and indexed knowledge base
        for relevant information.

        For image-related queries on a specific document,
        returns the extracted image documents directly.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "Knowledge-base search query cannot be empty."
            )

        # ----------------------------------------------------
        # Validate limit
        # ----------------------------------------------------

        if limit < 1:
            raise ValueError(
                "Search result limit must be at least 1."
            )

        # Never allow the LLM to request an excessive number
        # of vector-search results.
        #
        # This limit applies to normal vector search.
        # Image retrieval below returns all extracted images.
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

        # ====================================================
        # DIRECT PDF IMAGE RETRIEVAL
        # ====================================================
        #
        # If this is a specific document query and the user is
        # asking for images, don't use semantic search.
        #
        # Instead:
        #
        # PDF
        #   ├── image 1
        #   ├── image 2
        #   ├── image 3
        #   └── ...
        #
        # are retrieved directly using parent_id.
        # ====================================================

        if document_id and _is_image_query(clean_query):

            db = SessionLocal()

            try:
                # ------------------------------------------------
                # Verify that the document belongs to the user
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Get all children of the PDF
                # ------------------------------------------------

                children = document_repo.get_children(
                    db=db,
                    parent_id=parent_doc.id,
                )

                # ------------------------------------------------
                # Keep only image documents
                # ------------------------------------------------

                image_docs = [
                    doc
                    for doc in children
                    if (
                        not doc.is_folder
                        and doc.mime_type
                        and doc.mime_type.startswith("image/")
                    )
                ]

                logger.info(
                    "PDF IMAGE RETRIEVAL: "
                    "document_id=%s image_count=%s",
                    document_id,
                    len(image_docs),
                )

                # ------------------------------------------------
                # Format image results
                # ------------------------------------------------

                documents: list[dict[str, Any]] = []

                for image_doc in image_docs:
                    documents.append(
                        {
                            "filename": image_doc.file_name,
                            "chunk_index": None,
                            "text": (
                                f"Image extracted from "
                                f"{parent_doc.file_name}"
                            ),
                            "document_id": str(image_doc.id),
                            "parent_document_id": str(
                                parent_doc.id
                            ),
                            "content_type": image_doc.mime_type,
                            "gcs_path": image_doc.gcs_path,
                        }
                    )

                # ------------------------------------------------
                # No images found
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Return ALL images
                # ------------------------------------------------

                return documents

            finally:
                db.close()

        # ====================================================
        # NORMAL SEMANTIC / VECTOR SEARCH
        # ====================================================

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = embedding_manager.generate_embedding(
            [clean_query]
        )

        # ----------------------------------------------------
        # Search Qdrant
        #
        # user_id is always enforced.
        #
        # conversation_id normally scopes the search to the
        # current conversation.
        #
        # When document_id is provided, conversation_id is not
        # passed because document_id is already the narrower
        # scope.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Diagnostic logging
        # ----------------------------------------------------

        logger.info(
            "KB SEARCH RAW: raw_count=%s sample_payload=%s",
            len(results),
            (
                results[0].payload
                if results
                else None
            ),
        )

        # ----------------------------------------------------
        # Format search results
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Scope to a single uploaded document
            # ------------------------------------------------

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

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logger.info(
            "KB SEARCH COMPLETE: count=%s, filenames=%s",
            len(documents),
            [
                item.get("filename")
                for item in documents
            ],
        )

        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

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