import logging
from typing import Any

from langchain_core.tools import tool

from app.service.rag_clients import (
    embedding_manager,
    vector_store,
)


logger = logging.getLogger(__name__)


def create_search_knowledge_base_tool(
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
):
    """
    Create a knowledge-base search tool for the current
    authenticated user and conversation.

    user_id, conversation_id, and (optionally) document_id are
    injected by the application and are NOT exposed as LLM tool
    arguments.

    When document_id is provided, results are scoped to that one
    uploaded document (matching either its own document_id, or,
    for images extracted from a PDF, their parent_document_id) —
    used for "ask about this specific file" style queries.
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

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = embedding_manager.generate_embedding(
            [clean_query]
        )

        # ----------------------------------------------------
        # Search Qdrant
        #
        # user_id is always enforced — a user can never search
        # another user's documents.
        #
        # conversation_id normally scopes the search to the
        # current conversation (avoids diluting results with
        # chunks from the user's other conversations). But when
        # document_id is explicitly requested, we deliberately
        # DON'T pass conversation_id — vector_store.search treats
        # conversation_id=None as "no conversation filter" — so
        # the query still finds the document even if it was
        # originally uploaded under a different conversation_id
        # than the one the current chat is using. document_id is
        # already a much narrower, more specific scope than
        # conversation_id, so this stays safe.
        #
        # We also pull a larger candidate pool in that case and
        # filter to the exact document below, so we still end up
        # with up to `limit` matches from just that document.
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

        # Diagnostic: tells us whether Qdrant returned zero raw
        # matches at all (conversation_id/user_id filter issue),
        # vs matches came back but document_id filtering below
        # drops all of them (document_id/payload mismatch).
        logger.info(
            "KB SEARCH RAW: raw_count=%s sample_payload=%s",
            len(results),
            (results[0].payload if results else None),
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

            point_document_id = payload.get("document_id")
            point_parent_id = payload.get("parent_document_id")

            # Scope to a single uploaded document when requested.
            # An image extracted from a PDF is indexed under its
            # own document_id but keeps parent_document_id pointing
            # at the PDF, so match on either.
            if document_id:
                if (
                     str(point_document_id) != str(document_id)
                      and str(point_parent_id) != str(document_id)
                ):
                      continue

            documents.append(
                {
                    "filename": payload.get("filename"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": text,
                    "document_id":str(point_document_id)
                    if point_document_id
                    else None,
                    "parent_document_id": str(point_parent_id)
                    if point_parent_id
                    else None,
                    # "parent_document_id": point_parent_id,
                    "content_type": payload.get("content_type"),
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