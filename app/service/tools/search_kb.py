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
):
    @tool
    def search_knowledge_base(
        query: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Search uploaded documents and indexed knowledge base for relevant information."""

        if not query or not query.strip():
            raise ValueError(
                "Knowledge-base search query cannot be empty."
            )

        if limit < 1:
            raise ValueError(
                "Search result limit must be at least 1."
            )

        limit = min(limit, 4)

        logger.info(
            "KB SEARCH START: query=%s, limit=%s, user_id=%s, conversation_id=%s",
            query,
            limit,
            user_id,
            conversation_id,
        )

        query_embedding = embedding_manager.generate_embedding(
            [query.strip()]
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=limit,
        )

        documents = []

        for point in results:
            payload = point.payload or {}
            text = payload.get("text", "")

            if not text:
                continue

            documents.append(
                {
                    "filename": payload.get("filename"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": text,
                }
            )

        logger.info(
            "KB SEARCH COMPLETE: count=%s, filenames=%s",
            len(documents),
            [item.get("filename") for item in documents],
        )

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