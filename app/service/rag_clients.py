"""
Shared RAG clients (embedding model + vector store).

Both doc_service.py (upload/processing) and rag_service.py
(query/answering) import from HERE instead of creating their
own instances. This avoids initializing the Qdrant collection
twice and keeps a single source of truth.
"""

from app.utils.embedding import EmbeddingManager
from app.utils.vector_store import Vectorstore


embedding_manager = EmbeddingManager()

vector_store = Vectorstore(
    collection_name="documents"
)