import os
import uuid
from typing import List

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)


load_dotenv()


class Vectorstore:

    def __init__(
        self,
        collection_name: str = "pdf_documents"
    ):
        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL or QDRANT_API_KEY is not configured"
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        self._create_collection()

    def _create_collection(self):

        collections = self.client.get_collections()

        existing_collections = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name not in existing_collections:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

        self._ensure_indexes()

    def _ensure_indexes(self):

        indexes = [
            (
                "conversation_id",
                PayloadSchemaType.KEYWORD
            ),
            (
                "user_id",
                PayloadSchemaType.KEYWORD
            ),
            (
                "message_id",
                PayloadSchemaType.KEYWORD
            ),
            (
                "type",
                PayloadSchemaType.KEYWORD
            )
        ]

        for field_name, field_schema in indexes:

            try:

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )

                print(
                    f"Qdrant index created: "
                    f"{field_name} -> {field_schema}"
                )

            except Exception as e:

                error_message = str(e).lower()

                if (
                    "already exists" in error_message
                    or "already exist" in error_message
                ):
                    print(
                        f"Qdrant index already exists: "
                        f"{field_name}"
                    )

                else:
                    print(
                        f"Failed to create Qdrant index "
                        f"for '{field_name}': {e}"
                    )

                    raise

    def add_documents(
        self,
        chunks: List[str],
        embeddings,
        filename: str,
        conversation_id: str,
        user_id: str,
        document_id: str,
        content_type: str = "text",
        parent_document_id: str | None = None,
    ):

        conversation_id = str(conversation_id)
        user_id = str(user_id)
        document_id = str(document_id)
        parent_document_id = str(parent_document_id) if parent_document_id else document_id

        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk,
                        "type": "document",
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "document_id": document_id,
                        "parent_document_id": parent_document_id,
                        "content_type": content_type,
                    }
                )
            )

        if not points:
            raise ValueError("No document points available to store")

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def add_conversation_messages(
        self,
        messages,
        embeddings
    ):

        points = []

        for message, embedding in zip(
            messages,
            embeddings
        ):

            conversation_id = str(
                message["conversation_id"]
            )

            user_id = str(
                message["user_id"]
            )

            message_id = str(
                message["message_id"]
            )

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "message_id": message_id,
                        "role": message["role"],
                        "content": message["content"],
                        "type": "conversation"
                    }
                )
            )

        if not points:
            raise ValueError(
                "No conversation messages available to store"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )



    def search(
        self,
        query_embedding,
        user_id: str,
        conversation_id: str | None = None,
        top_k: int = 3
    ):

        user_id = str(user_id)

        must_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(
                    value=user_id
                )
            ),
            FieldCondition(
                key="type",
                match=MatchValue(
                    value="document"
                )
            )
        ]

        # Scope search to the current conversation whenever one is
        # supplied. Without this, KB search pulls chunks from ALL of
        # the user's conversations/documents, diluting top_k results
        # with irrelevant chunks — this was the main cause of the
        # knowledge-base/PDF answer quality regression.
        if conversation_id:
            must_conditions.append(
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(
                        value=str(conversation_id)
                    )
                )
            )

        document_filter = Filter(
            must=must_conditions
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=document_filter,
            limit=top_k
        )

        return results.points


    def search_conversation_history(
        self,
        query_embedding,
        user_id: str,
        conversation_id: str,
        top_k: int = 5
    ):

        user_id = str(user_id)
        conversation_id = str(conversation_id)

        history_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(
                        value=user_id
                    )
                ),
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(
                        value=conversation_id
                    )
                ),
                FieldCondition(
                    key="type",
                    match=MatchValue(
                        value="conversation"
                    )
                )
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=history_filter,
            limit=top_k
        )

        return results.points