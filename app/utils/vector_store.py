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

        try:
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

            # Ensure required payload indexes exist
            # (needed even if collection already existed before)
            self._ensure_indexes()

        except Exception:
            raise

    def _ensure_indexes(self):
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="conversation_id",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception:
            pass  # index already exists, safe to ignore

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception:
            pass

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

    def add_documents(
        self,
        chunks: List[str],
        embeddings,
        filename: str,
        conversation_id: int
    ):

        try:
            points = []

            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            ):

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "filename": filename,
                            "chunk_index": i,
                            "text": chunk,
                            "type": "document",
                            "conversation_id": conversation_id
                        }
                    )
                )

            if not points:
                raise ValueError(
                    "No document points available to store"
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        except Exception:
            raise

    def add_conversation_messages(
        self,
        messages,
        embeddings
    ):

        try:
            points = []

            for message, embedding in zip(
                messages,
                embeddings
            ):

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "conversation_id": message["conversation_id"],
                            "user_id": message["user_id"],
                            "message_id": message["message_id"],
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

        except Exception:
            raise

    def search(
        self,
        query_embedding,
        conversation_id: int,
        top_k: int = 3
    ):

        try:
            document_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(
                            value=conversation_id
                        )
                    ),
                    FieldCondition(
                        key="type",
                        match=MatchValue(
                            value="document"
                        )
                    )
                ]
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding[0].tolist(),
                query_filter=document_filter,
                limit=top_k
            )

            return results.points

        except Exception:
            raise

    def search_conversation_history(
        self,
        query_embedding,
        user_id: int,
        conversation_id: int,
        top_k: int = 5
    ):

        try:
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

        except Exception:
            raise