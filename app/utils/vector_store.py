import os
import uuid
from typing import List, Optional

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
    IsNullCondition,
    PayloadField,
)


load_dotenv()


class Vectorstore:

    def __init__(
        self,
        collection_name: str = "documents",
    ):
        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is not configured"
            )

        if not qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY is not configured"
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        self._create_collection()

    # ============================================================
    # COLLECTION
    # ============================================================

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
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_indexes()

    # ============================================================
    # PAYLOAD INDEXES
    # ============================================================

    def _ensure_indexes(self):

        indexes = [
            (
                "conversation_id",
                PayloadSchemaType.KEYWORD,
            ),
            (
                "user_id",
                PayloadSchemaType.KEYWORD,
            ),
            (
                "message_id",
                PayloadSchemaType.KEYWORD,
            ),
            (
                "type",
                PayloadSchemaType.KEYWORD,
            ),
            (
                "document_id",
                PayloadSchemaType.KEYWORD,
            ),
            (
                "parent_document_id",
                PayloadSchemaType.KEYWORD,
            ),
        ]

        for field_name, field_schema in indexes:

            try:

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

                print(
                    f"Qdrant index created: "
                    f"{field_name} -> {field_schema}"
                )

            except Exception as exc:

                error_message = str(exc).lower()

                if (
                    "already exists" in error_message
                    or "already exist" in error_message
                ):
                    print(
                        f"Qdrant index already exists: "
                        f"{field_name}"
                    )
                    continue

                raise

    # ============================================================
    # ADD DOCUMENTS
    # ============================================================

    def add_documents(
        self,
        chunks: List[str],
        embeddings,
        filename: str,
        conversation_id: Optional[str],
        user_id: str,
        document_id: str,
        content_type: str = "text",
        parent_document_id: Optional[str] = None,
    ):

        user_id = str(user_id)
        document_id = str(document_id)

        # IMPORTANT:
        #
        # Global document:
        #
        #     conversation_id = None
        #
        # Conversation document:
        #
        #     conversation_id = UUID
        #
        # Never convert None into "None".

        if conversation_id is not None:
            conversation_id = str(conversation_id)

        if parent_document_id is not None:
            parent_document_id = str(
                parent_document_id
            )
        else:
            parent_document_id = document_id

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "filename": filename,
                        "chunk_index": index,
                        "text": chunk,
                        "type": "document",
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "document_id": document_id,
                        "parent_document_id": parent_document_id,
                        "content_type": content_type,
                    },
                )
            )

        if not points:
            raise ValueError(
                "No document chunks available to index"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            "QDRANT DOCUMENT INDEXED | "
            f"document_id={document_id} | "
            f"conversation_id={conversation_id} | "
            f"user_id={user_id} | "
            f"filename={filename} | "
            f"chunks={len(points)}"
        )

    # ============================================================
    # ADD CONVERSATION MESSAGES
    # ============================================================

    def add_conversation_messages(
        self,
        messages,
        embeddings,
    ):

        points = []

        for message, embedding in zip(
            messages,
            embeddings,
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
                        "type": "conversation",
                    },
                )
            )

        if not points:
            raise ValueError(
                "No conversation messages available "
                "to index"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # ============================================================
    # DOCUMENT SEARCH
    # ============================================================

    def search(
        self,
        query_embedding,
        user_id: str,
        conversation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        top_k: int = 3,
    ):

        user_id = str(user_id)

        if conversation_id is not None:
            conversation_id = str(
                conversation_id
            )

        if document_id is not None:
            document_id = str(
                document_id
            )

        # --------------------------------------------------------
        # BASE SECURITY FILTER
        # --------------------------------------------------------

        must_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(
                    value=user_id,
                ),
            ),
            FieldCondition(
                key="type",
                match=MatchValue(
                    value="document",
                ),
            ),
        ]

        # --------------------------------------------------------
        # SPECIFIC DOCUMENT
        # --------------------------------------------------------
        #
        # When document_id is explicitly supplied:
        #
        #     user_id
        #     +
        #     type=document
        #     +
        #     document_id
        #
        # We intentionally do NOT apply conversation_id here.
        #
        # This allows a global document to be selected from
        # any conversation.
        # --------------------------------------------------------

        if document_id is not None:

            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    ),
                )
            )

            document_filter = Filter(
                must=must_conditions,
            )

        # --------------------------------------------------------
        # NORMAL CONVERSATION SEARCH
        # --------------------------------------------------------
        #
        # No document selected:
        #
        #     Global documents
        #             +
        #     Current conversation documents
        #
        # Other conversation documents are excluded.
        # --------------------------------------------------------

        elif conversation_id is not None:

            conversation_scope = [
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(
                        value=conversation_id,
                    ),
                ),
                IsNullCondition(
                    is_null=PayloadField(
                        key="conversation_id",
                    ),
                ),
            ]

            document_filter = Filter(
                must=must_conditions,
                should=conversation_scope,
            )

        # --------------------------------------------------------
        # GLOBAL ONLY
        # --------------------------------------------------------

        else:

            document_filter = Filter(
                must=[
                    *must_conditions,
                    IsNullCondition(
                        is_null=PayloadField(
                            key="conversation_id",
                        ),
                    ),
                ],
            )

        print(
            "QDRANT SEARCH | "
            f"user_id={user_id} | "
            f"conversation_id={conversation_id} | "
            f"document_id={document_id} | "
            f"top_k={top_k}"
        )

        # --------------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------------

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=document_filter,
            limit=top_k,
        )

        print(
            "QDRANT SEARCH RESULT | "
            f"count={len(results.points)}"
        )

        return results.points

    # ============================================================
    # DEBUG DOCUMENT
    # ============================================================
    #
    # This does NOT use vector similarity.
    #
    # It simply asks:
    #
    # "Does Qdrant contain any chunks with this document_id?"
    #
    # We need this to distinguish:
    #
    # 1. Document exists in DB but wasn't indexed in Qdrant
    #
    # from:
    #
    # 2. Document exists in Qdrant but search filter is wrong.
    #
    # ============================================================

    def debug_document_points(
        self,
        document_id: str,
    ):

        document_id = str(document_id)

        print("\n")
        print("=" * 80)
        print("QDRANT DOCUMENT DEBUG")
        print("=" * 80)
        print(
            "document_id:",
            document_id,
        )

        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id,
                        ),
                    ),
                ],
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points = result[0]

        print(
            "MATCHING POINTS:",
            len(points),
        )

        for point in points:

            print("-" * 80)

            print(
                "POINT ID:",
                point.id,
            )

            print(
                "PAYLOAD:",
                point.payload,
            )

        print("=" * 80)
        print("\n")

        return points

    # ============================================================
    # DEBUG ALL DOCUMENTS FOR USER
    # ============================================================

    def debug_user_documents(
        self,
        user_id: str,
    ):

        user_id = str(user_id)

        print("\n")
        print("=" * 80)
        print("QDRANT USER DOCUMENT DEBUG")
        print("=" * 80)
        print(
            "user_id:",
            user_id,
        )

        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(
                            value=user_id,
                        ),
                    ),
                    FieldCondition(
                        key="type",
                        match=MatchValue(
                            value="document",
                        ),
                    ),
                ],
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points = result[0]

        print(
            "DOCUMENT POINTS:",
            len(points),
        )

        for point in points:

            payload = point.payload or {}

            print("-" * 80)

            print(
                "filename:",
                payload.get("filename"),
            )

            print(
                "document_id:",
                payload.get("document_id"),
            )

            print(
                "conversation_id:",
                payload.get("conversation_id"),
            )

            print(
                "chunk_index:",
                payload.get("chunk_index"),
            )

        print("=" * 80)
        print("\n")

        return points

    # ============================================================
    # CONVERSATION HISTORY SEARCH
    # ============================================================

    def search_conversation_history(
        self,
        query_embedding,
        user_id: str,
        conversation_id: str,
        top_k: int = 5,
    ):

        user_id = str(user_id)
        conversation_id = str(
            conversation_id
        )

        history_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(
                        value=user_id,
                    ),
                ),
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(
                        value=conversation_id,
                    ),
                ),
                FieldCondition(
                    key="type",
                    match=MatchValue(
                        value="conversation",
                    ),
                ),
            ],
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=history_filter,
            limit=top_k,
        )

        return results.points