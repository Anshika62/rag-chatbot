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

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL or QDRANT_API_KEY is not configured"
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
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
                    distance=Distance.COSINE,
                ),
            )

        self._ensure_indexes()

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

                print(f"Qdrant index created: {field_name} -> {field_schema}")

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
                    continue

                raise

    # ============================================================
    # VECTOR NORMALIZATION HELPER
    # ============================================================
    #
    # NOTE: This method is called by search() and
    # search_conversation_history() but was not present in the
    # file as provided. Added here so the file is importable and
    # runnable. If you already have a _to_flat_vector
    # implementation elsewhere in your original file (above what
    # was pasted), replace this with your original version rather
    # than using this placeholder.
    # ============================================================

    def _to_flat_vector(self, embedding) -> List[float]:

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Handles embeddings shaped like [[...]] (batch of 1)
        if (
            isinstance(embedding, list)
            and len(embedding) == 1
            and isinstance(embedding[0], list)
        ):
            return embedding[0]

        return embedding

    # ============================================================
    # ADD DOCUMENTS
    # ============================================================

    def add_documents(
        self,
        chunks: List[str],
        embeddings,
        filename: str,
        conversation_id: str,
        user_id: str,
        document_id: str,
        content_type: str = "text",
        parent_document_id: Optional[str] = None,
    ):

        conversation_id = str(conversation_id)
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

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

            if not chunk or not str(chunk).strip():
                continue

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
                    },
                )
            )

        if not points:
            raise ValueError("No document points available to store")

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

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

            conversation_id = str(message["conversation_id"])
            user_id = str(message["user_id"])
            message_id = str(message["message_id"])

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
        content_type: Optional[str] = None,
        top_k: int = 3,
    ):
        """
        conversation_id=None:
            Only global documents (conversation_id IS NULL) are
            returned — used for document_id-scoped searches, where
            the caller (search_kb.py) does its own per-document
            filtering afterwards on a wider candidate set.

        conversation_id=<id>:
            Both this conversation's own documents AND global
            documents are returned (a global doc must be usable
            from any conversation).

        content_type:
            Optional filter ("text" or "image"). When set, only
            chunks indexed with that content_type are returned —
            lets an image-focused question skip straight to
            image-caption/OCR chunks instead of competing with
            text chunks on pure vector similarity.
        """

        user_id = str(user_id)

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

        if content_type:
            must_conditions.append(
                FieldCondition(
                    key="content_type",
                    match=MatchValue(
                        value=content_type,
                    ),
                ),
            )

        is_global_condition = IsNullCondition(
            is_null=PayloadField(
                key="conversation_id",
            ),
        )

        if conversation_id is not None:

            conversation_id = str(conversation_id)

            # AND(user_id, type[, content_type], OR(this conversation, global))
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(
                                value=conversation_id,
                            ),
                        ),
                        is_global_condition,
                    ],
                )
            )

        else:

            # Global-only: conversation_id must be null.
            must_conditions.append(is_global_condition)

        document_filter = Filter(
            must=must_conditions,
        )

        print(
            "QDRANT SEARCH | "
            f"user_id={user_id} | "
            f"conversation_id={conversation_id} | "
            f"content_type={content_type} | "
            f"top_k={top_k}"
        )

        # --------------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------------

        query_vector = self._to_flat_vector(query_embedding)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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
        conversation_id = str(conversation_id)

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

        query_vector = self._to_flat_vector(query_embedding)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=history_filter,
            limit=top_k,
        )

        return results.points or []