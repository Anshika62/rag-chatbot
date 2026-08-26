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
<<<<<<< HEAD
        collection_name: str = "pdf_documents",
=======
        collection_name: str = "documents",
>>>>>>> origin/main
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
<<<<<<< HEAD
            ("conversation_id", PayloadSchemaType.KEYWORD),
            ("user_id", PayloadSchemaType.KEYWORD),
            ("message_id", PayloadSchemaType.KEYWORD),
            ("type", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("parent_document_id", PayloadSchemaType.KEYWORD),
            ("content_type", PayloadSchemaType.KEYWORD),
=======
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
>>>>>>> origin/main
        ]

        for field_name, field_schema in indexes:

            try:

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

                print(f"Qdrant index created: {field_name} -> {field_schema}")

            except Exception as exc:

                error_message = str(exc).lower()

<<<<<<< HEAD
                if "already exists" in error_message or "already exist" in error_message:
                    print(f"Qdrant index already exists: {field_name}")
                else:
                    print(f"Failed to create Qdrant index for '{field_name}': {e}")
                    raise
=======
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
>>>>>>> origin/main

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

<<<<<<< HEAD
        parent_document_id = (
            str(parent_document_id)
            if parent_document_id
            else document_id
        )
=======
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
>>>>>>> origin/main

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            if not chunk or not str(chunk).strip():
                continue

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "filename": filename,
<<<<<<< HEAD
                        "chunk_index": i,
                        "text": str(chunk),
=======
                        "chunk_index": index,
                        "text": chunk,
>>>>>>> origin/main
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

<<<<<<< HEAD
        for message, embedding in zip(messages, embeddings):
=======
        for message, embedding in zip(
            messages,
            embeddings,
        ):
>>>>>>> origin/main

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
<<<<<<< HEAD
            raise ValueError("No conversation messages available to store")
=======
            raise ValueError(
                "No conversation messages available "
                "to index"
            )
>>>>>>> origin/main

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

<<<<<<< HEAD
    def _to_flat_vector(self, query_embedding):
        """
        embedding_manager.generate_embedding() always returns a 2D
        numpy array (shape (1, 384)) even for a single query text.
        Qdrant's query_points needs a flat 384-float list for this
        collection (regular, non-multi vector config). This always
        extracts a single flat vector regardless of input shape.
        """

        if hasattr(query_embedding, "ndim"):
            if query_embedding.ndim == 2:
                return query_embedding[0].tolist()
            return query_embedding.tolist()

        if isinstance(query_embedding, list) and query_embedding:
            first = query_embedding[0]
            if hasattr(first, "tolist"):
                return first.tolist()
            if isinstance(first, (list, tuple)):
                return list(first)

        return query_embedding
=======
    # ============================================================
    # DOCUMENT SEARCH
    # ============================================================
>>>>>>> origin/main

    def search(
        self,
        query_embedding,
        user_id: str,
        conversation_id: Optional[str] = None,
<<<<<<< HEAD
        top_k: int = 5,
        document_id: Optional[str] = None,
        content_type: Optional[str] = None,
=======
        document_id: Optional[str] = None,
        top_k: int = 3,
>>>>>>> origin/main
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
<<<<<<< HEAD
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="type", match=MatchValue(value="document")),
        ]

        if conversation_id:
=======
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

>>>>>>> origin/main
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
<<<<<<< HEAD
                    match=MatchValue(value=str(conversation_id)),
                )
            )

        if content_type:
            must_conditions.append(
                FieldCondition(
                    key="content_type",
                    match=MatchValue(value=str(content_type)),
                )
            )

        if document_id:

            document_id = str(document_id)

            document_filter = Filter(
                must=must_conditions,
                should=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="parent_document_id", match=MatchValue(value=document_id)),
                ],
            )

        else:

            document_filter = Filter(must=must_conditions)

        query_vector = self._to_flat_vector(query_embedding)
=======
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
>>>>>>> origin/main

        # --------------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------------

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=document_filter,
            limit=top_k,
<<<<<<< HEAD
        )

        return results.points or []
=======
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
>>>>>>> origin/main

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
<<<<<<< HEAD
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                FieldCondition(key="type", match=MatchValue(value="conversation")),
            ]
=======
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
>>>>>>> origin/main
        )

        query_vector = self._to_flat_vector(query_embedding)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=history_filter,
            limit=top_k,
        )

        return results.points or []