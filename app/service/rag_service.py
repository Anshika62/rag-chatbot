import os
import logging

import pdfplumber

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.embedding import EmbeddingManager
from app.utils.vector_store import Vectorstore

from app.repository.document_repo import (
    create_document,
    create_chunks
)

from app.repository.conversation_repo import (
    get_conversation,
    get_last_10_messages,
    create_message
)

from app.service.external.LLM_service import (
    generate_answer,
    generate_answer_stream
)


logger = logging.getLogger(__name__)


# ============================================================
# INITIALIZATION
# ============================================================

embedding_manager = EmbeddingManager()

vector_store = Vectorstore(
    collection_name="pdf_documents"
)

UPLOAD_DIR = "Uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(
    file,
    db: Session,
    conversation_id: int,
    user_id: int
):
    try:

        # ----------------------------------------------------
        # 1. Validate file
        # ----------------------------------------------------

        if not file or not file.filename:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is required"
            )

        # ----------------------------------------------------
        # 2. Save uploaded file
        # ----------------------------------------------------

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as f:

            f.write(
                file.file.read()
            )

        # ----------------------------------------------------
        # 3. Extract PDF text
        # ----------------------------------------------------

        text = ""

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                text += (
                    page.extract_text()
                    or ""
                )

        if not text.strip():

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file"
            )

        # ----------------------------------------------------
        # 4. Split document
        # ----------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = text_splitter.split_text(
            text
        )

        if not chunks:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid chunks generated from document"
            )

        logger.info(
            "Document chunking completed: "
            "filename=%s, chunks=%s",
            file.filename,
            len(chunks)
        )

        # ----------------------------------------------------
        # 5. Generate embeddings
        # ----------------------------------------------------

        embeddings = (
            embedding_manager.generate_embedding(
                chunks
            )
        )

        # ----------------------------------------------------
        # 6. Store vectors
        # ----------------------------------------------------

        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename,
            conversation_id=conversation_id,
            user_id=user_id
        )

        # ----------------------------------------------------
        # 7. Save document in DB
        # ----------------------------------------------------

        document = create_document(
            db=db,
            file_name=file.filename,
            user_id=user_id,
            conversation_id=conversation_id
        )

        # ----------------------------------------------------
        # 8. Save chunks in DB
        # ----------------------------------------------------

        create_chunks(
            db=db,
            doc_id=document.id,
            chunks=chunks
        )

        return {
            "message": "Document processed successfully",
            "document_id": document.id,
            "conversation_id": conversation_id,
            "filename": file.filename,
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Document processing failed: "
            "conversation_id=%s, user_id=%s",
            conversation_id,
            user_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to process document: {str(exc)}"
        )


# ============================================================
# NORMAL RAG QUERY
# ============================================================

def query_documents(
    question: str,
    db: Session,
    user_id: int,
    conversation_id: int
):
    try:

        # ----------------------------------------------------
        # 1. Validate question
        # ----------------------------------------------------

        if not question or not question.strip():

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        question = question.strip()

        # ----------------------------------------------------
        # 2. Validate conversation
        # ----------------------------------------------------

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # ----------------------------------------------------
        # 3. Get conversation history
        # ----------------------------------------------------

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in previous_messages
        ]

        # ----------------------------------------------------
        # 4. Generate query embedding
        # ----------------------------------------------------

        query_embedding = (
            embedding_manager.generate_embedding(
                [question]
            )
        )

        # ----------------------------------------------------
        # 5. Search relevant documents
        # ----------------------------------------------------

        results = vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=3
        )

        # ----------------------------------------------------
        # 6. Extract context
        # ----------------------------------------------------

        contexts = [
            point.payload.get("text", "")
            for point in results
            if point.payload.get("text")
        ]

        context = "\n\n".join(
            contexts
        )

        logger.info(
            "RAG query: "
            "conversation_id=%s, "
            "user_id=%s, "
            "contexts=%s",
            conversation_id,
            user_id,
            len(contexts)
        )

        # ----------------------------------------------------
        # 7. Generate answer
        # ----------------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            chat_history=chat_history,
            db=db,
            conversation_id=conversation_id
        )

        # ----------------------------------------------------
        # 8. Save user message
        # ----------------------------------------------------

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        # ----------------------------------------------------
        # 9. Save assistant message
        # ----------------------------------------------------

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer
        )

        return {
            "conversation_id": conversation_id,
            "conversation_number": (
                conversation.conversation_number
            ),

            "user_message_id": user_message.id,
            "user_message_number": (
                user_message.message_number
            ),

            "assistant_message_id": assistant_message.id,
            "assistant_message_number": (
                assistant_message.message_number
            ),

            "question": question,
            "answer": answer,
            "contexts": contexts
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "RAG query failed: "
            "conversation_id=%s, user_id=%s",
            conversation_id,
            user_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process RAG query"
        ) from exc


# ============================================================
# STREAMING RAG QUERY
# ============================================================

def query_documents_stream(
    question: str,
    db: Session,
    user_id: int,
    conversation_id: int
):
    """
    Streaming RAG + conversation tool flow.

    Events:

        start
        delta
        done
        error
    """

    conversation = None

    try:

        # ----------------------------------------------------
        # 1. Validate question
        # ----------------------------------------------------

        if not question or not question.strip():

            yield {
                "event": "error",
                "success": False,
                "error_code": "BAD_REQUEST",

                "conversation_id": conversation_id,
                "conversation_number": None,

                "message_id": None,
                "message_number": None,

                "delta": None,
                "text_content": "Question cannot be empty"
            }

            return

        question = question.strip()

        # ----------------------------------------------------
        # 2. Validate conversation
        # ----------------------------------------------------

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:

            yield {
                "event": "error",
                "success": False,
                "error_code": "NOT_FOUND",

                "conversation_id": conversation_id,
                "conversation_number": None,

                "message_id": None,
                "message_number": None,

                "delta": None,
                "text_content": "Conversation not found"
            }

            return

        # ----------------------------------------------------
        # 3. Get history
        # ----------------------------------------------------

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in previous_messages
        ]

        # ----------------------------------------------------
        # 4. Generate query embedding
        # ----------------------------------------------------

        query_embedding = (
            embedding_manager.generate_embedding(
                [question]
            )
        )

        # ----------------------------------------------------
        # 5. Search documents
        # ----------------------------------------------------

        results = vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=3
        )

        # ----------------------------------------------------
        # 6. Extract contexts
        # ----------------------------------------------------

        contexts = [
            point.payload.get("text", "")
            for point in results
            if point.payload.get("text")
        ]

        context = "\n\n".join(
            contexts
        )

        logger.info(
            "Starting conversation stream: "
            "conversation_id=%s, "
            "user_id=%s, "
            "conversation_number=%s, "
            "contexts=%s",
            conversation_id,
            user_id,
            conversation.conversation_number,
            len(contexts)
        )

        # ----------------------------------------------------
        # 7. Start event
        # ----------------------------------------------------

        yield {
            "event": "start",
            "success": True,
            "error_code": None,

            "conversation_id": conversation_id,
            "conversation_number": (
                conversation.conversation_number
            ),

            "message_id": None,
            "message_number": None,

            "delta": None,
            "text_content": ""
        }

        # ----------------------------------------------------
        # 8. Save user message
        # ----------------------------------------------------

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question
        )

        # ----------------------------------------------------
        # 9. Generate streaming answer
        # ----------------------------------------------------

        full_answer = ""

        for chunk in generate_answer_stream(
            question=question,
            context=context,
            chat_history=chat_history,
            db=db,
            conversation_id=conversation_id
        ):

            if not chunk:
                continue

            full_answer += chunk

            yield {
                "event": "delta",
                "success": True,
                "error_code": None,

                "conversation_id": conversation_id,
                "conversation_number": (
                    conversation.conversation_number
                ),

                "message_id": user_message.id,
                "message_number": (
                    user_message.message_number
                ),

                "delta": chunk,
                "text_content": full_answer
            }

        # ----------------------------------------------------
        # 10. Empty response protection
        # ----------------------------------------------------

        if not full_answer.strip():

            full_answer = (
                "I was unable to generate a response."
            )

        # ----------------------------------------------------
        # 11. Save assistant message
        # ----------------------------------------------------

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer
        )

        # ----------------------------------------------------
        # 12. Done event
        # ----------------------------------------------------

        yield {
            "event": "done",
            "success": True,
            "error_code": None,

            "conversation_id": conversation_id,
            "conversation_number": (
                conversation.conversation_number
            ),

            "message_id": assistant_message.id,
            "message_number": (
                assistant_message.message_number
            ),

            "delta": None,
            "text_content": full_answer
        }

        logger.info(
            "Conversation stream completed: "
            "conversation_id=%s, "
            "conversation_number=%s",
            conversation_id,
            conversation.conversation_number
        )

    except HTTPException as exc:

        logger.exception(
            "QUERY STREAM HTTP ERROR: "
            "conversation_id=%s, user_id=%s, error=%s",
            conversation_id,
            user_id,
            exc.detail
        )

        yield {
            "event": "error",
            "success": False,
            "error_code": "REQUEST_ERROR",

            "conversation_id": conversation_id,
            "conversation_number": (
                conversation.conversation_number
                if conversation
                else None
            ),

            "message_id": None,
            "message_number": None,

            "delta": None,
            "text_content": exc.detail
        }

    except Exception as exc:

        logger.exception(
            "QUERY STREAM ERROR: "
            "conversation_id=%s, user_id=%s, error=%s",
            conversation_id,
            user_id,
            str(exc)
        )

        yield {
            "event": "error",
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",

            "conversation_id": conversation_id,
            "conversation_number": (
                conversation.conversation_number
                if conversation
                else None
            ),

            "message_id": None,
            "message_number": None,

            "delta": None,
            "text_content": "Internal server error"
        }