import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repository.conversation_repo import (
    get_conversation,
    get_last_10_messages,
    create_message
)

from app.service.external.LLM_service import (
    generate_answer,
    generate_answer_stream
)

from app.service.rag_clients import embedding_manager, vector_store


logger = logging.getLogger(__name__)


# ============================================================
# NORMAL RAG QUERY
# ============================================================

def query_documents(
    question: str,
    db: Session,
    user_id: str,
    conversation_id: str
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

            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,

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
    user_id: str,
    conversation_id: str
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

                "message_id": None,

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

                "message_id": None,

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
            "contexts=%s",
            conversation_id,
            user_id,
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

            "message_id": None,

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

                "message_id": user_message.id,

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

            "message_id": assistant_message.id,

            "delta": None,
            "text_content": full_answer
        }

        logger.info(
            "Conversation stream completed: conversation_id=%s",
            conversation_id
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

            "message_id": None,

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

            "message_id": None,

            "delta": None,
            "text_content": "Internal server error"
        }