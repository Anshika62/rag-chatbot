import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repository.conversation_repo import (
    get_conversation,
    get_last_10_messages,
    create_message,
)

from app.service.external.LLM_service import (
    generate_answer,
    generate_answer_stream,
)


logger = logging.getLogger(__name__)


def query_documents(
    question: str,
    db: Session,
    user_id: str,
    conversation_id: str,
):
    try:
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty",
            )

        question = question.strip()

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in previous_messages
        ]

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question,
        )

        answer = generate_answer(
            question=question,
            chat_history=chat_history,
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        return {
            "conversation_id": conversation_id,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "question": question,
            "answer": answer,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Conversation query failed: "
            "conversation_id=%s user_id=%s",
            conversation_id,
            user_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process conversation query",
        ) from exc


def query_documents_stream(
    question: str,
    db: Session,
    user_id: str,
    conversation_id: str,
):
    try:
        if not question or not question.strip():
            yield {
                "event": "error",
                "success": False,
                "error_code": "BAD_REQUEST",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Question cannot be empty",
            }
            return

        question = question.strip()

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if not conversation:
            yield {
                "event": "error",
                "success": False,
                "error_code": "NOT_FOUND",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Conversation not found",
            }
            return

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
        )

        chat_history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in previous_messages
        ]

        yield {
            "event": "start",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": "",
        }

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question,
        )

        full_answer = ""

        for chunk in generate_answer_stream(
            question=question,
            chat_history=chat_history,
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
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
                "text_content": full_answer,
            }

        if not full_answer.strip():
            full_answer = "I was unable to generate a response."

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
        )

        yield {
            "event": "done",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "delta": None,
            "text_content": full_answer,
        }

    except HTTPException as exc:
        logger.exception(
            "QUERY STREAM HTTP ERROR: conversation_id=%s user_id=%s",
            conversation_id,
            user_id,
        )

        yield {
            "event": "error",
            "success": False,
            "error_code": "REQUEST_ERROR",
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": str(exc.detail),
        }

    except Exception as exc:
        logger.exception(
            "QUERY STREAM ERROR: conversation_id=%s user_id=%s error=%s",
            conversation_id,
            user_id,
            str(exc),
        )

        yield {
            "event": "error",
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "conversation_id": conversation_id,
            "message_id": None,
            "delta": None,
            "text_content": "Internal server error",
        }