import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repository.conversation_repo import (
    get_conversation,
    get_last_10_messages,
    create_message,
)

from app.service.external.llm_service import (
    generate_answer,
    generate_answer_stream,
    generate_suggestions,
)


logger = logging.getLogger(__name__)


def query_documents(
    question: str,
    db: Session,
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
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

        images_output: list = []

        answer = generate_answer(
            question=question,
            chat_history=chat_history,
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            images_output=images_output,
            document_id=document_id,
        )

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            images=images_output,
        )

        return {
            "conversation_id": conversation_id,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "question": question,
            "answer": answer,
            "images": images_output,
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
    document_id: str | None = None,
    image_paths: list[str] | None = None,
    latitude: float | None = None,      # NEW
    longitude: float | None = None,     # NEW
    address: str | None = None,         
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
                "images": [],
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
                "images": [],
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
            "images": [],
        }

        user_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=question,
        )

        full_answer = ""
        images_output: list = []

        # ====================================================
        # STREAM LOOP
        #
        # generate_answer_stream() yields structured pieces:
        #
        #     {"type": "thinking", "content": "..."}
        #     {"type": "answer", "content": "..."}
        #
        # Thinking pieces are streamed to the frontend but are
        # never added to full_answer or persisted.
        #
        # Only answer pieces are stored as the final response.
        # ====================================================

        for piece in generate_answer_stream(
            question=question,
            chat_history=chat_history,
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            images_output=images_output,
            document_id=document_id,
            image_paths=image_paths,
            latitude=latitude,        
            longitude=longitude,      
            address=address,          

        ):
            if not piece:
                continue

            piece_type = piece.get("type", "answer")
            piece_content = piece.get("content")

            if not piece_content:
                continue

            if piece_type == "thinking":

                yield {
                    "event": "thinking",
                    "success": True,
                    "error_code": None,
                    "conversation_id": conversation_id,
                    "message_id": user_message.id,
                    "delta": piece_content,
                    "text_content": full_answer,
                    "images": [],
                }

                continue

            # ================================================
            # LOCATION REQUEST
            #
            # get_location was called by the agent. This is not
            # normal answer text streaming — it's a one-shot
            # signal telling the frontend to show a location-
            # selection UI. Still accumulated into full_answer
            # so the usual persist/"done" flow below saves it as
            # the assistant's message, same as any other turn.
            # ================================================

            if piece_type == "location_request":

                full_answer += piece_content

                yield {
                    "event": "location_request",
                    "success": True,
                    "error_code": None,
                    "conversation_id": conversation_id,
                    "message_id": user_message.id,
                    "delta": piece_content,
                    "text_content": full_answer,
                    "images": [],
                    "methods": piece.get(
                        "methods",
                        ["current_location", "search", "map"],
                    ),
                }

                continue

            full_answer += piece_content

            yield {
                "event": "delta",
                "success": True,
                "error_code": None,
                "conversation_id": conversation_id,
                "message_id": user_message.id,
                "delta": piece_content,
                "text_content": full_answer,
                "images": [],
            }

        # ====================================================
        # FINAL ANSWER CLEANUP
        # ====================================================

        if not full_answer.strip():

            full_answer = "I was unable to generate a response."

        else:

            # Remove accidental leading/trailing whitespace from
            # the final answer before persisting it.
            full_answer = full_answer.strip()

        # ====================================================
        # SAVE ASSISTANT MESSAGE
        # ====================================================

        assistant_message = create_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            images=images_output,
        )

        # ====================================================
        # EXISTING DONE EVENT
        #
        # IMPORTANT:
        # Keep this event before suggestion generation.
        # This means the frontend receives the completed answer
        # before we start sending suggestions.
        # ====================================================

        yield {
            "event": "done",
            "success": True,
            "error_code": None,
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "delta": None,
            "text_content": full_answer,
            "images": images_output,
        }

        # ====================================================
        # GENERATE FOLLOW-UP SUGGESTIONS
        #
        # IMPORTANT:
        # This happens AFTER the done event.
        #
        # Suggestion generation is an optional enhancement.
        # If it fails, the already completed answer remains
        # successful and no error event is sent.
        # ====================================================

        try:

            suggestions = generate_suggestions(
                question=question,
                answer=full_answer,
                chat_history=chat_history,
            )

            if suggestions:

                yield {
                    "event": "suggestions",
                    "success": True,
                    "error_code": None,
                    "conversation_id": conversation_id,
                    "message_id": assistant_message.id,
                    "delta": None,
                    "text_content": "",
                    "images": [],
                    "suggestions": suggestions,
                }

            else:

                logger.info(
                    "No suggestions generated: "
                    "conversation_id=%s",
                    conversation_id,
                )

        except Exception:

            logger.exception(
                "Suggestion generation failed: "
                "conversation_id=%s user_id=%s",
                conversation_id,
                user_id,
            )

            # Do NOT send an error event here.
            #
            # The answer was already completed successfully.
            # Suggestion failure must not make the chat request
            # appear to have failed.

    except HTTPException as exc:
        logger.exception(
            "QUERY STREAM HTTP ERROR: "
            "conversation_id=%s user_id=%s",
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
            "images": [],
        }

    except Exception as exc:
        logger.exception(
            "QUERY STREAM ERROR: "
            "conversation_id=%s user_id=%s error=%s",
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
            "images": [],
        }