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
    latitude: float | None = None,
    longitude: float | None = None,
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

        # ====================================================
        # RESOLVE + PERSIST CONVERSATION LOCATION
        # ====================================================
        #
        # First request:
        #   frontend sends latitude + longitude
        #   -> save them in Conversation
        #
        # Next request:
        #   frontend does not send coordinates
        #   -> load them from Conversation
        #
        # Only update the stored location when BOTH coordinates
        # are available. This prevents partial location updates.
        # ====================================================

        if latitude is not None and longitude is not None:

            if (
                conversation.latitude != latitude
                or conversation.longitude != longitude
            ):
                conversation.latitude = latitude
                conversation.longitude = longitude

                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                logger.info(
                    "Conversation location persisted: "
                    "conversation_id=%s latitude=%s longitude=%s",
                    conversation_id,
                    latitude,
                    longitude,
                )

        else:

            latitude = conversation.latitude
            longitude = conversation.longitude

            logger.info(
                "Using persisted conversation location: "
                "conversation_id=%s latitude=%s longitude=%s",
                conversation_id,
                latitude,
                longitude,
            )

        # ====================================================
        # GET CHAT HISTORY
        # ====================================================

        previous_messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
        )

        # ====================================================
        # DETECT LOCATION CONTINUATION
        #
        # When the first request needs the user's location, the
        # original user message is already stored in the database.
        # The frontend then sends the same question again together
        # with latitude/longitude after the user selects a location.
        #
        # Do NOT create another user-message row for that continuation.
        # A continuation is identified safely by all three conditions:
        #   1. real coordinates are supplied on this request,
        #   2. the latest DB message is a user message, and
        #   3. its question matches the current question.
        #
        # A normal repeated question is not affected because a
        # completed turn ends with an assistant message.
        # ====================================================

        normalized_question = question.casefold().strip()
        last_message = previous_messages[-1] if previous_messages else None

        is_location_continuation = bool(
            latitude is not None
            and longitude is not None
            and last_message is not None
            and str(last_message.role).lower() in {"user", "human"}
            and str(last_message.content or "").casefold().strip()
            == normalized_question
        )

        if is_location_continuation:
            logger.info(
                "LOCATION CONTINUATION DETECTED: "
                "conversation_id=%s message_id=%s",
                conversation_id,
                last_message.id,
            )

        chat_history_messages = previous_messages

        # The current question is already the latest user message in
        # a location continuation. Do not pass that same message twice
        # as both history and the current question.
        if is_location_continuation:
            chat_history_messages = previous_messages[:-1]

        chat_history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in chat_history_messages
        ]

        # ====================================================
        # START EVENT
        # ====================================================

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

        # ====================================================
        # SAVE / REUSE USER MESSAGE
        #
        # A location continuation must reuse the original user
        # message. Creating a new row here would make the same
        # question appear twice in the database and, after refresh,
        # twice in the chat history.
        # ====================================================

        if is_location_continuation:
            user_message = last_message
            logger.info(
                "REUSING USER MESSAGE FOR LOCATION CONTINUATION: "
                "conversation_id=%s message_id=%s",
                conversation_id,
                user_message.id,
            )
        else:
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

            # ====================================================
            # THINKING
            # ====================================================

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

            # ====================================================
            # LOCATION REQUEST
            # ====================================================

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

            # ====================================================
            # MAP LOCATION
            # ====================================================

            if piece_type == "map_location":

                yield {
                    "event": "map_location",
                    "success": True,
                    "error_code": None,
                    "conversation_id": conversation_id,
                    "message_id": user_message.id,
                    "delta": None,
                    "text_content": full_answer,
                    "images": [],
                    "latitude": piece.get("latitude"),
                    "longitude": piece.get("longitude"),
                    "name": piece.get("name"),
                    "address": piece.get("address"),
                }

                continue

            # ====================================================
            # NORMAL ANSWER DELTA
            # ====================================================

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
        # GENERATE FOLLOW-UP SUGGESTIONS
        #
        # Compatibility-first: the existing answer streaming above is
        # untouched. Suggestions are generated after the full answer
        # is available, then emitted BEFORE `done` so SSE clients that
        # stop processing after `done` cannot lose the suggestions.
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
            # Suggestion failure must never fail the completed answer.

        # ====================================================
        # DONE EVENT
        #
        # Keep this LAST so `done` remains the final completion signal.
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