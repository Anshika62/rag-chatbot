
import json
import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.schemas.location_schema import LocationSubmission
from app.core.database import get_db
from app.core.response import success_response
from app.core.dependency import (
    get_current_user,
    get_current_conversation,
)

from app.service.tools.geocode_tool import (
    find_location_on_map,
)

from app.repository.conversation_repo import (
    create_conversation,
    delete_conversation,
    get_all_messages,
    get_conversations_by_user,
    update_conversation_title,
)

from app.schemas.conversation_schema import (
    ConversationTitleUpdate,
)

from app.service.tools.location_tool import (
    get_location,
)

from app.service.tools.places_tool import (
    search_nearby_places,
)


from app.schemas.query_schema import QueryRequest

from app.service.external.llm_service import generate_title
from app.service.rag_service import query_documents_stream


router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
)

logger = logging.getLogger(__name__)


# ============================================================
# SEND MESSAGE
# ============================================================


@router.post("")
def send_message(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    conversation_id = request.conversation_id

    # --------------------------------------------------------
    # Create new conversation when frontend sends
    # is_new_conv=True.
    #
    # DEFENSIVE FALLBACK:
    # Some frontend call sites omit is_new_conv entirely (it
    # then defaults to False via Pydantic) while ALSO omitting
    # conversation_id. That combination is indistinguishable
    # from "start a new conversation" — there is no existing
    # conversation to validate against. Previously this fell
    # through to the "existing conversation" branch below and
    # hard-failed with 400 "conversation_id is required", even
    # though the user's intent was clearly to start a fresh
    # chat. Treating "conversation_id is missing" the same as
    # "is_new_conv=True" removes that failure mode without
    # weakening the explicit-True case at all — it only changes
    # behavior for requests that would otherwise have been a
    # guaranteed 400.
    # --------------------------------------------------------

    starting_new_conversation = (
        request.is_new_conv
        or conversation_id is None
    )

    if starting_new_conversation:

        title = generate_title(
            request.question,
        )

        conversation = create_conversation(
            db=db,
            user_id=user.id,
            title=title,
        )

        conversation_id = str(conversation.id)

    # --------------------------------------------------------
    # Verify existing conversation ownership
    # --------------------------------------------------------

    else:

        from app.repository.conversation_repo import get_conversation

        conversation = get_conversation(
            db=db,
            conversation_id=str(conversation_id),
            user_id=str(user.id),
        )

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        conversation_id = str(conversation.id)

    # --------------------------------------------------------
    # Debug request scope
    # --------------------------------------------------------
    #
    # document_id=None means:
    #
    #     Global documents
    #     +
    #     Current conversation documents
    #
    # document_id=<UUID> means:
    #
    #     Search that specific accessible document.
    # --------------------------------------------------------

    logger.info(
       "CONVERSATION REQUEST | "
       "question=%s | "
       "conversation_id=%s | "
       "document_id=%s | "
       "is_new_conv=%s (effective=%s) | "
       "latitude=%s | longitude=%s",
       request.question,
       conversation_id,
       request.document_id,
       request.is_new_conv,
       starting_new_conversation,
       request.latitude,
       request.longitude,)

    # --------------------------------------------------------
    # SSE event generator
    # --------------------------------------------------------

    def event_generator():

        try:

            for event_data in query_documents_stream(
                question=request.question,
                db=db,
                user_id=str(user.id),
                conversation_id=conversation_id,
                document_id=request.document_id,
                latitude=request.latitude,
                longitude=request.longitude,
                address=request.address,
            ):

                event_name = event_data.get(
                    "event",
                    "message",
                )

                yield (
                    f"event: {event_name}\n"
                    f"data: {json.dumps(event_data)}\n\n"
                )

        except Exception:

            logger.exception(
                "Conversation streaming failed: "
                "conversation_id=%s, user_id=%s",
                conversation_id,
                user.id,
            )

            error_data = {
                "event": "error",
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Internal server error",
                "images": [],
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(error_data)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# LIST CONVERSATIONS
# ============================================================


@router.get("")
def list_conversations(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    conversations = get_conversations_by_user(
        db=db,
        user_id=str(user.id),
    )

    data = [
        {
            "conversation_id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }
        for conversation in conversations
    ]

    return success_response(
        message="Conversations fetched successfully",
        data=data,
        status_code=status.HTTP_200_OK,
    )


# ============================================================
# GET CONVERSATION DETAIL
# ============================================================


@router.get("/{conversation_id}")
def get_conversation_detail(
    conversation=Depends(get_current_conversation),
    db: Session = Depends(get_db),
):
    messages = get_all_messages(
        db=db,
        conversation_id=str(conversation.id),
    )

    def _parse_images(raw_images):
        if not raw_images:
            return []

        try:
            return json.loads(raw_images)

        except (TypeError, ValueError):

            logger.warning(
                "Unable to parse stored images JSON for a message "
                "in conversation_id=%s",
                conversation.id,
            )

            return []

    return success_response(
        message="Conversation fetched successfully",
        data={
            "conversation_id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": [
                {
                    "message_id": str(message.id),
                    "role": message.role,
                    "content": message.content,
                    "images": _parse_images(
                        message.images
                    ),
                    "created_at": (
                        message.created_at.isoformat()
                    ),
                }
                for message in messages
            ],
        },
        status_code=status.HTTP_200_OK,
    )


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================


@router.patch("/{conversation_id}")
def update_title(
    request: ConversationTitleUpdate,
    conversation=Depends(get_current_conversation),
    db: Session = Depends(get_db),
):
    conversation.title = request.title

    db.commit()
    db.refresh(conversation)

    return success_response(
        message="Title updated successfully",
        data={
            "conversation_id": str(conversation.id),
            "title": conversation.title,
        },
        status_code=status.HTTP_200_OK,
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================


@router.delete("/{conversation_id}")
def delete_conversation_endpoint(
    conversation=Depends(get_current_conversation),
    db: Session = Depends(get_db),
):
    db.delete(conversation)
    db.commit()

    return success_response(
        message="Conversation deleted successfully",
        data=None,
        status_code=status.HTTP_200_OK,
    )

