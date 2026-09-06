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

from app.core.database import get_db
from app.core.response import success_response
from app.core.dependency import (
    get_current_user,
    get_current_conversation,
)

from app.repository.conversation_repo import (
    create_conversation,
    get_all_messages,
    get_conversation,
    get_conversations_by_user,
)

from app.schemas.conversation_schema import (
    ConversationTitleUpdate,
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
# NORMALIZE LOCATION FROM REQUEST
# ============================================================
#
# Frontend may send location in different forms:
#
# 1. Flat:
#       latitude
#       longitude
#       address
#       full_address
#
# 2. Nested:
#       location: {
#           latitude,
#           longitude,
#           address
#       }
#
# 3. Nested:
#       coordinates: {
#           latitude,
#           longitude,
#           address
#       }
#
# Priority:
#       flat fields
#       -> location
#       -> coordinates
#
# Address:
#       address
#       -> full_address
#
# ============================================================


def _normalize_request_location(request: QueryRequest):

    def _pick(source: dict | None):

        if not source:
            return None, None, None

        latitude = source.get("latitude")
        longitude = source.get("longitude")

        address = (
            source.get("address")
            or source.get("full_address")
        )

        return (
            latitude,
            longitude,
            address,
        )

    # --------------------------------------------------------
    # First check flat fields
    # --------------------------------------------------------

    latitude = request.latitude
    longitude = request.longitude

    address = (
        request.address
        or request.full_address
    )

    # --------------------------------------------------------
    # If flat coordinates are incomplete,
    # check nested location / coordinates.
    # --------------------------------------------------------

    if latitude is None or longitude is None:

        for source in (
            request.location,
            request.coordinates,
        ):

            (
                nested_latitude,
                nested_longitude,
                nested_address,
            ) = _pick(source)

            if (
                nested_latitude is not None
                and nested_longitude is not None
            ):

                latitude = nested_latitude
                longitude = nested_longitude

                if not address:
                    address = nested_address

                break

    # --------------------------------------------------------
    # Coordinates exist but address is missing.
    # Try nested objects for address.
    # --------------------------------------------------------

    elif not address:

        for source in (
            request.location,
            request.coordinates,
        ):

            (
                _,
                _,
                nested_address,
            ) = _pick(source)

            if nested_address:

                address = nested_address
                break

    return (
        latitude,
        longitude,
        address,
    )


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

    # ========================================================
    # CREATE NEW CONVERSATION
    # ========================================================
    #
    # New conversation when:
    #
    #   is_new_conv=True
    #
    # OR
    #
    #   conversation_id is missing
    #
    # ========================================================

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

        conversation_id = str(
            conversation.id
        )

    # ========================================================
    # EXISTING CONVERSATION
    # ========================================================

    else:

        conversation = get_conversation(
            db=db,
            conversation_id=str(
                conversation_id
            ),
            user_id=str(user.id),
        )

        if not conversation:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        conversation_id = str(
            conversation.id
        )

    # ========================================================
    # NORMALIZE LOCATION
    # ========================================================
    #
    # IMPORTANT:
    #
    # This API layer does NOT save/load conversation location.
    #
    # It only extracts location from the current request and
    # passes it to rag_service.
    #
    # rag_service is responsible for:
    #
    #   1. Saving new coordinates to Conversation.
    #   2. Loading saved coordinates when request has none.
    #
    # ========================================================

    (
        resolved_latitude,
        resolved_longitude,
        resolved_address,
    ) = _normalize_request_location(
        request
    )

    # ========================================================
    # DEBUG REQUEST
    # ========================================================

    logger.info(
        "CONVERSATION REQUEST | "
        "question=%s | "
        "conversation_id=%s | "
        "document_id=%s | "
        "is_new_conv=%s | "
        "latitude=%s | "
        "longitude=%s | "
        "address=%s",
        request.question,
        conversation_id,
        request.document_id,
        request.is_new_conv,
        resolved_latitude,
        resolved_longitude,
        resolved_address,
    )

    # ========================================================
    # SSE EVENT GENERATOR
    # ========================================================

    def event_generator():

        try:

            for event_data in query_documents_stream(
                question=request.question,
                db=db,
                user_id=str(user.id),
                conversation_id=conversation_id,
                document_id=request.document_id,

                # Current request location.
                #
                # If None, rag_service will load the
                # previously persisted conversation location.
                latitude=resolved_latitude,
                longitude=resolved_longitude,
                address=resolved_address,
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

    # ========================================================
    # RETURN SSE STREAM
    # ========================================================

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
            "conversation_id": str(
                conversation.id
            ),
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
            ),
            "updated_at": (
                conversation.updated_at.isoformat()
            ),
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
    conversation=Depends(
        get_current_conversation
    ),
    db: Session = Depends(get_db),
):

    messages = get_all_messages(
        db=db,
        conversation_id=str(
            conversation.id
        ),
    )

    def _parse_images(raw_images):

        if not raw_images:
            return []

        try:

            return json.loads(
                raw_images
            )

        except (
            TypeError,
            ValueError,
        ):

            logger.warning(
                "Unable to parse stored images JSON "
                "for conversation_id=%s",
                conversation.id,
            )

            return []

    return success_response(
        message="Conversation fetched successfully",
        data={
            "conversation_id": str(
                conversation.id
            ),
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
            ),
            "updated_at": (
                conversation.updated_at.isoformat()
            ),
            "messages": [
                {
                    "message_id": str(
                        message.id
                    ),
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
    conversation=Depends(
        get_current_conversation
    ),
    db: Session = Depends(get_db),
):

    conversation.title = request.title

    db.commit()
    db.refresh(conversation)

    return success_response(
        message="Title updated successfully",
        data={
            "conversation_id": str(
                conversation.id
            ),
            "title": conversation.title,
        },
        status_code=status.HTTP_200_OK,
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================


@router.delete("/{conversation_id}")
def delete_conversation_endpoint(
    conversation=Depends(
        get_current_conversation
    ),
    db: Session = Depends(get_db),
):

    db.delete(conversation)
    db.commit()

    return success_response(
        message="Conversation deleted successfully",
        data=None,
        status_code=status.HTTP_200_OK,
    )