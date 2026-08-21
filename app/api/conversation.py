import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import verify_token
from app.core.response import success_response

from app.repository.user_repo import get_user_by_email
from app.repository.conversation_repo import (
    create_conversation,
    get_conversation,
    get_conversations_by_user,
    get_all_messages,
    update_conversation_title,
    delete_conversation
)

from app.schemas.conversation_schema import (
    QueryRequest,
    ConversationTitleUpdate
)

from app.service.external.LLM_service import generate_title
from app.service.rag_service import query_documents_stream


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"]
)

logger = logging.getLogger(__name__)


# ============================================================
# CURRENT USER
# ============================================================

def _get_current_user(
    db: Session,
    email: str
):
    user = get_user_by_email(
        db=db,
        email=email
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user


# ============================================================
# SEND MESSAGE / STREAM RESPONSE
# ============================================================

@router.post("")
def send_message(
    request: QueryRequest,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    user = _get_current_user(
        db=db,
        email=email
    )

    conversation_id = request.conversation_id

    # --------------------------------------------------------
    # Create new conversation
    # --------------------------------------------------------

    if conversation_id is None:

        title = generate_title(
            request.question
        )

        conversation = create_conversation(
            db=db,
            user_id=user.id,
            title=title
        )

        conversation_id = conversation.id

    # --------------------------------------------------------
    # Existing conversation
    # --------------------------------------------------------

    else:

        conversation = get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user.id
        )

        if not conversation:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

    # --------------------------------------------------------
    # SSE EVENT GENERATOR
    # --------------------------------------------------------

    def event_generator():

        try:

            for event_data in query_documents_stream(
                question=request.question,
                db=db,
                user_id=user.id,
                conversation_id=conversation_id
            ):

                event_name = event_data.get(
                    "event",
                    "message"
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
                user.id
            )

            error_data = {
                "event": "error",
                "success": False,
                "error_code": "INTERNAL_SERVER_ERROR",
                "conversation_id": conversation_id,
                "message_id": None,
                "delta": None,
                "text_content": "Internal server error"
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(error_data)}\n\n"
            )

    # --------------------------------------------------------
    # SSE RESPONSE
    # --------------------------------------------------------

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# LIST CONVERSATIONS
# ============================================================

@router.get("")
def list_conversations(
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    user = _get_current_user(
        db=db,
        email=email
    )

    conversations = get_conversations_by_user(
        db=db,
        user_id=user.id
    )

    data = [
        {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
            ),
            "updated_at": (
                conversation.updated_at.isoformat()
            )
        }
        for conversation in conversations
    ]

    return success_response(
        message="Conversations fetched successfully",
        data=data,
        status_code=status.HTTP_200_OK
    )


# ============================================================
# GET CONVERSATION DETAIL
# ============================================================

@router.get("/{conversation_id}")
def get_conversation_detail(
    conversation_id: str,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    user = _get_current_user(
        db=db,
        email=email
    )

    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user.id
    )

    if not conversation:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    messages = get_all_messages(
        db=db,
        conversation_id=conversation_id
    )

    return success_response(
        message="Conversation fetched successfully",
        data={
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
            ),
            "updated_at": (
                conversation.updated_at.isoformat()
            ),
            "messages": [
                {
                    "message_id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "created_at": (
                        message.created_at.isoformat()
                    )
                }
                for message in messages
            ]
        },
        status_code=status.HTTP_200_OK
    )


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

@router.patch("/{conversation_id}")
def update_title(
    conversation_id: str,
    request: ConversationTitleUpdate,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    user = _get_current_user(
        db=db,
        email=email
    )

    conversation = update_conversation_title(
        db=db,
        conversation_id=conversation_id,
        user_id=user.id,
        title=request.title
    )

    if not conversation:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return success_response(
        message="Title updated successfully",
        data={
            "conversation_id": conversation.id,
            "title": conversation.title
        },
        status_code=status.HTTP_200_OK
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================

@router.delete("/{conversation_id}")
def delete_conversation_endpoint(
    conversation_id: str,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    user = _get_current_user(
        db=db,
        email=email
    )

    deleted = delete_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user.id
    )

    if not deleted:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return success_response(
        message="Conversation deleted successfully",
        data=None,
        status_code=status.HTTP_200_OK
    )