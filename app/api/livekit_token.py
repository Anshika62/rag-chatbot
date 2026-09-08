import json
import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from livekit import api
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependency import get_current_user
from app.core.response import success_response
from app.livekit_agent.workers import AGENT_NAME
from app.repository.conversation_repo import (
    create_conversation,
    get_conversation,
)


class LiveKitTokenRequest(BaseModel):
    """
    conversation_id is optional: omit it to start a new conversation
    (mirrors send_message()'s is_new_conv behavior), or pass an
    existing one to continue it over voice.
    """

    conversation_id: str | None = None


router = APIRouter(
    prefix="/livekit",
    tags=["LiveKit"],
)

logger = logging.getLogger(__name__)


LIVEKIT_API_KEY = (os.getenv("LIVEKIT_API_KEY") or "").strip()
LIVEKIT_API_SECRET = (os.getenv("LIVEKIT_API_SECRET") or "").strip()


@router.post("/token")
def create_livekit_token(
    request: LiveKitTokenRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Mint a LiveKit join token for a voice session, tied to a
    conversation.

    - If request.conversation_id is provided, it is validated the
      same way send_message() validates it (must exist and belong
      to this user).
    - If not provided, a new conversation is created the same way
      send_message() creates one for is_new_conv - no new
      conversation-creation logic is introduced here.

    The resulting conversation_id + user_id are embedded in the
    room's metadata as JSON. app/livekit_agent/entrypoint.py reads
    that same metadata to construct LiveKitLLM(db, user_id,
    conversation_id) - this is the single source of truth for both
    sides, so nothing new is invented beyond "read what this
    endpoint writes."
    """

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not configured",
        )

    conversation_id = request.conversation_id

    if conversation_id:

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

    else:

        conversation = create_conversation(
            db=db,
            user_id=user.id,
            title="Voice conversation",
        )

        conversation_id = str(conversation.id)

    room_name = f"voice-{conversation_id}"
    participant_identity = f"user-{user.id}"

    room_metadata = json.dumps(
        {
            "user_id": str(user.id),
            "conversation_id": conversation_id,
        }
    )

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .with_metadata(room_metadata)
        # --------------------------------------------------------
        # entrypoint.py registers the agent with an agent_name
        # (@server.rtc_session(agent_name=AGENT_NAME)), which uses
        # LiveKit's EXPLICIT dispatch model - the agent does NOT
        # auto-join every room. This tells LiveKit Cloud to dispatch
        # that specific named agent into this room. Without this,
        # a participant can join the room and nothing will ever
        # respond, independent of any RAG/adapter code.
        # --------------------------------------------------------
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata=room_metadata,
                    )
                ]
            )
        )
    )

    logger.info(
        "LIVEKIT TOKEN ISSUED: user_id=%s conversation_id=%s room=%s",
        user.id,
        conversation_id,
        room_name,
    )

    return success_response(
        message="LiveKit token created successfully",
        data={
            "token": token.to_jwt(),
            "room_name": room_name,
            "conversation_id": conversation_id,
            "livekit_url": os.getenv("LIVEKIT_URL"),
        },
        status_code=status.HTTP_200_OK,
    )