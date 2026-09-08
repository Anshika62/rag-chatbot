import json
import logging
import os

from dotenv import load_dotenv
from sqlalchemy.exc import DBAPIError

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.plugins import deepgram, elevenlabs

from app.core.database import SessionLocal
from app.livekit_agent.workers import AGENT_NAME
from app.livekit_agent.livekit_llm import LiveKitLLM


load_dotenv()

logger = logging.getLogger(__name__)

server = AgentServer()


# ============================================================
# RESOLVE user_id / conversation_id FOR THIS ROOM
# ============================================================
#
# ASSUMPTION - please verify/adjust against whatever code issues
# LiveKit access tokens for this app:
#
# query_documents_stream() requires an EXISTING conversation
# (it calls get_conversation(db, conversation_id, user_id) and
# yields a "not_found" error event if none exists) - it does not
# create one. So a conversation must already exist (created via
# the normal POST /conversation flow) before the room starts, and
# whatever mints the LiveKit join token must attach that
# conversation_id and the user_id to the room, e.g.:
#
#   token.with_metadata(json.dumps({
#       "user_id": "...",
#       "conversation_id": "...",
#   }))
#
# This function only READS that metadata - it does not introduce
# any new token/auth architecture. If your token endpoint puts
# these values somewhere else (participant attributes, room name
# encoding, etc.) swap this function's body accordingly; nothing
# else in this file needs to change.
# ============================================================


def _resolve_user_and_conversation(ctx: JobContext) -> tuple[str | None, str | None]:

    raw_metadata = ctx.room.metadata

    if not raw_metadata:
        return None, None

    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, ValueError):
        logger.warning(
            "LiveKit room metadata is not valid JSON: room=%s",
            ctx.room.name,
        )
        return None, None

    return (
        metadata.get("user_id"),
        metadata.get("conversation_id"),
    )


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):

    print(f"Agent joined room: {ctx.room.name}")

    user_id, conversation_id = _resolve_user_and_conversation(ctx)

    if not user_id or not conversation_id:
        logger.error(
            "LiveKit session started without user_id/conversation_id "
            "in room metadata (room=%s) - the RAG-backed LLM cannot "
            "look up a conversation. See _resolve_user_and_conversation().",
            ctx.room.name,
        )

    # --------------------------------------------------------
    # Same session lifecycle as app.core.database.get_db(): open
    # one session for this job, close it when the job ends, and
    # swallow a DBAPIError on close the same way get_db() does
    # (idle-timeout dropping the connection server-side).
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        session = AgentSession(
            stt=deepgram.STT(),

            llm=LiveKitLLM(
                db=db,
                user_id=user_id,
                conversation_id=conversation_id,
            ),

            tts=elevenlabs.TTS(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
            ),
        )

        await session.start(
            room=ctx.room,
            agent=Agent(
                instructions="You are a helpful voice AI assistant."
            ),
        )

    finally:

        try:
            db.close()

        except DBAPIError:

            logger.warning(
                "LiveKit DB session cleanup: underlying connection "
                "was already closed. Discarding it safely."
            )


if __name__ == "__main__":
    cli.run_app(server)