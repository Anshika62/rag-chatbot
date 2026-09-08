from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, LLMStream

# --------------------------------------------------------------------
# CHANGED: integrate at the same layer the existing REST API uses.
#
# app/api/conversation.py (send_message) does NOT call
# generate_answer_stream() directly - it calls
# query_documents_stream() in app/service/rag_service.py, which wraps
# generate_answer_stream() and additionally:
#   - loads/persists chat_history (get_last_10_messages)
#   - loads/persists conversation latitude/longitude
#   - creates the user message row and the assistant message row
#   - generates follow-up suggestions
#   - emits a richer event schema (event: start/thinking/delta/
#     location_request/map_location/suggestions/done/error) instead
#     of generate_answer_stream's raw {"type": ..., "content": ...}
#
# Calling generate_answer_stream() directly (the previous version of
# this file) skipped all of the above - no conversation memory across
# voice turns, and nothing ever got saved to the DB. Calling
# query_documents_stream() instead reuses 100% of that existing logic
# with zero duplication, exactly as required.
# --------------------------------------------------------------------
from app.service.rag_service import query_documents_stream

logger = logging.getLogger(__name__)

# Sentinel used to signal "generator finished" across the thread boundary.
_DONE = object()


class LiveKitLLM(llm.LLM):
    """Adapter between LiveKit and the existing RAG conversation flow
    (app.service.rag_service.query_documents_stream)."""

    def __init__(
        self,
        *,
        db,
        user_id: str,
        conversation_id: str,
    ):
        super().__init__()

        self.db = db
        self.user_id = user_id
        self.conversation_id = conversation_id

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: Any = None,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Any = None,
    ) -> LLMStream:

        return LiveKitLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            db=self.db,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )


class LiveKitLLMStream(LLMStream):

    def __init__(
        self,
        llm_instance: LiveKitLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: Any,
        db,
        user_id: str,
        conversation_id: str,
    ):
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

        self.db = db
        self.user_id = user_id
        self.conversation_id = conversation_id

    def _run_query_documents_stream_in_thread(
        self,
        question: str,
        out_queue: "queue.Queue",
    ) -> None:
        """
        Runs on a worker thread.

        query_documents_stream() (and everything it calls, including
        generate_answer_stream()) is a PLAIN SYNCHRONOUS generator that
        does blocking network/DB I/O. _run() below is a coroutine on
        LiveKit's asyncio event loop - iterating a blocking generator
        there directly would freeze that loop (audio pump, room
        heartbeats, etc.) for the entire RAG turn. Running it on a
        thread and handing events back through a queue avoids that
        without touching any RAG code.
        """

        try:

            for event_data in query_documents_stream(
                question=question,
                db=self.db,
                user_id=self.user_id,
                conversation_id=self.conversation_id,
            ):

                out_queue.put(event_data)

        except Exception as exc:  # noqa: BLE001

            out_queue.put(exc)

        finally:

            out_queue.put(_DONE)

    async def _run(self) -> None:

        try:
            # --------------------------------------------------
            # Get latest user message
            # --------------------------------------------------

            messages = self._chat_ctx.messages()

            question = ""

            for message in reversed(messages):

                if message.role == "user":
                    question = message.text_content
                    break

            if not question:
                return

            # --------------------------------------------------
            # Existing RAG flow, off the event loop (see docstring
            # above), bridged back via a thread-safe queue.
            # --------------------------------------------------

            loop = asyncio.get_event_loop()
            event_queue: "queue.Queue" = queue.Queue()

            worker = threading.Thread(
                target=self._run_query_documents_stream_in_thread,
                args=(question, event_queue),
                daemon=True,
            )
            worker.start()

            while True:

                event_data = await loop.run_in_executor(
                    None, event_queue.get
                )

                if event_data is _DONE:
                    break

                if isinstance(event_data, Exception):
                    raise event_data

                if not event_data:
                    continue

                # ----------------------------------------------
                # query_documents_stream() uses the key "event"
                # (not generate_answer_stream's "type"), and the
                # streamed text is under "delta" (not "content").
                # ----------------------------------------------

                event_type = event_data.get("event")

                # --------------------------------------------------
                # Answer delta -> forward to LiveKit as a chat chunk.
                # --------------------------------------------------

                if event_type == "delta":

                    text = event_data.get("delta", "")

                    if not text:
                        continue

                    await self._event_ch.send(
                        ChatChunk(
                            id="livekit-rag",
                            delta=ChoiceDelta(
                                role="assistant",
                                content=text,
                            ),
                        )
                    )

                # --------------------------------------------------
                # Everything else (start / thinking / location_request /
                # map_location / suggestions / done) has no voice-channel
                # representation yet. Skip safely without breaking the
                # RAG flow - these events still reach the normal REST/SSE
                # clients unchanged, this adapter just doesn't surface
                # them over voice.
                # --------------------------------------------------

                elif event_type == "error":

                    logger.warning(
                        "LiveKit LLM: RAG stream returned an error "
                        "event: %s",
                        event_data.get("text_content"),
                    )

                elif event_type in (
                    "start",
                    "thinking",
                    "location_request",
                    "map_location",
                    "suggestions",
                    "done",
                ):
                    continue

        except Exception as e:

            logger.exception("LiveKit LLM error: %s", e)

        finally:

            self._event_ch.close()