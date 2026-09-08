from __future__ import annotations

from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, LLMStream

from app.service.external.llm_service import generate_answer


class LiveKitLLM(llm.LLM):
    """Basic adapter between LiveKit and the existing LLM service."""

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
        )


class LiveKitLLMStream(LLMStream):

    async def _run(self) -> None:
        try:
            # Get the latest user message
            messages = self._chat_ctx.messages()

            question = ""

            for message in reversed(messages):
                if message.role == "user":
                    question = message.text_content
                    break

            if not question:
                return

            # Call existing LLM
            for piece in generate_answer(question=question):
                if not piece:
                    continue

                # generate_answer may return a string
                # or a dictionary depending on the existing service.
                if isinstance(piece, str):
                    text = piece

                elif isinstance(piece, dict):
                    text = (
                        piece.get("answer")
                        or piece.get("content")
                        or piece.get("text")
                        or ""
                    )

                else:
                    text = str(piece)

                if not text:
                    continue

                await self._event_ch.send(
                    ChatChunk(
                        id="livekit-basic",
                        delta=ChoiceDelta(
                            role="assistant",
                            content=text,
                        ),
                    )
                )

        except Exception as e:
            print(f"LiveKit LLM error: {e}")

        finally:
            self._event_ch.close()