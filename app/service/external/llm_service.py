import json
import logging
import os
from typing import Generator, Optional

from fastapi import HTTPException, status
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.service.tools.conversation_tool import (
    create_conversation_tools,
)


logger = logging.getLogger(__name__)


# ============================================================
# LLM CONFIGURATION
# ============================================================
#
# Both models are configurable via environment variables instead
# of being hard-coded, per the "environment variables for
# model/API configuration" requirement. Defaults preserve the
# existing behavior for the main LLM.
#
# MAIN LLM (llm):
#   Handles normal conversation + decides which tool(s) to call.
#   Used for everything by default.
#
# REASONING LLM (reasoning_llm):
#   A separate, reasoning-capable model used ONLY for the final
#   answer-synthesis step, and only when the tool results are
#   non-trivial (multiple tools, multiple retrieved chunks, or
#   text+image combined — see _should_use_reasoning below).
#   Simple chat ("Hello", "What is Python?") and simple single-
#   tool calls (weather, datetime, a single short KB hit) never
#   reach this model, so they stay fast and cheap.
#
#   The reasoning model is qwen/qwen3.6-27b, currently Groq's
#   highest-intelligence-ranked reasoning model. Unlike
#   openai/gpt-oss-120b (which rejects reasoning_format entirely
#   and only exposes reasoning via a separate include_reasoning
#   flag), qwen3.6-27b officially supports reasoning_format="raw",
#   which makes Groq inline the chain-of-thought as
#   <think>...</think> at the start of the streamed content —
#   exactly what _stream_with_thinking_split() below is built to
#   split into "thinking" vs "answer" pieces. That function also
#   still checks additional_kwargs["reasoning_content"] as a second
#   mechanism, so switching REASONING_MODEL to a gpt-oss model
#   later would keep working without further changes. Every piece
#   — thinking or answer — is yielded as {"type": "thinking"/
#   "answer", "content": ...} so the frontend can render a live
#   "thinking..." trace as it happens, while still saving only the
#   "answer" portion as the final message content.
# ============================================================


LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
)

REASONING_MODEL_NAME = os.getenv(
    "REASONING_MODEL",
    "qwen/qwen3.6-27b",
)


llm = ChatGroq(
    model=LLM_MODEL_NAME,
    temperature=0.2,
)


reasoning_llm = ChatGroq(
    model=REASONING_MODEL_NAME,
    temperature=0.3,
    reasoning_format="raw",
)


THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"


# ============================================================
# SYSTEM PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to:
1. Conversation history
2. Conversation history tool
3. Uploaded-document knowledge-base search tool
4. Document image analysis tool (analyze_document_image) — for
   analyzing a SPECIFIC image already extracted from an uploaded
   document/PDF, identified by document_id
5. Current date and time tool
6. Weather tool
7. Image analysis tool (analyze_image) — only present when the
   user has attached an image directly to their CURRENT message

Rules:

- Answer normal conversational questions directly.

- Use get_conversation_history when the provided history is
  insufficient.

- Use search_knowledge_base whenever the answer may be present
  in uploaded documents or the knowledge base.

- If the user asks about an uploaded document, PDF, file, policy,
  manual, FAQ, guideline, documentation, or indexed content,
  search the knowledge base before answering.

- If an uploaded document is available for the current
  conversation, ALWAYS use search_knowledge_base first for
  factual questions that could reasonably be answered from
  that document.

- The user does NOT need to mention the document, PDF, file,
  or say "in this uploaded document".

- For example, if an uploaded document is available and the
  user asks "Who is Anshika?", "What is Anshika's role?", or
  "When did Anshika join?", search_knowledge_base before
  answering.

- When an uploaded document is available, do not answer a
  factual question from general knowledge if the uploaded
  document could contain the answer.

- When search_knowledge_base returns relevant document content,
  use that retrieved content as the source of truth.

- Do not invent information from uploaded documents.

- If an image was attached directly to the current message
  (the analyze_image tool is available), use analyze_image to
  answer questions about THAT image.

- Prefer analyze_image over search_knowledge_base for a
  just-attached image.

- Only use search_knowledge_base for images that were uploaded
  previously as part of the document knowledge base.

- When search_knowledge_base returns an image result, use its
  caption/text to answer the question and treat the returned
  image metadata as the related image.

- Do not confuse an image document_id with its parent PDF
  document_id.

- If the user asks about an image inside a PDF, use the image
  result whose parent_document_id matches the PDF.

- When the user asks you to explain, interpret, or describe what
  an already-uploaded image/diagram/chart/figure actually shows
  (not just "is there an image"), first use search_knowledge_base
  (content_type="image") to find the relevant image and its
  document_id, then call analyze_document_image with that
  document_id and the user's specific question. Do not answer
  such questions using only the cached caption/OCR text if
  analyze_document_image is available — the actual image should
  be analyzed for the user's specific question.

- Do not guess a document_id for analyze_document_image. Only use
  a document_id that was actually returned by search_knowledge_base
  in this conversation.

- If the knowledge-base search finds no relevant information,
  say that the information was not found in the uploaded
  knowledge base.

- Use conversation history for follow-up questions.

- Keep answers clear and concise.

- Use get_current_datetime when the user asks for the current
  date or time.

- Never guess the current date or time. Always use
  get_current_datetime for current date/time questions.

- Use get_weather when the user asks about current weather,
  temperature, rainfall, humidity, wind, or weather conditions
  for a location.

- Never invent current weather information. Always use
  get_weather for current weather questions.
"""


# ============================================================
# BUILD LLM MESSAGES
# ============================================================


def _build_messages(
    question: str,
    chat_history: Optional[list[dict]] = None,
    document_available: bool = False,
):

    chat_history = chat_history or []

    recent_history = chat_history[-10:]

    history_text = "\n".join(
        f"{message.get('role', '')}: "
        f"{message.get('content', '')}"
        for message in recent_history
    )

    if not history_text:

        history_text = (
            "No previous conversation history."
        )

    document_context = (
        "YES. This turn is scoped to one specific uploaded "
        "document. If the current question could be answered "
        "from it, use search_knowledge_base before answering."
        if document_available
        else
        "Documents MAY be available in the knowledge base for "
        "this user — either global documents (uploaded without "
        "being tied to a conversation) or documents uploaded "
        "within this conversation. No single document_id is "
        "pre-selected for this turn, but search_knowledge_base "
        "still searches across all documents accessible to this "
        "user/conversation. If the current question could "
        "reasonably be answered from an uploaded document, use "
        "search_knowledge_base before answering from general "
        "knowledge — do not assume no document exists just "
        "because none is pre-selected."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """Conversation History:

{history}

Uploaded Document Available:

{document_context}

Current User Question:

{question}""",
            ),
        ]
    )

    return prompt.format_messages(
        history=history_text,
        document_context=document_context,
        question=question,
    )


# ============================================================
# CREATE TOOLS
# ============================================================


def _create_tools(
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
):

    if (
        db is None
        or user_id is None
        or conversation_id is None
    ):
        return []

    return create_conversation_tools(
        db=db,
        user_id=str(user_id),
        conversation_id=str(conversation_id),
        document_id=(
            str(document_id)
            if document_id
            else None
        ),
        image_paths=image_paths,
    )


# ============================================================
# BIND TOOLS
# ============================================================


def _bind_tools(tools: list):

    return (
        llm.bind_tools(tools)
        if tools
        else llm
    )


# ============================================================
# GET TOOL
# ============================================================


def _get_tool(
    tools: list,
    tool_name: str,
):

    return next(
        (
            current_tool
            for current_tool in tools
            if current_tool.name == tool_name
        ),
        None,
    )


# ============================================================
# EXECUTE TOOL CALLS
# ============================================================


def _execute_tool_calls(
    tools: list,
    tool_calls: list,
    conversation_id: Optional[str],
    log_prefix: str = "",
):

    tool_messages = []

    collected_images = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call.get(
            "args",
            {},
        )

        selected_tool = _get_tool(
            tools=tools,
            tool_name=tool_name,
        )

        if selected_tool is None:

            raise RuntimeError(
                f"Requested tool not found: {tool_name}"
            )

        logger.info(
            "%sTOOL EXECUTING: tool=%s args=%s "
            "conversation_id=%s",
            log_prefix,
            tool_name,
            tool_args,
            conversation_id,
        )

        try:

            tool_result = selected_tool.invoke(
                tool_args
            )

        except Exception:

            logger.exception(
                "%sTOOL FAILED: tool=%s args=%s "
                "conversation_id=%s",
                log_prefix,
                tool_name,
                tool_args,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"The '{tool_name}' tool failed and "
                        "is temporarily unavailable. Let the "
                        "user know and answer with whatever "
                        "other information is available."
                    ),
                }
            )

            continue

        logger.info(
            "%sTOOL RESULT: tool=%s conversation_id=%s",
            log_prefix,
            tool_name,
            conversation_id,
        )

        # ====================================================
        # COLLECT IMAGE REFERENCES
        # ====================================================

        if (
            tool_name == "search_knowledge_base"
            and isinstance(tool_result, list)
        ):

            for item in tool_result:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                content_type = item.get(
                    "content_type"
                )

                if (
                    content_type
                    and content_type.startswith("image/")
                    and item.get("document_id")
                ):

                    image_document_id = str(
                        item.get("document_id")
                    )

                    collected_images.append(
                        {
                            "document_id": item.get(
                                "document_id"
                            ),
                            "parent_document_id": (
                                item.get(
                                    "parent_document_id"
                                )
                                or item.get(
                                    "document_id"
                                )
                            ),
                            "filename": item.get(
                                "filename"
                            ),
                            "url": (
                                f"/documents/"
                                f"{image_document_id}/file"
                            ),
                        }
                    )

        elif (
            tool_name == "analyze_document_image"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("document_id")
        ):

            image_document_id = str(
                tool_result.get("document_id")
            )

            collected_images.append(
                {
                    "document_id": tool_result.get(
                        "document_id"
                    ),
                    "parent_document_id": (
                        tool_result.get(
                            "parent_document_id"
                        )
                        or tool_result.get(
                            "document_id"
                        )
                    ),
                    "filename": tool_result.get(
                        "filename"
                    ),
                    "url": (
                        f"/documents/"
                        f"{image_document_id}/file"
                    ),
                }
            )

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result),
            }
        )

    return (
        tool_messages,
        collected_images,
    )


# ============================================================
# REASONING DECISION (Section 16)
#
# Keep simple questions simple: "Hello", "What is Python?", a
# single short knowledge-base hit, or a single weather/datetime
# call never reach the reasoning model. Reasoning is only used
# when the tool results actually need to be compared/combined —
# multiple tools used together, an image analyzed alongside other
# content, or a substantial amount of retrieved text to reason
# over.
# ============================================================


REASONING_MIN_TOOL_CONTENT_CHARS = int(
    os.getenv(
        "REASONING_MIN_TOOL_CONTENT_CHARS",
        "600",
    )
)


def _should_use_reasoning(
    tool_calls: list,
    collected_images: list,
    extra_messages: list,
) -> bool:

    if not tool_calls:
        return False

    tool_names = {
        tool_call["name"]
        for tool_call in tool_calls
    }

    # Multiple different tools used together -> results need to
    # be combined/synthesized.
    if len(tool_names) > 1:
        return True

    # An image was analyzed alongside other tool output -> combining
    # vision output with text is exactly the case Section 16 calls
    # out for reasoning.
    if collected_images:
        return True

    # A single search_knowledge_base call that returned a
    # substantial amount of text (multiple/long chunks) benefits
    # from a reasoning pass to compare and synthesize across them.
    if "search_knowledge_base" in tool_names:

        combined_length = sum(
            len(str(message.get("content", "")))
            for message in extra_messages
            if (
                isinstance(message, dict)
                and message.get("role") == "tool"
            )
        )

        if combined_length > REASONING_MIN_TOOL_CONTENT_CHARS:
            return True

    return False


# ============================================================
# THINKING/ANSWER STREAM SPLITTER
#
# The reasoning model emits a single continuous token stream that
# looks like:
#
#     <think> ...chain of thought... </think> ...final answer...
#
# This splits that stream into separate "thinking" and "answer"
# events as it arrives, so the caller can show live "thinking..."
# output (like other reasoning-model chat UIs) without it ending
# up as part of the saved/displayed final answer. A small tail of
# text is always held back while scanning for a tag, in case a
# tag like "<think>" is split across two streamed chunks.
#
# Two different mechanisms are checked on every chunk, since
# different reasoning models expose their chain-of-thought
# differently:
#
#   1. additional_kwargs["reasoning_content"] (or ["reasoning"])
#      — used by openai/gpt-oss-20b / openai/gpt-oss-120b on Groq,
#      which return reasoning as a separate field per chunk
#      instead of inlining it in .content.
#
#   2. <think>...</think> tags inline inside .content — used by
#      DeepSeek-R1-distill and some other reasoning models.
#
# A model only ever uses one of the two, so only one branch will
# ever produce output for a given REASONING_MODEL — the other is
# simply a silent no-op.
# ============================================================


def _stream_with_thinking_split(
    model_stream,
):

    state = "answer"
    pending = ""

    for chunk in model_stream:

        # ---- mechanism 1: separate reasoning field per chunk ----

        extra_kwargs = getattr(
            chunk,
            "additional_kwargs",
            None,
        ) or {}

        reasoning_piece = (
            extra_kwargs.get("reasoning_content")
            or extra_kwargs.get("reasoning")
        )

        if reasoning_piece:

            yield {
                "type": "thinking",
                "content": reasoning_piece,
            }

        # ---- mechanism 2: inline <think> tags inside .content ----

        content = getattr(
            chunk,
            "content",
            "",
        ) or ""

        if not content:
            continue

        pending += content

        while pending:

            if state == "answer":

                tag_index = pending.find(
                    THINK_START_TAG
                )

                if tag_index == -1:

                    safe_length = max(
                        0,
                        len(pending)
                        - len(THINK_START_TAG),
                    )

                    if safe_length:

                        yield {
                            "type": "answer",
                            "content": pending[
                                :safe_length
                            ],
                        }

                        pending = pending[
                            safe_length:
                        ]

                    break

                if tag_index:

                    yield {
                        "type": "answer",
                        "content": pending[
                            :tag_index
                        ],
                    }

                pending = pending[
                    tag_index
                    + len(THINK_START_TAG):
                ]

                state = "thinking"

            else:

                tag_index = pending.find(
                    THINK_END_TAG
                )

                if tag_index == -1:

                    safe_length = max(
                        0,
                        len(pending)
                        - len(THINK_END_TAG),
                    )

                    if safe_length:

                        yield {
                            "type": "thinking",
                            "content": pending[
                                :safe_length
                            ],
                        }

                        pending = pending[
                            safe_length:
                        ]

                    break

                if tag_index:

                    yield {
                        "type": "thinking",
                        "content": pending[
                            :tag_index
                        ],
                    }

                pending = pending[
                    tag_index
                    + len(THINK_END_TAG):
                ]

                state = "answer"

    if pending:

        yield {
            "type": state,
            "content": pending,
        }


# ============================================================
# STRIP THINKING (non-streamed reasoning responses)
#
# Mirrors _stream_with_thinking_split above, but for a single
# already-complete response string (used by generate_answer,
# the non-streaming path). Removes any <think>...</think> block
# so the model's private chain-of-thought is never returned as
# part of the final answer.
# ============================================================


def _strip_thinking(text: str) -> str:

    if not text:
        return text

    result = []
    remaining = text

    while True:

        start_index = remaining.find(THINK_START_TAG)

        if start_index == -1:
            result.append(remaining)
            break

        result.append(remaining[:start_index])

        end_index = remaining.find(
            THINK_END_TAG,
            start_index + len(THINK_START_TAG),
        )

        if end_index == -1:
            # Unclosed tag — drop everything from the tag onward
            # rather than risk leaking a partial chain-of-thought.
            break

        remaining = remaining[
            end_index + len(THINK_END_TAG):
        ]

    return "".join(result).strip()


# ============================================================
# REASONING CONTEXT BUILDER
#
# Builds the minimal set of messages the reasoning model needs:
# the original conversation/question messages plus only the tool
# results actually produced for this turn (retrieved chunks,
# vision analysis, weather/datetime results, etc). We never hand
# the reasoning model the raw database/conversation dump — only
# what _execute_tool_calls already gathered for this turn.
# ============================================================


def _build_reasoning_messages(
    base_messages: list,
    tool_messages: list,
):

    reasoning_instruction = (
        "human",
        "Using ONLY the information above (the question, "
        "conversation context, and tool results), reason "
        "carefully and produce one clear, well-synthesized "
        "final answer for the user.\n\n"
        "Important:\n"
        "- Some retrieved tool results may NOT be relevant to "
        "the user's actual question (retrieval is not perfect). "
        "Silently discard anything irrelevant — do not mention, "
        "summarize, or reference it in your answer.\n"
        "- Only use content that directly helps answer the "
        "question asked.\n"
        "- Write a natural, concise, conversational answer. Do "
        "not dump raw retrieved text, filenames, metadata, or "
        "internal tool details into the answer.\n"
        "- Do not mention that you are a separate reasoning "
        "step, that you used tools, or that some results were "
        "discarded.",
    )

    return (
        base_messages
        + tool_messages
        + [reasoning_instruction]
    )


# ============================================================
# GENERATE ANSWER
# ============================================================


def generate_answer(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    images_output: Optional[list] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
):

    """
    document_id:
        If provided, scopes the knowledge-base search tool
        to that single uploaded document.

    image_paths:
        Local file path(s) of images attached directly to
        THIS message.
    """

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(document_id),
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(tools)

        response = llm_with_tools.invoke(
            messages
        )

        if not response.tool_calls:

            return response.content

        logger.info(
            "LLM TOOL CALLS: tools=%s "
            "conversation_id=%s",
            [
                tool_call["name"]
                for tool_call in response.tool_calls
            ],
            conversation_id,
        )

        tool_messages = [
            response
        ]

        extra_messages, collected_images = (
            _execute_tool_calls(
                tools=tools,
                tool_calls=response.tool_calls,
                conversation_id=conversation_id,
            )
        )

        tool_messages.extend(
            extra_messages
        )

        if images_output is not None:

            images_output.extend(
                collected_images
            )

        # ====================================================
        # REASONING STAGE (final-answer synthesis)
        #
        # Only used when the tool results actually need to be
        # compared/combined (see _should_use_reasoning). Simple
        # single-tool turns keep using the Main LLM directly.
        # ====================================================

        use_reasoning = _should_use_reasoning(
            tool_calls=response.tool_calls,
            collected_images=collected_images,
            extra_messages=extra_messages,
        )

        logger.info(
            "REASONING %s: conversation_id=%s",
            "SELECTED" if use_reasoning else "SKIPPED",
            conversation_id,
        )

        if use_reasoning:

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                reasoning_messages = _build_reasoning_messages(
                    base_messages=messages,
                    tool_messages=tool_messages,
                )

                reasoning_response = reasoning_llm.invoke(
                    reasoning_messages
                )

                final_answer = _strip_thinking(
                    reasoning_response.content
                )

                if final_answer:

                    logger.info(
                        "REASONING COMPLETE: conversation_id=%s",
                        conversation_id,
                    )

                    return final_answer

                logger.warning(
                    "REASONING EMPTY RESULT, falling back to "
                    "main LLM: conversation_id=%s",
                    conversation_id,
                )

            except Exception:

                logger.exception(
                    "REASONING FAILED, falling back to main "
                    "LLM: conversation_id=%s",
                    conversation_id,
                )

        # ====================================================
        # MAIN LLM FINAL ANSWER (default path / reasoning
        # fallback)
        # ====================================================

        final_response = llm_with_tools.invoke(
            messages + tool_messages
        )

        return final_response.content

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "LLM RESPONSE ERROR: conversation_id=%s "
            "error=%s",
            conversation_id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate response from LLM",
        ) from exc


# ============================================================
# PARSE STREAMING TOOL CALLS
# ============================================================


def _parse_tool_calls(
    tool_call_chunks: list,
):

    if not tool_call_chunks:

        return []

    tool_calls = {}

    for chunk in tool_call_chunks:

        index = chunk.get(
            "index",
            0,
        )

        if index not in tool_calls:

            tool_calls[index] = {
                "id": "",
                "name": "",
                "args": "",
            }

        tool_calls[index]["id"] += (
            chunk.get("id") or ""
        )

        tool_calls[index]["name"] += (
            chunk.get("name") or ""
        )

        tool_calls[index]["args"] += (
            chunk.get("args") or ""
        )

    parsed_calls = []

    for tool_call in tool_calls.values():

        args_text = tool_call["args"].strip()

        if args_text:

            try:

                args = json.loads(
                    args_text
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    "Unable to parse tool arguments"
                ) from exc

        else:

            args = {}

        parsed_calls.append(
            {
                "id": tool_call["id"],
                "name": tool_call["name"],
                "args": args,
            }
        )

    return parsed_calls


# ============================================================
# GENERATE STREAMING ANSWER
# ============================================================


def generate_answer_stream(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    images_output: Optional[list] = None,
    document_id: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
) -> Generator[dict, None, None]:

    """
    document_id:
        If provided, scopes the knowledge-base search tool
        to that single uploaded document.

    image_paths:
        Local file path(s) of image(s) attached directly to
        THIS message.

    Yields:
        dicts of the form {"type": "thinking" | "answer",
        "content": str}. "thinking" pieces are the reasoning
        model's live chain-of-thought (only ever produced during
        the REASONING STAGE below) and should be shown to the
        user as a transient "thinking..." trace, never saved as
        the final message content. "answer" pieces are the real
        response and should be both streamed and saved.
    """

    try:

        messages = _build_messages(
            question=question,
            chat_history=chat_history,
            document_available=bool(document_id),
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
            image_paths=image_paths,
        )

        llm_with_tools = _bind_tools(tools)

        logger.info(
            "LLM STREAM START: conversation_id=%s "
            "tools=%s",
            conversation_id,
            [
                tool.name
                for tool in tools
            ],
        )

        streamed_chunks = []

        tool_call_chunks = []

        for chunk in llm_with_tools.stream(
            messages
        ):

            streamed_chunks.append(
                chunk
            )

            current_tool_chunks = getattr(
                chunk,
                "tool_call_chunks",
                None,
            )

            if current_tool_chunks:

                tool_call_chunks.extend(
                    current_tool_chunks
                )

        tool_calls = _parse_tool_calls(
            tool_call_chunks
        )

        # ====================================================
        # NORMAL STREAMING RESPONSE
        # ====================================================

        if not tool_calls:

            for chunk in streamed_chunks:

                if chunk.content:

                    yield {
                        "type": "answer",
                        "content": chunk.content,
                    }

            return

        # ====================================================
        # TOOL CALLS DETECTED
        # ====================================================

        logger.info(
            "STREAM TOOL CALLS: tools=%s "
            "conversation_id=%s",
            [
                tool_call["name"]
                for tool_call in tool_calls
            ],
            conversation_id,
        )

        # ====================================================
        # RECONSTRUCT AI TOOL-CALL RESPONSE
        # ====================================================

        full_ai_response = None

        for chunk in streamed_chunks:

            if full_ai_response is None:

                full_ai_response = chunk

            else:

                full_ai_response = (
                    full_ai_response + chunk
                )

        if full_ai_response is None:

            raise RuntimeError(
                "Unable to reconstruct tool-call response"
            )

        tool_messages = [
            full_ai_response
        ]

        # ====================================================
        # EXECUTE TOOLS
        # ====================================================

        extra_messages, collected_images = (
            _execute_tool_calls(
                tools=tools,
                tool_calls=tool_calls,
                conversation_id=conversation_id,
                log_prefix="STREAM ",
            )
        )

        tool_messages.extend(
            extra_messages
        )

        if images_output is not None:

            images_output.extend(
                collected_images
            )

        # ====================================================
        # GENERATE FINAL RESPONSE
        # ====================================================

        final_messages = (
            messages + tool_messages
        )

        # ====================================================
        # REASONING STAGE (final-answer synthesis)
        #
        # Only used when the tool results actually need to be
        # compared/combined (see _should_use_reasoning). Simple
        # single-tool turns keep streaming from the Main LLM
        # directly, unchanged from prior behavior.
        # ====================================================

        use_reasoning = _should_use_reasoning(
            tool_calls=tool_calls,
            collected_images=collected_images,
            extra_messages=extra_messages,
        )

        logger.info(
            "REASONING %s: conversation_id=%s",
            "SELECTED" if use_reasoning else "SKIPPED",
            conversation_id,
        )

        if use_reasoning:

            reasoning_yielded_any = False

            try:

                logger.info(
                    "REASONING START: model=%s "
                    "conversation_id=%s",
                    REASONING_MODEL_NAME,
                    conversation_id,
                )

                reasoning_messages = _build_reasoning_messages(
                    base_messages=messages,
                    tool_messages=tool_messages,
                )

                reasoning_answer_yielded = False

                for piece in _stream_with_thinking_split(
                    reasoning_llm.stream(reasoning_messages)
                ):

                    if not piece["content"]:
                        continue

                    if piece["type"] == "thinking":

                        # Live chain-of-thought — shown to the
                        # user as a transient "thinking..." trace.
                        # Never saved as the final answer.

                        yield {
                            "type": "thinking",
                            "content": piece["content"],
                        }

                    elif piece["type"] == "answer":

                        reasoning_yielded_any = True
                        reasoning_answer_yielded = True

                        yield {
                            "type": "answer",
                            "content": piece["content"],
                        }

                if reasoning_answer_yielded:

                    logger.info(
                        "REASONING COMPLETE: conversation_id=%s",
                        conversation_id,
                    )

                    return

                logger.warning(
                    "REASONING EMPTY RESULT, falling back to "
                    "main LLM: conversation_id=%s",
                    conversation_id,
                )

            except Exception:

                logger.exception(
                    "REASONING FAILED: conversation_id=%s",
                    conversation_id,
                )

                # If the reasoning model already streamed part of
                # an answer before failing, do NOT also stream the
                # Main LLM's answer — that would produce a garbled,
                # duplicated response. Only fall back to the Main
                # LLM when reasoning produced nothing at all.
                if reasoning_yielded_any:
                    return

        # ====================================================
        # MAIN LLM FINAL ANSWER (default path / reasoning
        # fallback)
        # ====================================================

        for chunk in llm_with_tools.stream(
            final_messages
        ):

            if chunk.content:

                yield {
                    "type": "answer",
                    "content": chunk.content,
                }

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "LLM STREAM RESPONSE ERROR: "
            "conversation_id=%s error=%s",
            conversation_id,
            str(exc),
        )

        raise RuntimeError(
            f"Unable to generate streaming response: {exc}"
        ) from exc


# ============================================================
# GENERATE CONVERSATION TITLE
# ============================================================


def generate_title(
    question: str,
) -> str:

    try:

        title_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Generate a short, clear title
(3-6 words) for a conversation
that starts with the given user message.

Rules:
- Do not use quotes.
- Do not add punctuation at the end.
- Return only the title.
""",
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )

        messages = title_prompt.format_messages(
            question=question
        )

        response = llm.invoke(
            messages
        )

        title = response.content.strip()

        return (
            title[:100]
            if title
            else question[:50]
        )

    except Exception as exc:

        logger.exception(
            "TITLE GENERATION ERROR: error=%s",
            str(exc),
        )

        return question[:50]