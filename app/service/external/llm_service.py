import json
import logging
import os
from typing import Generator, Optional, Any

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
# model/API configuration" requirement.
#
# MAIN LLM:
#   Handles normal conversation + decides which tool(s) to call.
#
# REASONING LLM:
#   Used ONLY for complex final answer synthesis.
#   Its private chain-of-thought is never exposed to the frontend.
#   Instead, a concise safe reasoning summary can be shown before
#   the final answer.
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

SAFE_REASONING_START = "[REASONING SUMMARY]"
SAFE_REASONING_END = "[/REASONING SUMMARY]"

SAFE_ANSWER_START = "[FINAL ANSWER]"
SAFE_ANSWER_END = "[/FINAL ANSWER]"

MAX_KB_SEARCH_ATTEMPTS = int(
    os.getenv("MAX_KB_SEARCH_ATTEMPTS", "4")
)
MAX_VISIBLE_THINKING_STEPS = 4


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


def _extract_kb_sources(tool_result: Any) -> list[dict]:
    """Extract real source metadata from search_knowledge_base results."""
    if not isinstance(tool_result, list):
        return []

    sources = []
    for item in tool_result:
        if not isinstance(item, dict):
            continue

        filename = item.get("filename")
        document_id = item.get("document_id")
        parent_document_id = item.get("parent_document_id")
        page_number = item.get("page_number")
        chunk_index = item.get("chunk_index")

        if not filename and not document_id:
            continue

        sources.append({
            "document_id": str(document_id) if document_id else None,
            "parent_document_id": (
                str(parent_document_id)
                if parent_document_id
                else None
            ),
            "filename": filename,
            "page_number": page_number,
            "chunk_index": chunk_index,
        })

    return sources


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    """Remove duplicate source entries while preserving order."""
    unique = []
    seen = set()

    for source in sources:
        key = (
            source.get("document_id"),
            source.get("parent_document_id"),
            source.get("filename"),
            source.get("page_number"),
            source.get("chunk_index"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(source)

    return unique


def _execute_tool_calls(
    tools: list,
    tool_calls: list,
    conversation_id: Optional[str],
    log_prefix: str = "",
):
    tool_messages = []
    collected_images = []
    collected_sources = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        selected_tool = _get_tool(
            tools=tools,
            tool_name=tool_name,
        )

        if selected_tool is None:
            raise RuntimeError(
                f"Requested tool not found: {tool_name}"
            )

        logger.info(
            "%sTOOL EXECUTING: tool=%s args=%s conversation_id=%s",
            log_prefix,
            tool_name,
            tool_args,
            conversation_id,
        )

        try:
            tool_result = selected_tool.invoke(tool_args)
        except Exception:
            logger.exception(
                "%sTOOL FAILED: tool=%s args=%s conversation_id=%s",
                log_prefix,
                tool_name,
                tool_args,
                conversation_id,
            )

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": (
                    f"The '{tool_name}' tool failed and is temporarily "
                    "unavailable. Let the user know and answer with "
                    "whatever other information is available."
                ),
            })
            continue

        logger.info(
            "%sTOOL RESULT: tool=%s conversation_id=%s",
            log_prefix,
            tool_name,
            conversation_id,
        )

        if (
            tool_name == "search_knowledge_base"
            and isinstance(tool_result, list)
        ):
            collected_sources.extend(
                _extract_kb_sources(tool_result)
            )

            for item in tool_result:
                if not isinstance(item, dict):
                    continue

                content_type = item.get("content_type")

                if (
                    content_type
                    and content_type.startswith("image/")
                    and item.get("document_id")
                ):
                    image_document_id = str(item.get("document_id"))

                    collected_images.append({
                        "document_id": item.get("document_id"),
                        "parent_document_id": (
                            item.get("parent_document_id")
                            or item.get("document_id")
                        ),
                        "filename": item.get("filename"),
                        "url": (
                            f"/documents/{image_document_id}/file"
                        ),
                    })

        elif (
            tool_name == "analyze_document_image"
            and isinstance(tool_result, dict)
            and tool_result.get("success")
            and tool_result.get("document_id")
        ):
            image_document_id = str(
                tool_result.get("document_id")
            )

            collected_images.append({
                "document_id": tool_result.get("document_id"),
                "parent_document_id": (
                    tool_result.get("parent_document_id")
                    or tool_result.get("document_id")
                ),
                "filename": tool_result.get("filename"),
                "url": (
                    f"/documents/{image_document_id}/file"
                ),
            })

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(tool_result),
        })

    return (
        tool_messages,
        collected_images,
        _deduplicate_sources(collected_sources),
    )


# ============================================================
# REASONING DECISION
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

    if len(tool_names) > 1:
        return True

    if collected_images:
        return True

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
# ============================================================


def _stream_with_thinking_split(
    model_stream,
):

    state = "answer"
    pending = ""

    for chunk in model_stream:

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

                        pending = pending[
                            safe_length:
                        ]

                    break

                if tag_index:

                    pending = pending[
                        tag_index:
                    ]

                pending = pending[
                    tag_index
                    + len(THINK_END_TAG):
                ]

                state = "answer"

    if pending and state == "answer":
        yield {
            "type": "answer",
            "content": pending,
        }


# ============================================================
# STRIP THINKING
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
            break

        remaining = remaining[
            end_index + len(THINK_END_TAG):
        ]

    return "".join(result).strip()


# ============================================================
# SAFE REASONING OUTPUT PARSER
# ============================================================


def _parse_safe_reasoning_output(
    text: str,
) -> tuple[str, str]:

    cleaned = _strip_thinking(
        text or ""
    ).strip()

    if not cleaned:
        return "", ""

    summary = ""
    answer = cleaned

    summary_start = cleaned.find(
        SAFE_REASONING_START
    )

    summary_end = cleaned.find(
        SAFE_REASONING_END
    )

    answer_start = cleaned.find(
        SAFE_ANSWER_START
    )

    answer_end = cleaned.find(
        SAFE_ANSWER_END
    )

    if (
        summary_start != -1
        and summary_end != -1
        and summary_end > summary_start
    ):
        summary = cleaned[
            summary_start
            + len(SAFE_REASONING_START):
            summary_end
        ].strip()

    if answer_start != -1:

        answer_content_start = (
            answer_start
            + len(SAFE_ANSWER_START)
        )

        if (
            answer_end != -1
            and answer_end > answer_content_start
        ):
            answer = cleaned[
                answer_content_start:
                answer_end
            ].strip()
        else:
            answer = cleaned[
                answer_content_start:
            ].strip()

    elif summary_end != -1:
        answer = cleaned[
            summary_end
            + len(SAFE_REASONING_END):
        ].strip()

    return summary, answer


# ============================================================
# REASONING CONTEXT BUILDER
# ============================================================


def _build_reasoning_messages(
    base_messages: list,
    tool_messages: list,
):

    reasoning_instruction = (
        "human",
        """
Using ONLY the information above (the question, conversation
context, and tool results), produce a final response in exactly
this structure:

[REASONING SUMMARY]
- Give a concise but useful user-facing explanation of the key
  evidence considered.
- Mention important information that was compared, combined,
  interpreted, or selected from the available tool results.
- Explain the important decision or conclusion that follows from
  that evidence.
- Keep this summary focused on useful reasoning, not hidden
  internal chain-of-thought.
- Do not reveal private chain-of-thought, internal deliberation,
  hidden tokens, or private model reasoning.
[/REASONING SUMMARY]

[FINAL ANSWER]
Give one clear, well-synthesized final answer for the user.
Use only the available information. Do not invent information.
Do not mention that you are a separate reasoning step.
[/FINAL ANSWER]
""",
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
    """Generate a non-streaming answer with up to four KB searches."""
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
        current_messages = list(messages)

        kb_search_count = 0
        all_tool_messages = []
        all_images = []
        all_sources = []
        first_tool_calls = []

        while True:
            response = llm_with_tools.invoke(current_messages)

            if not response.tool_calls:
                final_answer = response.content
                break

            if not first_tool_calls:
                first_tool_calls = list(response.tool_calls)

            current_messages.append(response)
            all_tool_messages.append(response)

            kb_calls = [
                call
                for call in response.tool_calls
                if call["name"] == "search_knowledge_base"
            ]

            remaining = max(
                0,
                MAX_KB_SEARCH_ATTEMPTS - kb_search_count,
            )

            executable_calls = []
            blocked_kb_calls = []
            allowed_kb_calls = 0

            for call in response.tool_calls:
                if call["name"] == "search_knowledge_base":
                    if allowed_kb_calls < remaining:
                        executable_calls.append(call)
                        allowed_kb_calls += 1
                    else:
                        blocked_kb_calls.append(call)
                else:
                    executable_calls.append(call)

            for call in blocked_kb_calls:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        "Maximum knowledge-base search attempts have "
                        "been reached. Do not search again. Use the "
                        "information already retrieved. If it is "
                        "insufficient, say that the information was "
                        "not found."
                    ),
                })

            if blocked_kb_calls and not executable_calls:
                final_response = llm.invoke(current_messages)
                final_answer = final_response.content
                break

            (
                extra_messages,
                collected_images,
                collected_sources,
            ) = _execute_tool_calls(
                tools=tools,
                tool_calls=executable_calls,
                conversation_id=conversation_id,
            )

            current_messages.extend(extra_messages)
            all_tool_messages.extend(extra_messages)
            all_images.extend(collected_images)
            all_sources.extend(collected_sources)
            kb_search_count += allowed_kb_calls

        if images_output is not None:
            images_output.extend(all_images)

        if all_tool_messages:
            use_reasoning = _should_use_reasoning(
                tool_calls=first_tool_calls,
                collected_images=all_images,
                extra_messages=all_tool_messages,
            )

            if use_reasoning:
                try:
                    reasoning_messages = _build_reasoning_messages(
                        base_messages=messages,
                        tool_messages=all_tool_messages,
                    )

                    reasoning_response = reasoning_llm.invoke(
                        reasoning_messages
                    )

                    reasoning_summary, reasoning_answer = (
                        _parse_safe_reasoning_output(
                            reasoning_response.content
                        )
                    )

                    if reasoning_answer:
                        final_answer = reasoning_answer

                except Exception:
                    logger.exception(
                        "REASONING FAILED, using main LLM answer: "
                        "conversation_id=%s",
                        conversation_id,
                    )

        logger.info(
            "KB SEARCH ATTEMPTS: count=%s conversation_id=%s",
            kb_search_count,
            conversation_id,
        )

        return _strip_thinking(final_answer)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "LLM RESPONSE ERROR: conversation_id=%s error=%s",
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
    Stream an answer while allowing the main LLM to perform up to four
    Knowledge Base searches when earlier results are insufficient.

    Generic activity messages are not emitted as reasoning.
    For complex reasoning, only a safe user-facing reasoning summary
    is emitted. Private chain-of-thought is never sent to the frontend.
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
        current_messages = list(messages)

        kb_search_count = 0
        all_tool_messages = []
        all_images = []
        all_sources = []
        first_tool_calls = []

        while True:
            streamed_chunks = []
            tool_call_chunks = []

            for chunk in llm_with_tools.stream(
                current_messages
            ):
                streamed_chunks.append(chunk)

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

            # ------------------------------------------------
            # NO TOOL CALL
            # ------------------------------------------------
            if not tool_calls:

                if not all_tool_messages:
                    for chunk in streamed_chunks:
                        if chunk.content:
                            yield {
                                "type": "answer",
                                "content": chunk.content,
                            }
                    return

                # The main LLM has enough information.
                # For complex cases, run the reasoning model.
                use_reasoning = _should_use_reasoning(
                    tool_calls=first_tool_calls,
                    collected_images=all_images,
                    extra_messages=all_tool_messages,
                )

                if use_reasoning:
                    try:
                        reasoning_messages = _build_reasoning_messages(
                            base_messages=messages,
                            tool_messages=all_tool_messages,
                        )

                        reasoning_response = reasoning_llm.invoke(
                            reasoning_messages
                        )

                        reasoning_summary, reasoning_answer = (
                            _parse_safe_reasoning_output(
                                reasoning_response.content
                            )
                        )

                        # Only the safe user-facing reasoning summary
                        # is sent as a thinking event.
                        if reasoning_summary:
                            yield {
                                "type": "thinking",
                                "content": reasoning_summary,
                            }

                        if reasoning_answer:
                            yield {
                                "type": "answer",
                                "content": reasoning_answer,
                            }

                            if images_output is not None:
                                images_output.extend(all_images)

                            if all_sources:
                                yield {
                                    "type": "sources",
                                    "sources": _deduplicate_sources(
                                        all_sources
                                    ),
                                }

                            return

                    except Exception:
                        logger.exception(
                            "REASONING FAILED: conversation_id=%s",
                            conversation_id,
                        )

                # Fall back to the answer produced by the main LLM.
                for chunk in streamed_chunks:
                    if chunk.content:
                        yield {
                            "type": "answer",
                            "content": chunk.content,
                        }

                if images_output is not None:
                    images_output.extend(all_images)

                if all_sources:
                    yield {
                        "type": "sources",
                        "sources": _deduplicate_sources(all_sources),
                    }

                return

            if not first_tool_calls:
                first_tool_calls = list(tool_calls)

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

            current_messages.append(full_ai_response)
            all_tool_messages.append(full_ai_response)

            kb_calls = [
                call
                for call in tool_calls
                if call["name"] == "search_knowledge_base"
            ]

            remaining = max(
                0,
                MAX_KB_SEARCH_ATTEMPTS - kb_search_count,
            )

            executable_calls = []
            blocked_kb_calls = []
            allowed_kb_calls = 0

            for call in tool_calls:

                if call["name"] == "search_knowledge_base":

                    if allowed_kb_calls < remaining:
                        executable_calls.append(call)
                        allowed_kb_calls += 1
                    else:
                        blocked_kb_calls.append(call)

                else:
                    executable_calls.append(call)

            # Add blocked-call results after the AI tool-call message.
            for call in blocked_kb_calls:

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        "Maximum knowledge-base search attempts have "
                        "been reached. Do not search again. Use the "
                        "information already retrieved. If it is "
                        "insufficient, say that the information was "
                        "not found."
                    ),
                })

            # Execute the allowed tools.
            if executable_calls:

                (
                    extra_messages,
                    collected_images,
                    collected_sources,
                ) = _execute_tool_calls(
                    tools=tools,
                    tool_calls=executable_calls,
                    conversation_id=conversation_id,
                    log_prefix="STREAM ",
                )

                current_messages.extend(extra_messages)
                all_tool_messages.extend(extra_messages)
                all_images.extend(collected_images)
                all_sources.extend(collected_sources)

            kb_search_count += allowed_kb_calls

            if allowed_kb_calls:

                deduped_sources = _deduplicate_sources(
                    all_sources
                )

                if deduped_sources:
                    yield {
                        "type": "sources",
                        "sources": deduped_sources,
                    }

            logger.info(
                "KB SEARCH ATTEMPTS: count=%s conversation_id=%s",
                kb_search_count,
                conversation_id,
            )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "LLM STREAM RESPONSE ERROR: conversation_id=%s error=%s",
            conversation_id,
            str(exc),
        )

        raise RuntimeError(
            f"Unable to generate streaming response: {exc}"
        ) from exc


# ============================================================
# GENERATE FOLLOW-UP SUGGESTIONS
# ============================================================


def generate_suggestions(
    question: str,
    answer: str,
    chat_history: Optional[list[dict]] = None,
) -> list[str]:
    """
    Generate exactly three concise follow-up questions based on
    the user's current question and the assistant's final answer.

    This function is intentionally independent of the existing
    RAG/tool/reasoning answer-generation flow. Suggestion generation
    must never cause the completed chat answer to fail.
    """

    try:
        if not question or not question.strip():
            return []

        if not answer or not answer.strip():
            return []

        recent_history = (chat_history or [])[-6:]

        history_text = "\n".join(
            f"{message.get('role', '')}: "
            f"{message.get('content', '')}"
            for message in recent_history
        )

        if not history_text:
            history_text = "No previous conversation history."

        suggestion_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You generate follow-up suggestions for an AI chatbot.

Based on the user's current question and the assistant's answer,
generate exactly 3 useful follow-up questions that the user could
ask next.

Rules:
- Return exactly 3 suggestions.
- Each suggestion must be a complete question.
- Keep each suggestion concise and natural.
- Suggestions must be relevant to the assistant's answer.
- Suggestions should help the user explore the topic further.
- Do not repeat the user's current question.
- Do not answer the suggestions.
- Do not include explanations.
- Do not include numbering.
- Do not include bullet points.
- Do not include markdown.
- Return ONLY a valid JSON array of strings.

Example output:
[
  "What are the main benefits?",
  "Can you explain this in more detail?",
  "What are the limitations?"
]
""",
                ),
                (
                    "human",
                    """
Conversation History:

{history}

Current User Question:

{question}

Assistant Answer:

{answer}
""",
                ),
            ]
        )

        messages = suggestion_prompt.format_messages(
            history=history_text,
            question=question.strip(),
            answer=answer.strip(),
        )

        response = llm.invoke(messages)
        content = response.content

        if not isinstance(content, str):
            logger.warning(
                "SUGGESTION GENERATION returned non-string content"
            )
            return []

        content = content.strip()

        if not content:
            return []

        if content.startswith("```"):
            if content.startswith("```json"):
                content = content[len("```json"):]
            else:
                content = content[len("```"):]

            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

        try:
            suggestions = json.loads(content)

        except json.JSONDecodeError:
            logger.warning(
                "SUGGESTION GENERATION returned invalid JSON: %s",
                content,
            )
            return []

        if not isinstance(suggestions, list):
            logger.warning(
                "SUGGESTION GENERATION returned non-list JSON"
            )
            return []

        cleaned_suggestions = []

        for suggestion in suggestions:

            if not isinstance(suggestion, str):
                continue

            suggestion = suggestion.strip()

            if not suggestion:
                continue

            if suggestion in cleaned_suggestions:
                continue

            cleaned_suggestions.append(suggestion)

            if len(cleaned_suggestions) == 3:
                break

        if len(cleaned_suggestions) != 3:
            logger.warning(
                "SUGGESTION GENERATION expected 3 suggestions, got %s",
                len(cleaned_suggestions),
            )
            return []

        logger.info(
            "SUGGESTIONS GENERATED: count=%s",
            len(cleaned_suggestions),
        )

        return cleaned_suggestions

    except Exception as exc:
        logger.exception(
            "SUGGESTION GENERATION ERROR: error=%s",
            str(exc),
        )
        return []


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