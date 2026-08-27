import json
import logging
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


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to:
1. Conversation history
2. Conversation history tool
3. Uploaded-document knowledge-base search tool
4. Current date and time tool
5. Weather tool
6. Image analysis tool (analyze_image) — only present when the
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
        "YES. An uploaded document is available for this "
        "conversation. If the current question could be "
        "answered from the uploaded document, use "
        "search_knowledge_base before answering."
        if document_available
        else
        "No specific uploaded document is selected for "
        "this turn."
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
) -> Generator[str, None, None]:

    """
    document_id:
        If provided, scopes the knowledge-base search tool
        to that single uploaded document.

    image_paths:
        Local file path(s) of image(s) attached directly to
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

                    yield chunk.content

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

        for chunk in llm_with_tools.stream(
            final_messages
        ):

            if chunk.content:

                yield chunk.content

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