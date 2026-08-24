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


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
)


SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to:
1. Conversation history
2. Conversation history tool
3. Uploaded-document knowledge-base search tool
4. Current date and time tool
5. Weather tool

Rules:

- Answer normal conversational questions directly.
- Use get_conversation_history when the provided history is insufficient.
- Use search_knowledge_base whenever the answer may be present in uploaded
  documents or the knowledge base.
- If the user asks about an uploaded document, PDF, file, policy, manual,
  FAQ, guideline, documentation, or indexed content, search the knowledge
  base before answering.
- Use the retrieved document content as the source of truth for
  document-related questions.
- Do not invent information from uploaded documents.
- If the knowledge-base search finds no relevant information, say that the
  information was not found in the uploaded knowledge base.
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


def _build_messages(
    question: str,
    chat_history: Optional[list[dict]] = None,
):
    chat_history = chat_history or []

    recent_history = chat_history[-10:]

    history_text = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in recent_history
    )

    if not history_text:
        history_text = "No previous conversation history."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """Conversation History:

{history}

Current User Question:

{question}""",
            ),
        ]
    )

    return prompt.format_messages(
        history=history_text,
        question=question,
    )


def _create_tools(
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
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
    )


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


def generate_answer(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
):
    try:
        messages = _build_messages(
            question=question,
            chat_history=chat_history,
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        llm_with_tools = (
            llm.bind_tools(tools)
            if tools
            else llm
        )

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            return response.content

        logger.info(
            "LLM TOOL CALLS: tools=%s conversation_id=%s",
            [
                tool_call["name"]
                for tool_call in response.tool_calls
            ],
            conversation_id,
        )

        tool_messages = [response]

        for tool_call in response.tool_calls:
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
                "TOOL EXECUTING: tool=%s args=%s conversation_id=%s",
                tool_name,
                tool_args,
                conversation_id,
            )

            tool_result = selected_tool.invoke(tool_args)

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result),
                }
            )

        final_response = llm_with_tools.invoke(
            messages + tool_messages
        )

        return final_response.content

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


def _parse_tool_calls(tool_call_chunks: list):
    if not tool_call_chunks:
        return []

    tool_calls = {}

    for chunk in tool_call_chunks:
        index = chunk.get("index", 0)

        if index not in tool_calls:
            tool_calls[index] = {
                "id": "",
                "name": "",
                "args": "",
            }

        tool_calls[index]["id"] += chunk.get("id") or ""
        tool_calls[index]["name"] += chunk.get("name") or ""
        tool_calls[index]["args"] += chunk.get("args") or ""

    parsed_calls = []

    for tool_call in tool_calls.values():
        args_text = tool_call["args"].strip()

        if args_text:
            try:
                args = json.loads(args_text)
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


def generate_answer_stream(
    question: str,
    chat_history: Optional[list[dict]] = None,
    db=None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Generator[str, None, None]:
    try:
        messages = _build_messages(
            question=question,
            chat_history=chat_history,
        )

        tools = _create_tools(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        llm_with_tools = (
            llm.bind_tools(tools)
            if tools
            else llm
        )

        logger.info(
            "LLM STREAM START: conversation_id=%s tools=%s",
            conversation_id,
            [tool.name for tool in tools],
        )

        streamed_chunks = []
        tool_call_chunks = []

        for chunk in llm_with_tools.stream(messages):
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

        if not tool_calls:
            for chunk in streamed_chunks:
                if chunk.content:
                    yield chunk.content
            return

        logger.info(
            "STREAM TOOL CALLS: tools=%s conversation_id=%s",
            [
                tool_call["name"]
                for tool_call in tool_calls
            ],
            conversation_id,
        )

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
                "STREAM TOOL EXECUTING: tool=%s args=%s conversation_id=%s",
                tool_name,
                tool_args,
                conversation_id,
            )

            tool_result = selected_tool.invoke(
                tool_args
            )

            logger.info(
                "STREAM TOOL RESULT: tool=%s conversation_id=%s",
                tool_name,
                conversation_id,
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result),
                }
            )

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
            "LLM STREAM RESPONSE ERROR: conversation_id=%s error=%s",
            conversation_id,
            str(exc),
        )

        raise RuntimeError(
            f"Unable to generate streaming response: {exc}"
        ) from exc


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

        response = llm.invoke(messages)

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