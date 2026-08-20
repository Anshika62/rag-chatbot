from fastapi import HTTPException, status

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.service.tools.conversation_tool import (
    create_conversation_tools
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to:

1. Conversation history
2. Retrieved document context
3. Conversation tools when required

Follow these rules:

- Use retrieved document context when the question is about
  uploaded documents.
- Use conversation history for follow-up questions.
- If information from the conversation is required and is not
  available in the provided history, use the available
  conversation tools.
- Do not invent information.
- If the answer is not available from the provided context,
  conversation history, or tools, clearly say that you do not
  have enough information.
- Keep answers clear and relevant.
"""


# ============================================================
# BUILD MESSAGES
# ============================================================

def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict] | None = None
):
    if chat_history is None:
        chat_history = []

    recent_history = chat_history[-10:]

    history_text = ""

    for message in recent_history:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role}: {content}\n"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
            ),
            (
                "human",
                """Conversation History:

{history}

Retrieved Document Context:

{context}

Current User Question:

{question}"""
            )
        ]
    )

    return prompt.format_messages(
        history=history_text,
        context=context or "No relevant document context found.",
        question=question
    )


# ============================================================
# CREATE TOOLS
# ============================================================

def _create_tools(
    db=None,
    conversation_id: int | None = None
):
    if db is None or conversation_id is None:
        return []

    return create_conversation_tools(
        db=db,
        conversation_id=conversation_id
    )


# ============================================================
# NORMAL ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
    db=None,
    conversation_id: int | None = None
):
    """
    Non-streaming RAG + conversation tool calling.

    Flow:

    User Question
        ↓
    RAG Context
        ↓
    Conversation History
        ↓
    LLM
        ↓
    Tool Call if required
        ↓
    Tool Result
        ↓
    Final LLM Answer
    """

    try:

        messages = _build_messages(
            question=question,
            context=context,
            chat_history=chat_history
        )

        # ----------------------------------------------------
        # Create request-scoped tools
        # ----------------------------------------------------

        tools = _create_tools(
            db=db,
            conversation_id=conversation_id
        )

        # ----------------------------------------------------
        # Bind tools
        # ----------------------------------------------------

        if tools:

            llm_with_tools = llm.bind_tools(
                tools
            )

        else:

            llm_with_tools = llm

        # ----------------------------------------------------
        # First LLM call
        # ----------------------------------------------------

        response = llm_with_tools.invoke(
            messages
        )

        # ----------------------------------------------------
        # Check tool calls
        # ----------------------------------------------------

        if response.tool_calls:

            tool_messages = [
                response
            ]

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call.get(
                    "args",
                    {}
                )

                selected_tool = next(
                    (
                        tool
                        for tool in tools
                        if tool.name == tool_name
                    ),
                    None
                )

                if selected_tool is None:

                    raise RuntimeError(
                        f"Requested tool not found: "
                        f"{tool_name}"
                    )

                # --------------------------------------------
                # Execute tool
                # --------------------------------------------

                tool_result = selected_tool.invoke(
                    tool_args
                )

                # --------------------------------------------
                # Add tool result
                # --------------------------------------------

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    }
                )

            # ------------------------------------------------
            # Final LLM call
            # ------------------------------------------------

            final_response = llm_with_tools.invoke(
                messages + tool_messages
            )

            return final_response.content

        # ----------------------------------------------------
        # Normal answer
        # ----------------------------------------------------

        return response.content

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate response from LLM"
        ) from exc


# ============================================================
# STREAMING ANSWER
# ============================================================

def generate_answer_stream(
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
    db=None,
    conversation_id: int | None = None
):
    """
    Streaming RAG + tool calling.

    Important:

    Tool execution itself is not streamed.

    Flow:

        First LLM call
             ↓
        Tool requested?
          /       \
        No        Yes
        ↓          ↓
      Stream    Execute tool
      answer       ↓
                   Final LLM
                     ↓
                   Stream
                   answer
    """

    try:

        messages = _build_messages(
            question=question,
            context=context,
            chat_history=chat_history
        )

        # ----------------------------------------------------
        # Create request-scoped tools
        # ----------------------------------------------------

        tools = _create_tools(
            db=db,
            conversation_id=conversation_id
        )

        # ----------------------------------------------------
        # Bind tools
        # ----------------------------------------------------

        if tools:

            llm_with_tools = llm.bind_tools(
                tools
            )

        else:

            llm_with_tools = llm

        # ----------------------------------------------------
        # First call
        #
        # We need invoke() first because we need to know
        # whether the model wants to call a tool.
        # ----------------------------------------------------

        response = llm_with_tools.invoke(
            messages
        )

        # ====================================================
        # TOOL CALL FLOW
        # ====================================================

        if response.tool_calls:

            tool_messages = [
                response
            ]

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call.get(
                    "args",
                    {}
                )

                selected_tool = next(
                    (
                        tool
                        for tool in tools
                        if tool.name == tool_name
                    ),
                    None
                )

                if selected_tool is None:

                    raise RuntimeError(
                        f"Requested tool not found: "
                        f"{tool_name}"
                    )

                # --------------------------------------------
                # Execute tool
                # --------------------------------------------

                tool_result = selected_tool.invoke(
                    tool_args
                )

                # --------------------------------------------
                # Add result
                # --------------------------------------------

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    }
                )

            # ------------------------------------------------
            # Final LLM response
            # ------------------------------------------------

            final_messages = (
                messages + tool_messages
            )

            # ------------------------------------------------
            # Stream final answer
            # ------------------------------------------------

            for chunk in llm_with_tools.stream(
                final_messages
            ):

                if not chunk.content:
                    continue

                yield chunk.content

            return

        # ====================================================
        # NORMAL STREAMING FLOW
        # ====================================================

        # The first response already contains the answer.
        # Stream it as one chunk to keep the interface
        # consistent with the normal streaming flow.

        if response.content:

            yield response.content

    except HTTPException:
        raise

    except Exception as exc:

        # IMPORTANT:
        #
        # Do NOT yield "[ERROR: ...]"
        #
        # Raise the exception so rag_service can create
        # the proper SSE error event.

        raise RuntimeError(
            f"Unable to generate streaming response: {exc}"
        ) from exc


# ============================================================
# GENERATE CONVERSATION TITLE
# ============================================================

def generate_title(
    question: str
) -> str:
    """
    Generate a short conversation title.
    """

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
- Return ONLY the title.
"""
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )

        messages = title_prompt.format_messages(
            question=question
        )

        response = llm.invoke(
            messages
        )

        title = response.content.strip()

        if not title:

            return question[:50]

        return title[:100]

    except Exception:

        return question[:50]