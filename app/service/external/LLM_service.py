from fastapi import HTTPException, status

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.service.tools.conversation_tool import create_conversation_tools


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


def generate_answer(
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
    db=None,
    conversation_id: int | None = None
):
    """
    Generate an answer using:
    1. Retrieved RAG context
    2. Previous conversation messages
    3. Current user question
    4. External tools when required
    """

    try:
        if chat_history is None:
            chat_history = []

        recent_history = chat_history[-10:]

        history_text = ""

        for message in recent_history:
            role = message.get("role", "")
            content = message.get("content", "")

            history_text += f"{role}: {content}\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Answer the user's question using BOTH the conversation history
AND the retrieved document context, as appropriate.

- If the question is a follow-up or refers to something mentioned
  earlier in the conversation, use the conversation history.
- If the question is about document content, use the retrieved context.
- If additional conversation information is required, use the available
  conversation tools.
- If neither the history nor the context has the answer, say so clearly.

Do not make up information."""
                ),
                (
                    "human",
                    """Conversation History:
{history}

Retrieved Context:
{context}

Current Question:
{question}"""
                )
            ]
        )

        messages = prompt.format_messages(
            history=history_text,
            context=context,
            question=question
        )

        # Create request-specific tools using the current DB session
        tools = []

        if db is not None and conversation_id is not None:
            tools = create_conversation_tools(
                db=db,
                conversation_id=conversation_id
            )

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # First LLM call
        response = llm_with_tools.invoke(messages)

        # Check whether LLM requested a tool
        if response.tool_calls:

            tool_messages = [response]

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Find requested tool
                selected_tool = next(
                    (
                        tool
                        for tool in tools
                        if tool.name == tool_name
                    ),
                    None
                )

                if selected_tool is None:
                    continue

                # Execute tool
                tool_result = selected_tool.invoke(tool_args)

                # Add tool result to conversation
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    }
                )

            # Send tool result back to LLM
            final_response = llm_with_tools.invoke(
                messages + tool_messages
            )

            return final_response.content

        # Normal response when no tool is required
        return response.content

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate response from LLM"
        )


def generate_answer_stream(
    question: str,
    context: str,
    chat_history: list[dict] | None = None
):
    """
    Streams the LLM response token by token as a generator.
    """

    try:
        if chat_history is None:
            chat_history = []

        recent_history = chat_history[-10:]

        history_text = ""

        for message in recent_history:
            role = message.get("role", "")
            content = message.get("content", "")

            history_text += f"{role}: {content}\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Answer the user's question using BOTH the conversation history
AND the retrieved document context, as appropriate.

- If the question is a follow-up or refers to something mentioned
  earlier in the conversation, use the conversation history.
- If the question is about document content, use the retrieved context.
- If neither the history nor the context has the answer, say so clearly.

Do not make up information."""
                ),
                (
                    "human",
                    """Conversation History:
{history}

Retrieved Context:
{context}

Current Question:
{question}"""
                )
            ]
        )

        messages = prompt.format_messages(
            history=history_text,
            context=context,
            question=question
        )

        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        yield f"[ERROR: Unable to generate response: {str(e)}]"


def generate_title(question: str) -> str:
    """
    Generates a short conversation title from the first user question.
    """

    try:
        title_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate a short, clear title (3-6 words) for a
conversation that starts with the given user message.
Do not use quotes. Do not add punctuation at the end.
Return ONLY the title text, nothing else."""
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

        response = llm.invoke(messages)

        title = response.content.strip()

        return title if title else question[:50]

    except Exception:
        return question[:50]