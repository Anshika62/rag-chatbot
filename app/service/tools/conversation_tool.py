from langchain_core.tools import tool

from app.repository.conversation_repo import (
    get_last_10_messages,
)

from app.service.tools.search_kb import (
    create_search_knowledge_base_tool,
)

from app.service.tools.datetime_tool import (
    get_current_datetime,
)

from app.service.tools.weather_tool import (
    get_weather,
)


# ============================================================
# CREATE CONVERSATION TOOLS
# ============================================================


def create_conversation_tools(
    db,
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
):
    """
    Create all tools available to the current conversation.

    The application injects:
        - db
        - user_id
        - conversation_id
        - optional document_id

    These values are NOT exposed as arguments to the LLM.

    Available tools:

    1. get_conversation_history
       -> Previous messages from the current conversation.

    2. search_knowledge_base
       -> Uploaded documents / PDFs / images / CSV / Excel /
          DOCX / TXT / Markdown knowledge-base search.

    3. get_current_datetime
       -> Current date and time.

    4. get_weather
       -> Current weather information.
    """

    user_id = str(user_id)
    conversation_id = str(conversation_id)

    if document_id:
        document_id = str(document_id)

    # ========================================================
    # CONVERSATION HISTORY TOOL
    # ========================================================

    @tool
    def get_conversation_history() -> str:
        """
        Fetch recent messages from the current conversation.

        Use this when the current prompt does not contain enough
        context to answer a follow-up question.

        The conversation_id is injected by the application and
        must never be supplied by the LLM.
        """

        messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
        )

        if not messages:
            return (
                "No previous conversation history found."
            )

        return "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

    # ========================================================
    # KNOWLEDGE BASE TOOL
    # ========================================================

    search_knowledge_base = (
        create_search_knowledge_base_tool(
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
        )
    )

    # ========================================================
    # RETURN ALL TOOLS
    #
    # Each responsibility remains a separate tool.
    # The LLM decides which tool is required.
    # ========================================================

    return [
        get_conversation_history,
        search_knowledge_base,
        get_current_datetime,
        get_weather,
    ]