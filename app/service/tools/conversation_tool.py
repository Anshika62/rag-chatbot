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

from app.service.tools.image_tool import (
    create_image_tool,
)


# ============================================================
# CREATE CONVERSATION TOOLS
# ============================================================


def create_conversation_tools(
    db,
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
    image_paths: list[str] | None = None,
):
    """
    Create all tools available to the current conversation.

    The application injects:
        - db
        - user_id
        - conversation_id
        - optional document_id
        - optional image_paths

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

    5. analyze_image (only when image_paths is provided)
       -> Vision analysis of image(s) attached directly to the
          CURRENT chat message (not a previously uploaded/indexed
          document). Used for "what's in this picture I just
          sent" style questions, answered live against the actual
          image bytes instead of a stored caption.
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

    tools = [
        get_conversation_history,
        search_knowledge_base,
        get_current_datetime,
        get_weather,
    ]

    # ========================================================
    # IMAGE ANALYSIS TOOL (only when an image is attached to
    # THIS message)
    # ========================================================

    valid_image_paths = [
        path
        for path in (image_paths or [])
        if path
    ]

    if valid_image_paths:

        tools.append(
            create_image_tool(
                image_paths=valid_image_paths,
            )
        )

    return tools