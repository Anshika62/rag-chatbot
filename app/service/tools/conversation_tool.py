from langchain_core.tools import tool

from app.repository.conversation_repo import (
    get_last_10_messages,
)

from app.service.tools.search_kb import (
    create_search_knowledge_base_tool,
    
)
from app.service.tools.datetime_tool import get_current_datetime
from app.service.tools.weather_tool import get_weather

def create_conversation_tools(
    db,
    user_id: str,
    conversation_id: str,
    document_id: str | None = None,
):
    """
    Create tools available to the current conversation.

    user_id, conversation_id, and (optionally) document_id are
    supplied by the authenticated/validated application context.

    These values are NOT exposed as LLM tool arguments.
    """

    @tool
    def get_conversation_history() -> str:
        """
        Fetch recent messages from the current conversation.

        The conversation context is injected by the application
        and is not provided by the LLM.
        """

        messages = get_last_10_messages(
            db=db,
            conversation_id=str(conversation_id),
        )

        if not messages:
            return "No previous conversation history found."

        return "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

    search_knowledge_base = create_search_knowledge_base_tool(
        user_id=str(user_id),
        conversation_id=str(conversation_id),
        document_id=(
            str(document_id)
            if document_id
            else None
        ),
    )

    return [
        get_conversation_history,
        search_knowledge_base,
        get_current_datetime,
        get_weather,
    ]