from langchain_core.tools import tool

from app.repository.conversation_repo import get_last_10_messages
from app.service.tools.search_kb import (
    create_search_knowledge_base_tool,
)


def create_conversation_tools(
    db,
    user_id: str,
    conversation_id: str,
):
    @tool
    def get_conversation_history() -> str:
        """Fetch recent messages from the current conversation."""

        messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id,
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
    )

    return [
        get_conversation_history,
        search_knowledge_base,
    ]