from langchain_core.tools import tool

from app.repository.conversation_repo import get_last_10_messages


def create_conversation_tools(
    db,
    conversation_id: int
):
    """
    Create conversation-related tools for the current conversation.
    """

    @tool
    def get_conversation_history() -> str:
        """
        Fetch recent messages from the current conversation.

        Use this tool when additional conversation history
        is required to answer the user's question.
        """

        messages = get_last_10_messages(
            db=db,
            conversation_id=conversation_id
        )

        if not messages:
            return "No previous conversation history found."

        history = []

        for message in messages:
            history.append(
                f"{message.role}: {message.content}"
            )

        return "\n".join(history)

    return [
        get_conversation_history
    ]