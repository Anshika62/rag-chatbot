from langchain_core.tools import tool

from app.service.external.tavily_service import (
    tavily_service,
)


@tool
def tavily_web_search(
    query: str,
) -> list[dict]:
    """
    Search the public web for current or external information.

    Use this tool when the user's question requires information
    that may be current, changing, or not available in the
    uploaded knowledge base.

    Do not use this tool for questions that can be answered
    directly from the conversation or uploaded documents.
    """

    return tavily_service.search(
        query=query,
        max_results=5,
    )