import logging
import os
from typing import Any

from tavily import TavilyClient


logger = logging.getLogger(__name__)


class TavilyService:
    """Service wrapper around Tavily web search."""

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("TAVILY_API_KEY") or ""
        ).strip()

        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not configured"
            )

        self.client = TavilyClient(
            api_key=self.api_key
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the web using Tavily.

        Returns normalized search results containing:
        - title
        - url
        - content
        """

        query = (query or "").strip()

        if not query:
            return []

        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
            )

            results = response.get(
                "results",
                [],
            )

            normalized_results = []

            for result in results:

                if not isinstance(result, dict):
                    continue

                normalized_results.append(
                    {
                        "title": result.get(
                            "title",
                            "",
                        ),
                        "url": result.get(
                            "url",
                            "",
                        ),
                        "content": result.get(
                            "content",
                            "",
                        ),
                    }
                )

            return normalized_results

        except Exception:
            logger.exception(
                "TAVILY SEARCH FAILED"
            )

            raise


tavily_service = TavilyService()