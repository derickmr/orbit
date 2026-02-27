from __future__ import annotations

from tavily import TavilyClient
from backend.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

_client = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str) -> list[dict]:
    """Search the web via Tavily. Returns list of {title, url, content, score}."""
    response = _client.search(
        query=query,
        max_results=TAVILY_MAX_RESULTS,
        search_depth="advanced",
    )
    return response.get("results", [])
