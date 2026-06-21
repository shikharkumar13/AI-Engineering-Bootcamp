"""
tools.py — Tools used by the Multi-Agent Content Factory

  - search_tool: web search for the researcher (Tavily-backed, with graceful fallback)
  - character_counter: check platform character limits
  - hashtag_generator: generate hashtags for social posts
"""

import os
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    from tavily import TavilyClient
    _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    _tavily_client = None


@tool("Web Search")
def search_tool(query: str) -> str:
    """Search the web for current, accurate information on a topic.
    Returns titles, URLs, and summaries of the top results. Use specific,
    well-formed search queries for best results."""

    if not _tavily_client:
        return (
            "ERROR: TAVILY_API_KEY not configured. Get a free key at "
            "https://tavily.com and add it to your .env file. "
            f"(Query was: '{query}')"
        )

    response = _tavily_client.search(query=query, max_results=5, search_depth="basic")
    results = response.get("results", [])

    if not results:
        return f"No results found for: '{query}'"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\nURL: {r['url']}\nSummary: {r['content'][:280]}"
        )
    return "\n\n".join(formatted)


@tool("Character Counter")
def character_counter(text: str) -> str:
    """Count the characters in a piece of text. Use this to verify content
    fits a platform's character limit (e.g. Twitter's 280-character limit
    per tweet, LinkedIn's ~1300 character sweet spot)."""
    count = len(text)
    return f"Character count: {count}"


@tool("Hashtag Generator")
def hashtag_generator(topic: str, count: int = 3) -> str:
    """Generate relevant hashtags for a topic, for use in social media posts.
    Provide the main topic and how many hashtags you want (default 3)."""
    words = [w for w in topic.replace(",", "").replace(".", "").split() if len(w) > 2]
    # Combine first two words into one camel-case hashtag, then individual words
    hashtags = []
    if len(words) >= 2:
        hashtags.append("#" + "".join(w.capitalize() for w in words[:2]))
    for w in words[2:2+count-1]:
        hashtags.append(f"#{w.capitalize()}")
    return " ".join(hashtags[:count]) if hashtags else "#General"
