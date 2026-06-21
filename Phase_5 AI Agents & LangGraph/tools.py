"""
tools.py — Tools for the AI Research Agent

Each tool is decorated with @tool, giving the LLM a name, description,
and argument schema it can use to decide when and how to call it.
"""

import os
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Tavily is optional — fall back to DuckDuckGo-style scraping if no API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    from tavily import TavilyClient
    _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    _tavily_client = None


# ── Web search ──────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Use this to find recent facts, news, statistics, or any information that
    requires up-to-date knowledge beyond your training data. Returns a list of
    results with titles, URLs, and short summaries.

    Args:
        query: The search query — be specific and use natural search terms
        max_results: Number of results to return (default 5, max 10)
    """
    max_results = min(max_results, 10)

    if _tavily_client:
        response = _tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        if not results:
            return f"No results found for: '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Summary: {r['content'][:300]}"
            )
        return "\n\n".join(formatted)

    else:
        # Fallback: no Tavily key configured
        return (
            "ERROR: TAVILY_API_KEY not configured. "
            "Get a free key at https://tavily.com and add it to your .env file. "
            f"(Query was: '{query}')"
        )


# ── Page fetching ───────────────────────────────────────────────────────────────

@tool
def fetch_page_content(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract the readable text content of a specific web page.

    Use this AFTER web_search, to read the full content of a promising result
    that the search summary alone didn't fully answer. Do not use this for
    general searching — use web_search for that.

    Args:
        url: The exact URL to fetch (must include http:// or https://)
        max_chars: Maximum characters to return (default 4000, to control token usage)
    """
    try:
        response = httpx.get(
            url,
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [line for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [content truncated]"

        if not text.strip():
            return f"Page loaded but contained no extractable text: {url}"

        return text

    except httpx.TimeoutException:
        return f"Failed to fetch {url}: request timed out after 10s"
    except httpx.HTTPStatusError as e:
        return f"Failed to fetch {url}: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Failed to fetch {url}: {type(e).__name__}: {e}"


# ── Calculator ──────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Use this for any arithmetic, percentages, averages, or numeric calculations
    needed while researching (e.g. comparing statistics, computing growth rates).

    Args:
        expression: A math expression, e.g. "150 * 1.08", "(45+67+23)/3", "2**10"
    """
    try:
        import numexpr
        result = numexpr.evaluate(expression).item()
        return str(result)
    except ImportError:
        # Fallback to restricted eval if numexpr isn't installed
        try:
            allowed = {"__builtins__": {}}
            result = eval(expression, allowed, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ── Date/time ───────────────────────────────────────────────────────────────────

@tool
def current_date() -> str:
    """Return today's date. Use this when the research topic involves
    recency (e.g. 'latest', 'this year', 'recent developments') so you know
    what 'current' means in context.
    """
    from datetime import datetime
    return datetime.now().strftime("%A, %B %d, %Y")


# ── Note-taking (agent's working memory) ─────────────────────────────────────────

class ResearchNotes:
    """
    A simple in-memory notepad the agent can use to track findings across
    multiple search/fetch calls during one research task.
    """
    def __init__(self):
        self.notes: list[dict] = []

    def add(self, finding: str, source: str):
        self.notes.append({"finding": finding, "source": source})

    def all(self) -> list[dict]:
        return self.notes

    def clear(self):
        self.notes = []

    def formatted(self) -> str:
        if not self.notes:
            return "(no notes yet)"
        return "\n".join(
            f"{i+1}. {n['finding']} (source: {n['source']})"
            for i, n in enumerate(self.notes)
        )


_research_notes = ResearchNotes()


@tool
def save_finding(finding: str, source: str) -> str:
    """Save an important fact or finding to your research notes, along with
    its source. Use this whenever you discover a fact worth including in the
    final report — this builds your evidence base as you research.

    Args:
        finding: The specific fact or insight discovered
        source: Where this finding came from (URL or search result title)
    """
    _research_notes.add(finding, source)
    return f"Saved. You now have {len(_research_notes.all())} notes."


@tool
def review_notes() -> str:
    """Review all findings saved so far in this research session.
    Use this before writing your final report to make sure you have
    covered all the important points you discovered.
    """
    return _research_notes.formatted()


def get_notes_store() -> ResearchNotes:
    """Access the underlying notes store (used by the agent for report generation)."""
    return _research_notes


def reset_notes():
    """Clear notes between research sessions."""
    _research_notes.clear()


# ── Tool registry ───────────────────────────────────────────────────────────────

ALL_TOOLS = [
    web_search,
    fetch_page_content,
    calculator,
    current_date,
    save_finding,
    review_notes,
]
