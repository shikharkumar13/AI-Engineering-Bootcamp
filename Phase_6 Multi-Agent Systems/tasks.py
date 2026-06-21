"""
tasks.py — Task definitions for the Multi-Agent Content Factory

Defines the sequential pipeline tasks (research -> write -> edit) and the
three parallel formatting tasks (blog, LinkedIn, Twitter).

Demonstrates:
  - context=[...] for explicit inter-agent context passing
  - output_pydantic for structured task outputs
  - {topic} templated task descriptions (filled at kickoff time)
"""

from crewai import Task
from pydantic import BaseModel, Field

from agents import researcher, writer, editor, blog_formatter, linkedin_formatter, twitter_formatter


# ── Structured output schemas ──────────────────────────────────────────────────

class ResearchFindings(BaseModel):
    facts: list[str] = Field(description="Specific, verifiable facts discovered, each with inline source")
    sources: list[str] = Field(description="List of distinct sources used")
    confidence: str = Field(description="Overall confidence in findings: high, medium, or low")


class PlatformContent(BaseModel):
    platform: str = Field(description="The target platform: blog, linkedin, or twitter")
    content: str = Field(description="The fully formatted content ready to publish")
    character_count: int = Field(description="Total character count of the content")


# ── Pipeline tasks (sequential) ────────────────────────────────────────────────

def build_research_task() -> Task:
    return Task(
        description=(
            "Research the following topic thoroughly: {topic}\n\n"
            "Find at least 5 specific, verifiable facts using web search. "
            "For each fact, note exactly where it came from. Prioritize information "
            "from the last 12 months if the topic is time-sensitive. Distinguish "
            "clearly between confirmed facts and speculation or opinion."
        ),
        expected_output=(
            "Structured research findings: a list of 5-8 facts each with a source, "
            "a list of distinct sources, and an overall confidence rating."
        ),
        agent=researcher,
        output_pydantic=ResearchFindings,
    )


def build_writing_task(research_task: Task) -> Task:
    return Task(
        description=(
            "Using the research findings provided, write a clear, engaging "
            "600-800 word article on: {topic}\n\n"
            "Structure: a compelling hook opening, 3-4 body sections with "
            "subheadings, and a concise conclusion. Cite facts naturally in "
            "the prose — don't just list them."
        ),
        expected_output="A complete article draft in markdown format with headers.",
        agent=writer,
        context=[research_task],
    )


def build_editing_task(writing_task: Task, research_task: Task) -> Task:
    return Task(
        description=(
            "Edit the draft article for clarity, accuracy, tone consistency, "
            "and engagement. Cross-check claims against the original research "
            "to catch any inaccuracies introduced during writing. Fix unclear "
            "sentences and strengthen weak transitions. Preserve the overall "
            "structure and voice — refine, don't rewrite from scratch."
        ),
        expected_output="A polished, publication-ready final article in markdown.",
        agent=editor,
        context=[writing_task, research_task],
    )


# ── Formatting tasks (can run in parallel after editing) ──────────────────────

def build_blog_task(editing_task: Task) -> Task:
    return Task(
        description=(
            "Format the edited article as a publication-ready blog post. Add "
            "SEO-friendly H2/H3 headers, a one-sentence meta description at the "
            "top, and ensure proper markdown formatting throughout."
        ),
        expected_output="The final blog post in markdown, with a meta description.",
        agent=blog_formatter,
        context=[editing_task],
        output_pydantic=PlatformContent,
    )


def build_linkedin_task(editing_task: Task) -> Task:
    return Task(
        description=(
            "Adapt the edited article into a LinkedIn post. Use a strong hook "
            "as the first line, short paragraphs, and an authentic professional "
            "voice. Keep the TOTAL content under 1300 characters. End with a "
            "thought-provoking question to drive engagement. Include 2-3 relevant "
            "hashtags at the end."
        ),
        expected_output="A complete LinkedIn post, under 1300 characters, with hashtags.",
        agent=linkedin_formatter,
        context=[editing_task],
        output_pydantic=PlatformContent,
    )


def build_twitter_task(editing_task: Task) -> Task:
    return Task(
        description=(
            "Adapt the edited article into a Twitter/X thread of 5-7 tweets. "
            "EVERY tweet must be under 280 characters — verify each one with "
            "the character counter tool. The first tweet must hook attention "
            "immediately. Number each tweet (1/7, 2/7, etc). Add 2-3 relevant "
            "hashtags to the final tweet only."
        ),
        expected_output="A numbered Twitter thread, each tweet under 280 characters.",
        agent=twitter_formatter,
        context=[editing_task],
        output_pydantic=PlatformContent,
    )
