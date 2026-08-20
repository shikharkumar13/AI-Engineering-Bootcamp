"""
content_factory.py — Multi-Agent Content Factory orchestrator

Pipeline:
  Researcher → Writer → Editor → [Blog, LinkedIn, Twitter formatters in parallel]

Demonstrates:
  - Sequential CrewAI process for the research/write/edit phase
  - True parallel execution for the platform-formatting phase (asyncio.gather)
  - Structured output extraction (Pydantic) from each task
  - Cost/timing instrumentation across the whole pipeline
"""

import time
import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv
from crewai import Crew, Process

from agents import researcher, writer, editor, blog_formatter, linkedin_formatter, twitter_formatter
from tasks import (
    build_research_task, build_writing_task, build_editing_task,
    build_blog_task, build_linkedin_task, build_twitter_task,
    ResearchFindings, PlatformContent,
)

load_dotenv()


@dataclass
class PipelineResult:
    topic:              str
    research_findings:  ResearchFindings | None
    draft:              str
    edited_article:     str
    platform_content:   dict[str, str] = field(default_factory=dict)
    timings:            dict[str, float] = field(default_factory=dict)

    def pretty_print(self):
        print(f"\n{'═'*64}")
        print(f"  CONTENT FACTORY RESULT: {self.topic}")
        print(f"{'═'*64}")

        print(f"\n📋 Research:")
        if self.research_findings:
            for fact in self.research_findings.facts[:5]:
                print(f"  • {fact}")

        print(f"\n✍️  Edited Article "
              f"({self.timings.get('research_write_edit', 0):.1f}s research+write+edit, "
              f"{len(self.edited_article)} chars):")
        print(f"  {self.edited_article[:300]}...")

        print(f"\n📢 Platform Content ({self.timings.get('formatting', 0):.1f}s parallel):")
        for platform, content in self.platform_content.items():
            print(f"\n  [{platform.upper()}] ({len(content)} chars)")
            print(f"  {content[:200]}...")

        total = sum(self.timings.values())
        print(f"\n⏱  Total pipeline time: {total:.1f}s")
        print(f"{'═'*64}\n")


class ContentFactory:
    """
    Orchestrates the full multi-agent content pipeline.

    Usage:
        factory = ContentFactory()
        result = factory.run("The rise of small language models in 2025")
        result.pretty_print()
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, topic: str) -> PipelineResult:
        """
        Run the full pipeline: sequential research/write/edit,
        then parallel formatting for blog/LinkedIn/Twitter.
        """
        timings = {}

        # ── Phase 1: Sequential research → write → edit ────────────────────
        t0 = time.time()
        research_task = build_research_task()
        writing_task  = build_writing_task(research_task)
        editing_task  = build_editing_task(writing_task, research_task)

        core_crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        core_result = core_crew.kickoff(inputs={"topic": topic})
        timings["research_write_edit"] = time.time() - t0

        # Extract structured research output
        research_output = core_result.tasks_output[0]
        research_findings = research_output.pydantic if research_output.pydantic else None

        edited_article = core_result.raw   # editing_task's output (last task)

        # ── Phase 2: Parallel formatting ────────────────────────────────────
        t0 = time.time()
        platform_content = asyncio.run(self._format_all_platforms(edited_article))
        timings["formatting"] = time.time() - t0

        return PipelineResult(
            topic=topic,
            research_findings=research_findings,
            draft=core_result.tasks_output[1].raw,  # writing_task's output
            edited_article=edited_article,
            platform_content=platform_content,
            timings=timings,
        )

    async def _format_all_platforms(self, edited_article: str) -> dict[str, str]:
        """
        Run blog/LinkedIn/Twitter formatting concurrently.
        Each runs as its own single-agent crew, so they have no shared
        dependency and can execute in parallel — unlike the research/write/edit
        phase, which is inherently sequential.
        """
        loop = asyncio.get_event_loop()

        def run_formatter(agent, platform_name: str) -> str:
            from crewai import Task as CrewTask
            task = CrewTask(
                description=(
                    f"Format this article for {platform_name}:\n\n{edited_article}"
                ),
                expected_output=f"Content formatted appropriately for {platform_name}",
                agent=agent,
            )
            mini_crew = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                              verbose=self.verbose)
            result = mini_crew.kickoff()
            return result.raw

        tasks = [
            loop.run_in_executor(None, run_formatter, blog_formatter,     "a blog post"),
            loop.run_in_executor(None, run_formatter, linkedin_formatter, "LinkedIn"),
            loop.run_in_executor(None, run_formatter, twitter_formatter,  "Twitter/X thread"),
        ]

        blog, linkedin, twitter = await asyncio.gather(*tasks)

        return {"blog": blog, "linkedin": linkedin, "twitter": twitter}

    def run_sequential_formatting_for_comparison(self, edited_article: str) -> dict:
        """
        Same formatting phase but sequential — for comparing timing against
        the parallel version. See demo.py for the side-by-side benchmark.
        """
        from crewai import Task as CrewTask

        results = {}
        for agent, platform_name, key in [
            (blog_formatter,     "a blog post",       "blog"),
            (linkedin_formatter, "LinkedIn",           "linkedin"),
            (twitter_formatter,  "Twitter/X thread",   "twitter"),
        ]:
            task = CrewTask(
                description=f"Format this article for {platform_name}:\n\n{edited_article}",
                expected_output=f"Content formatted for {platform_name}",
                agent=agent,
            )
            mini_crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
            result = mini_crew.kickoff()
            results[key] = result.raw

        return results
