"""
demo.py — Phase 06: Multi-Agent Systems

Run: python demo.py

Demos:
  1. Single agent + task (minimal CrewAI example)
  2. Sequential crew with context passing (research -> write)
  3. Full content factory pipeline (research -> write -> edit -> format x3)
  4. Sequential vs parallel formatting — timing comparison
  5. Structured output extraction (Pydantic) from tasks
  6. Inspecting agent specialization (role/goal/backstory effect)
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

DIVIDER = "═" * 62

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: Minimal single agent + task
# ─────────────────────────────────────────────────────────────────────────────

def demo_1_minimal_crew():
    section("DEMO 1 — Minimal CrewAI Example (1 agent, 1 task)")

    from crewai import Agent, Task, Crew, Process, LLM

    llm = LLM(model="gpt-4o-mini", temperature=0.3)

    summarizer = Agent(
        role="Technical Summarizer",
        goal="Produce clear, accurate 2-sentence summaries of technical topics",
        backstory="You are an expert at distilling complex topics into their essence.",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Summarize what {topic} is, in exactly 2 sentences.",
        expected_output="A 2-sentence summary.",
        agent=summarizer,
    )

    crew = Crew(agents=[summarizer], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff(inputs={"topic": "the transformer architecture"})

    print(f"\n  Result: {result.raw}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Sequential context passing
# ─────────────────────────────────────────────────────────────────────────────

def demo_2_context_passing():
    section("DEMO 2 — Sequential Context Passing (Researcher → Writer)")

    from agents import researcher, writer
    from tasks import build_research_task, build_writing_task
    from crewai import Crew, Process

    research_task = build_research_task()
    writing_task  = build_writing_task(research_task)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": "the benefits of using LoRA for fine-tuning"})

    print(f"\n  Research task output (raw):")
    print(f"  {result.tasks_output[0].raw[:300]}...")
    print(f"\n  Writing task output (used research as context):")
    print(f"  {result.tasks_output[1].raw[:300]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Full content factory pipeline
# ─────────────────────────────────────────────────────────────────────────────

def demo_3_full_pipeline():
    section("DEMO 3 — Full Content Factory Pipeline")

    from content_factory import ContentFactory

    factory = ContentFactory(verbose=False)  # quiet for cleaner demo output
    result = factory.run("The shift from monolithic LLMs to mixture-of-experts architectures")

    result.pretty_print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Sequential vs parallel formatting timing
# ─────────────────────────────────────────────────────────────────────────────

def demo_4_parallel_vs_sequential():
    section("DEMO 4 — Sequential vs Parallel Formatting (timing comparison)")

    from content_factory import ContentFactory
    import asyncio

    factory = ContentFactory(verbose=False)

    # Use a fixed sample "edited article" to isolate the formatting phase timing
    sample_article = """
    # The Rise of Small Language Models

    In 2025, the AI industry has seen a notable shift: instead of chasing ever-larger
    models, leading labs are investing heavily in small language models (SLMs) that
    run efficiently on edge devices while maintaining strong performance on targeted tasks.

    Models like Phi-3, Gemma 2, and Llama 3.2's smaller variants demonstrate that with
    careful training data curation and architecture choices, a 3-8 billion parameter
    model can match or exceed the performance of models 10x its size on specific tasks.

    This shift matters because it democratizes AI deployment — SLMs can run on
    smartphones, IoT devices, and consumer laptops without requiring cloud infrastructure
    or expensive GPU access, opening AI capabilities to a much broader range of applications.
    """

    print("\n  Running formatting SEQUENTIALLY (one platform at a time)...")
    t0 = time.time()
    sequential_results = factory.run_sequential_formatting_for_comparison(sample_article)
    sequential_time = time.time() - t0
    print(f"  Sequential time: {sequential_time:.1f}s")

    print("\n  Running formatting in PARALLEL (all 3 platforms at once)...")
    t0 = time.time()
    parallel_results = asyncio.run(factory._format_all_platforms(sample_article))
    parallel_time = time.time() - t0
    print(f"  Parallel time:   {parallel_time:.1f}s")

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"\n  🚀 Speedup: {speedup:.1f}x faster with parallel execution")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Structured output extraction
# ─────────────────────────────────────────────────────────────────────────────

def demo_5_structured_output():
    section("DEMO 5 — Structured Output (Pydantic) from Tasks")

    from agents import researcher
    from tasks import build_research_task
    from crewai import Crew, Process

    research_task = build_research_task()
    crew = Crew(agents=[researcher], tasks=[research_task], process=Process.sequential, verbose=False)

    result = crew.kickoff(inputs={"topic": "WebAssembly adoption in 2025"})

    research_output = result.tasks_output[0]

    if research_output.pydantic:
        findings = research_output.pydantic
        print(f"\n  Structured ResearchFindings object:")
        print(f"  Confidence: {findings.confidence}")
        print(f"  Facts ({len(findings.facts)}):")
        for fact in findings.facts:
            print(f"    • {fact}")
        print(f"  Sources ({len(findings.sources)}):")
        for src in findings.sources:
            print(f"    - {src}")
    else:
        print(f"\n  Raw output (structured parsing didn't trigger):")
        print(f"  {research_output.raw[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Specialization effect — same task, different personas
# ─────────────────────────────────────────────────────────────────────────────

def demo_6_specialization_effect():
    section("DEMO 6 — Effect of Agent Specialization (role/goal/backstory)")

    from crewai import Agent, Task, Crew, Process, LLM

    llm = LLM(model="gpt-4o-mini", temperature=0.3)

    generic_agent = Agent(
        role="Assistant",
        goal="Help with the task",
        backstory="You are a helpful assistant.",
        llm=llm,
        verbose=False,
    )

    specialized_agent = Agent(
        role="Senior Technical Editor with 15 years at major publications",
        goal=(
            "Provide direct, specific, actionable editing feedback. Identify exactly "
            "which sentences are unclear and explain precisely why, with a suggested fix."
        ),
        backstory=(
            "You have edited thousands of technical articles. You are known for "
            "blunt, precise feedback — vague comments like 'this could be clearer' "
            "are unacceptable. You always point to the EXACT phrase and explain the issue."
        ),
        llm=llm,
        verbose=False,
    )

    sample_text = (
        "The new system is good and works well for users who need it. It has "
        "many features that help with various tasks and is pretty efficient overall."
    )

    for label, agent in [("GENERIC", generic_agent), ("SPECIALIZED", specialized_agent)]:
        task = Task(
            description=f"Give editing feedback on this paragraph:\n\n{sample_text}",
            expected_output="Specific editing feedback",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        print(f"\n  [{label} AGENT]")
        print(f"  {result.raw[:350]}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  Phase 06 — Multi-Agent Systems: All Demos")
    print(DIVIDER)

    if not os.getenv("TAVILY_API_KEY"):
        print("\n  ⚠ TAVILY_API_KEY not set — researcher's web_search will return errors.")
        print("  Get a free key at https://tavily.com and add it to .env\n")

    demo_1_minimal_crew()
    demo_2_context_passing()
    demo_3_full_pipeline()
    demo_4_parallel_vs_sequential()
    demo_5_structured_output()
    demo_6_specialization_effect()

    print(f"\n{DIVIDER}")
    print("  ✅ Phase 06 complete.")
    print("  You built: a 6-agent content factory with sequential research/write/edit")
    print("  and parallel multi-platform formatting.")
    print(DIVIDER)
