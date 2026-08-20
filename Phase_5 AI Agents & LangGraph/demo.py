"""
demo.py — Phase 05: AI Agents & LangGraph

Run: python demo.py

Demos:
  1. Manual ReAct loop (understand the mechanism)
  2. Tool binding and inspection
  3. LangGraph agent — basic research task
  4. Memory across multiple turns (same thread_id)
  5. Loop detection
  6. Human-in-the-loop approval
  7. Full research report generation
"""

import os
from dotenv import load_dotenv

load_dotenv()

DIVIDER = "═" * 62

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: Manual ReAct (no framework — understand the mechanism)
# ─────────────────────────────────────────────────────────────────────────────

def demo_1_manual_react():
    section("DEMO 1 — Manual ReAct Loop (no framework)")

    from openai import OpenAI
    import re

    client = OpenAI()

    def fake_search(query: str) -> str:
        db = {
            "capital of japan": "Tokyo is the capital of Japan.",
            "tokyo population":  "Tokyo's metropolitan area has approximately 37 million people, "
                                  "the largest in the world.",
        }
        key = query.lower().strip()
        for k, v in db.items():
            if k in key or key in k:
                return v
        return f"No results for '{query}'"

    system_prompt = """Solve the task step by step using this format:

Thought: <reasoning>
Action: search
Action Input: <query>

After seeing the Observation, continue or conclude with:
Thought: I now have enough information.
Final Answer: <answer>"""

    question = "What is the capital of Japan and what is its population?"
    scratchpad = f"Question: {question}\n"

    print(f"\n  Question: {question}\n")

    for step in range(4):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scratchpad},
            ],
            temperature=0,
            stop=["Observation:"],
        )
        text = response.choices[0].message.content
        print(f"  --- Step {step+1} ---")
        print(f"  {text}\n")
        scratchpad += text

        if "Final Answer:" in text:
            break

        match = re.search(r"Action Input:\s*(.+)", text)
        if match:
            query = match.group(1).strip().strip('"')
            observation = fake_search(query)
            print(f"  Observation: {observation}\n")
            scratchpad += f"\nObservation: {observation}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Tool inspection
# ─────────────────────────────────────────────────────────────────────────────

def demo_2_tool_inspection():
    section("DEMO 2 — Tool Definitions (what the LLM sees)")

    from tools import ALL_TOOLS

    for t in ALL_TOOLS:
        print(f"\n  Tool: {t.name}")
        print(f"  Description: {t.description[:100]}...")
        print(f"  Args schema: {t.args}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Basic research task
# ─────────────────────────────────────────────────────────────────────────────

def demo_3_basic_research():
    section("DEMO 3 — LangGraph Research Agent (basic task)")

    from research_agent import ResearchAgent

    agent = ResearchAgent()
    result = agent.research(
        "What is the current state of the art in LLM context window sizes?",
        thread_id="demo3",
        verbose=True,
    )

    print(f"\n  Total steps: {result['step_count']}")
    print(f"  Notes saved: {len(result['notes'])}")
    print(f"\n  Final message:\n  {result['final_message'][:400]}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Memory across turns
# ─────────────────────────────────────────────────────────────────────────────

def demo_4_memory():
    section("DEMO 4 — Memory Across Multiple Turns")

    from research_agent import ResearchAgent
    from langchain_core.messages import HumanMessage

    agent = ResearchAgent()
    thread_id = "demo4-memory"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n  Turn 1: asking about a topic")
    result1 = agent.graph.invoke(
        {"messages": [HumanMessage(content="What is LangGraph used for?")], "step_count": 0},
        config=config,
    )
    print(f"  Response: {result1['messages'][-1].content[:200]}")

    print("\n  Turn 2: follow-up referring back to Turn 1")
    result2 = agent.graph.invoke(
        {"messages": [HumanMessage(content="How is it different from plain LangChain chains?")],
         "step_count": result1.get("step_count", 0)},
        config=config,
    )
    print(f"  Response: {result2['messages'][-1].content[:200]}")
    print("\n  (Notice: the agent understood 'it' refers to LangGraph from Turn 1)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Loop detection
# ─────────────────────────────────────────────────────────────────────────────

def demo_5_loop_detection():
    section("DEMO 5 — Loop Detection")

    from research_agent import LoopGuard

    guard = LoopGuard(max_repeated=3)

    print("\n  Simulating repeated identical tool calls:")
    for i in range(5):
        stuck, reason = guard.check("web_search", {"query": "same query every time"})
        status = "🛑 STOPPED" if stuck else "✓ continuing"
        print(f"  Call {i+1}: {status}" + (f" — {reason}" if stuck else ""))
        if stuck:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Human-in-the-loop approval
# ─────────────────────────────────────────────────────────────────────────────

def demo_6_human_in_the_loop():
    section("DEMO 6 — Human-in-the-Loop Approval")

    from research_agent import ResearchAgent
    from langchain_core.messages import HumanMessage, SystemMessage
    from research_agent import RESEARCH_SYSTEM_PROMPT

    agent = ResearchAgent(require_fetch_approval=True)
    thread_id = "demo6-hitl"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n  Starting research with approval required before EVERY tool call...")

    initial_state = {
        "messages": [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content="Search for the latest Python version released."),
        ],
        "step_count": 0,
    }

    result = agent.graph.invoke(initial_state, config=config)

    pending = agent.get_pending_action(thread_id)
    if pending:
        print(f"\n  ⏸  Agent paused before executing:")
        for tc in pending["tool_calls"]:
            print(f"     {tc['name']}({tc['args']})")

        # Simulate approval (in a real app, this comes from user input)
        print(f"\n  ✓ Auto-approving for demo purposes...")
        result = agent.resume_after_approval(thread_id)

        # May pause again for the next tool call
        pending2 = agent.get_pending_action(thread_id)
        attempts = 0
        while pending2 and attempts < 5:
            print(f"\n  ⏸  Agent paused again before:")
            for tc in pending2["tool_calls"]:
                print(f"     {tc['name']}({tc['args']})")
            print(f"  ✓ Auto-approving...")
            result = agent.resume_after_approval(thread_id)
            pending2 = agent.get_pending_action(thread_id)
            attempts += 1

        print(f"\n  Final answer: {result['messages'][-1].content[:300]}")
    else:
        print(f"\n  No tool call was needed: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: Full structured report
# ─────────────────────────────────────────────────────────────────────────────

def demo_7_research_report():
    section("DEMO 7 — Full Research Report Generation")

    from research_agent import ResearchAgent

    agent = ResearchAgent()
    topic = "Key differences between RAG and fine-tuning for LLM customization"

    print(f"\n  Generating report on: '{topic}'")
    print(f"  (This runs the full research loop, then synthesizes a report)\n")

    report = agent.generate_report(topic)

    print(f"\n{'─'*60}")
    print(f"REPORT: {report.title}")
    print(f"{'─'*60}")
    print(f"\nSummary:\n  {report.summary}")
    print(f"\nKey Findings:")
    for i, finding in enumerate(report.key_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\nSources:")
    for src in report.sources:
        print(f"  - {src}")
    if report.open_questions:
        print(f"\nOpen Questions:")
        for q in report.open_questions:
            print(f"  ? {q}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  Phase 05 — AI Agents & LangGraph: All Demos")
    print(DIVIDER)

    if not os.getenv("TAVILY_API_KEY"):
        print("\n  ⚠ TAVILY_API_KEY not set — web_search will return errors.")
        print("  Get a free key at https://tavily.com and add it to .env")
        print("  Demos will still run to show the agent's reasoning structure.\n")

    demo_1_manual_react()
    demo_2_tool_inspection()
    demo_3_basic_research()
    demo_4_memory()
    demo_5_loop_detection()
    demo_6_human_in_the_loop()
    demo_7_research_report()

    print(f"\n{DIVIDER}")
    print("  ✅ Phase 05 complete.")
    print("  You built: a LangGraph research agent with tools, memory,")
    print("  loop protection, human-in-the-loop approval, and structured reports.")
    print(DIVIDER)
