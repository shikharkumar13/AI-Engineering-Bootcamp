# Phase 05 — AI Research Agent

## What this project does
An autonomous agent built with LangGraph that:
- Searches the web (Tavily) and reads pages to research any topic
- Tracks findings in a notepad as it goes
- Avoids infinite loops with a repeat-call guard
- Supports human-in-the-loop approval before tool execution
- Maintains conversation memory across turns
- Produces a structured, cited research report

## Project structure
```
Phase_5 AI Agents & LangGraph/
├── tools.py               ← All agent tools (search, fetch, calculator, notes)
├── research_agent.py      ← ResearchAgent class (LangGraph StateGraph)
├── demo.py                ← 7 demos
├── requirements.txt
└── .env.example           ← copy to .env and fill in your keys
```

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure .env
```bash
cp .env.example .env
```
`.env.example` contains:
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...     # free tier at https://tavily.com

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=phase05-research-agent
```

### 3. Run
```bash
python demo.py
```

## API reference

```python
from research_agent import ResearchAgent

# Basic research
agent = ResearchAgent()
result = agent.research("Latest developments in AI agents", verbose=True)
print(result["final_message"])
print(f"Took {result['step_count']} steps, saved {len(result['notes'])} findings")

# Structured report
report = agent.generate_report("RAG vs fine-tuning trade-offs")
print(report.title)
print(report.summary)
for finding in report.key_findings:
    print(f"- {finding}")

# Multi-turn with memory (same thread_id)
config_thread = "user-session-1"
agent.research("What is LangGraph?", thread_id=config_thread)
agent.research("How does it compare to AutoGPT?", thread_id=config_thread)  # remembers context

# Human-in-the-loop
agent_hitl = ResearchAgent(require_fetch_approval=True)
agent_hitl.research("...", thread_id="approval-demo")
pending = agent_hitl.get_pending_action("approval-demo")
if pending:
    print(f"Agent wants to: {pending['tool_calls']}")
    # ... get user approval ...
    agent_hitl.resume_after_approval("approval-demo")
```

## Concepts demonstrated

| Demo | Concept |
|------|---------|
| Demo 1 | Manual ReAct loop — Thought/Action/Observation from scratch |
| Demo 2 | Tool schemas — what the LLM actually sees |
| Demo 3 | Full LangGraph agent on a real research task |
| Demo 4 | Memory persistence across multiple turns (same thread_id) |
| Demo 5 | Loop detection guard |
| Demo 6 | Human-in-the-loop: interrupt_before + resume |
| Demo 7 | Structured report generation from agent findings |

## Architecture

```
        START
          │
          ▼
     ┌─────────┐
  ┌─►│  agent  │ (decides: call a tool, or conclude?)
  │  └────┬────┘
  │       │
  │  tools_condition
  │       │
  │   ┌───┴────┐
  │  YES      NO
  │   │        │
  │   ▼        ▼
  │ ┌──────┐  END
  └─┤tools │
    └──────┘
    (with loop guard + optional human approval)
```
