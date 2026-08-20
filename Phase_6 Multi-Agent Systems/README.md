# Phase 06 — Multi-Agent Content Factory

## What this project does
A 6-agent CrewAI pipeline that turns a topic into publication-ready content for
three platforms:

```
Researcher → Writer → Editor → ┬─► Blog Formatter
                                ├─► LinkedIn Formatter   (parallel)
                                └─► Twitter Formatter
```

- **Researcher** finds 5+ sourced facts via web search
- **Writer** drafts a 600-800 word article from the research
- **Editor** refines the draft, cross-checking claims against research
- **Three formatters** run in parallel, adapting the final article for
  blog (SEO headers), LinkedIn (1300 char, engagement question), and
  Twitter (5-7 tweet thread, 280 char/tweet)

## Project structure
```
Phase_6 Multi-Agent Systems/
├── agents.py            ← 6 agent definitions (role/goal/backstory)
├── tasks.py              ← Task definitions with context-passing wiring
├── tools.py               ← search_tool, character_counter, hashtag_generator
├── content_factory.py     ← ContentFactory orchestrator (sequential + parallel)
├── demo.py                ← 6 demos
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
TAVILY_API_KEY=tvly-...    # free tier at https://tavily.com
```

### 3. Run
```bash
python demo.py
```

## API reference

```python
from content_factory import ContentFactory

factory = ContentFactory(verbose=True)
result = factory.run("The future of multimodal AI models")

# Structured research findings
print(result.research_findings.facts)
print(result.research_findings.confidence)

# Article versions
print(result.draft)             # writer's first draft
print(result.edited_article)    # editor's polished version

# Platform-specific content (generated in parallel)
print(result.platform_content["blog"])
print(result.platform_content["linkedin"])
print(result.platform_content["twitter"])

# Timing breakdown
print(result.timings)
# {"research_write_edit": 24.3, "formatting": 8.1}

result.pretty_print()  # formatted console output
```

## Concepts demonstrated

| Demo | Concept |
|------|---------|
| Demo 1 | Minimal CrewAI: 1 Agent + 1 Task + 1 Crew |
| Demo 2 | Sequential context passing (`context=[research_task]`) |
| Demo 3 | Full pipeline: sequential core + parallel formatting fan-out |
| Demo 4 | Sequential vs parallel formatting — measured speedup |
| Demo 5 | `output_pydantic` — structured task output |
| Demo 6 | Specialization effect: generic vs specific role/goal/backstory |

## Why sequential THEN parallel

Research → Write → Edit is inherently sequential: you cannot write before
researching, cannot edit before writing. But once the article is edited and
approved, formatting it for 3 different platforms has **no dependency between
the three outputs**, so `content_factory.py` runs that phase with
`asyncio.gather()`, cutting formatting time by roughly 3x (see Demo 4).
