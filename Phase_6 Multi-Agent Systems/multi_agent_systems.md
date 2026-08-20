# Phase 06 — Multi-Agent Systems

> **Prerequisites:** Phases 01-05 complete. You can build a single LangGraph agent with
> tools, memory, and human-in-the-loop approval.  
> **What you'll learn:** CrewAI's agent/task/crew model; specialization patterns;
> sequential vs parallel execution; how context passes between agents; lightweight
> alternatives (OpenAI Swarm, Assistants API).  
> **Project:** A Multi-Agent Content Factory: Researcher → Writer → Editor →
> Publisher, producing blog, LinkedIn, and Twitter content in parallel.

---

## Table of Contents

1. [The Big Picture — Why Multiple Agents?](#1-the-big-picture--why-multiple-agents)
2. [CrewAI Fundamentals — Agents, Tasks, Crews](#2-crewai-fundamentals--agents-tasks-crews)
3. [Agent Specialization Patterns](#3-agent-specialization-patterns)
4. [Sequential vs Parallel Execution](#4-sequential-vs-parallel-execution)
5. [Inter-Agent Context Passing](#5-inter-agent-context-passing)
6. [Lightweight Alternatives — Swarm & Assistants API](#6-lightweight-alternatives--swarm--assistants-api)
7. [Key Takeaways](#7-key-takeaways)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. The Big Picture — Why Multiple Agents?

### 1.1 Where a single agent breaks down

In Phase 05, you built one agent that could search, read, take notes, and report, all
driven by a single system prompt. That works for a focused research task. But consider
a content production pipeline:

```
Topic → Research facts → Write a draft → Edit for clarity and tone
      → Format for blog → Format for LinkedIn → Format for Twitter
```

If you ask a single agent to do all of this in one system prompt, you run into real
problems:

1. **Prompt dilution.** A system prompt trying to be "a researcher AND a writer AND
   an editor AND a social media expert" produces mediocre results at each role. Each
   skill is described less precisely than if it had its own dedicated prompt.

2. **Context bloat.** By the time the agent reaches the "format for Twitter" step,
   its context window is full of research notes, draft iterations, and editing
   feedback, none of which is needed to write a 280-character tweet.

3. **No natural checkpoints.** You cannot easily say "show me the draft before
   formatting it for social media"; it's all one continuous trace.

4. **No specialization-driven quality.** A "Senior Editor" persona genuinely produces
   different (often better) editing feedback than a generalist persona doing editing
   as one of five jobs.

---

### 1.2 The multi-agent paradigm

Multi-agent systems decompose a complex task into specialized roles, each with its own
focused system prompt, its own tools, and a clearly defined input/output contract. A
**crew** (CrewAI's term) or **graph** (LangGraph's term, for multi-agent graphs)
coordinates how these agents pass work to each other.

```
┌─────────────┐      ┌──────────┐      ┌────────┐      ┌────────────┐
│ Researcher  │ ───► │  Writer  │ ───► │ Editor │ ───► │  Publisher  │
│ (find facts)│      │ (draft)  │      │(refine)│      │ (3 formats) │
└─────────────┘      └──────────┘      └────────┘      └────────────┘
```

**From your ML/software background:** This is the same principle as microservices vs a
monolith, or separation of concerns in software architecture. Each agent is a focused
"function" with a clear contract, but the function is implemented by prompting an LLM
with a narrow role instead of writing imperative code.

---

### 1.3 When multi-agent systems are worth the complexity

| Use a single agent | Use multiple agents |
|---|---|
| One coherent task (research, then answer) | Distinct phases needing different "expertise" |
| Tools are all related to one job | Tools are role-specific (search vs writing vs formatting) |
| Output is one artifact | Output needs multiple specialized formats/perspectives |
| Latency must be minimized | Some extra latency for quality is acceptable |
| Debugging by a single trace is fine | You want to inspect/approve each stage separately |

**Important caution, repeated from Phase 05:** Multi-agent systems are slower and more
expensive than single agents (more LLM calls), which are in turn slower than fixed
pipelines. Always start with the simplest architecture that could work, and add agents
only when a single agent's output quality genuinely suffers from doing too much.

---

## 2. CrewAI Fundamentals — Agents, Tasks, Crews

### 2.1 Installation

```bash
pip install crewai crewai-tools
```

CrewAI is built on top of LangChain concepts but provides a higher-level, more
opinionated API specifically designed for multi-agent orchestration. It trades some of
LangGraph's flexibility for much faster setup of common patterns.

---

### 2.2 The Agent

An `Agent` in CrewAI is defined by a **role**, a **goal**, and a **backstory**. These
three fields are not just labels: they are concatenated into the system prompt the LLM
receives, exactly like the persona pattern from Phase 02.

**Note on the `llm=` parameter:** CrewAI is built on LangChain concepts, but `Agent`'s
`llm=` field expects CrewAI's own `LLM` class (or a plain model-name string like
`"gpt-4o-mini"`), not a `langchain_openai.ChatOpenAI` instance: passing a LangChain
chat model directly raises a Pydantic validation error.

```python
from crewai import Agent, LLM

llm = LLM(model="gpt-4o-mini", temperature=0.3)

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover accurate, well-sourced facts on the given topic, "
         "prioritizing recent and credible information",
    backstory=(
        "You are a meticulous research analyst with 10 years of experience "
        "in technology journalism. You are known for never stating a claim "
        "without a source, and for distinguishing between confirmed facts "
        "and speculation."
    ),
    llm=llm,
    verbose=True,           # print the agent's reasoning as it works
    allow_delegation=False, # this agent cannot hand off work to other agents
    max_iter=10,            # max reasoning loops before forcing a conclusion
)
```

**Why role/goal/backstory matters:** Just as in Phase 02's persona pattern, these fields
shape the tone, rigor, and focus of the agent's output. A "Senior Research Analyst" with
a backstory emphasizing rigor will produce noticeably more careful, source-cited output
than a generic "helpful assistant," because the prompt activates a different region of
the model's learned behavior distribution.

```python
writer = Agent(
    role="Content Writer",
    goal="Transform research findings into engaging, clear, well-structured prose",
    backstory=(
        "You are a skilled content writer who has written for major tech "
        "publications. You excel at making complex topics accessible without "
        "oversimplifying them. You write in active voice and avoid jargon "
        "unless it's necessary and explained."
    ),
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Senior Editor",
    goal="Refine drafts for clarity, accuracy, tone consistency, and engagement",
    backstory=(
        "You are a senior editor with a sharp eye for unclear sentences, "
        "unsupported claims, and inconsistent tone. You provide direct, "
        "actionable feedback and aren't afraid to cut unnecessary content."
    ),
    llm=llm,
    verbose=True,
)
```

---

### 2.3 The Task

A `Task` defines a specific unit of work for an agent: a description of what to do, the
expected output format, and which agent is responsible for it.

```python
from crewai import Task

research_task = Task(
    description=(
        "Research the following topic thoroughly: {topic}\n\n"
        "Find at least 5 specific, verifiable facts. For each fact, note "
        "where it came from. Prioritize information from the last 12 months "
        "if the topic is time-sensitive."
    ),
    expected_output=(
        "A structured list of 5-8 facts, each with a one-line source attribution. "
        "Format: '- [Fact]. (Source: [where this came from])'"
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the research findings provided, write a clear, engaging "
        "600-800 word article on: {topic}\n\n"
        "Structure: a compelling hook opening, 3-4 body sections with "
        "subheadings, and a concise conclusion. Cite facts naturally in the prose."
    ),
    expected_output="A complete article draft in markdown format with headers.",
    agent=writer,
    context=[research_task],   # ← this task receives research_task's output as context
)
```

**The `context=[...]` parameter is the core mechanism for passing information between
agents**, covered in depth in Section 5.

---

### 2.4 The Crew

A `Crew` brings agents and tasks together and executes them according to a process
(sequential or hierarchical, covered in Section 4).

```python
from crewai import Crew, Process

content_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,   # tasks run in the order listed
    verbose=True,
)

# kickoff() runs the entire crew and returns the final output
result = content_crew.kickoff(inputs={"topic": "The rise of small language models in 2025"})
print(result.raw)   # the final task's output

# Access individual task outputs
for task_output in result.tasks_output:
    print(f"\n--- {task_output.agent} ---")
    print(task_output.raw[:200])
```

**What happens during `kickoff()`:**
1. CrewAI executes `research_task` using the `researcher` agent
2. The output of `research_task` is automatically injected into `writing_task`'s prompt
   (because `writing_task.context = [research_task]`)
3. `writing_task` executes using the `writer` agent
4. This continues through the task list
5. The final task's output becomes the crew's overall result

---

### 2.5 A complete minimal example

```python
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()
llm = LLM(model="gpt-4o-mini", temperature=0.3)

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, well-sourced facts",
    backstory="An experienced analyst who values accuracy over speed.",
    llm=llm,
)

writer = Agent(
    role="Writer",
    goal="Write clear, engaging content from research",
    backstory="A writer skilled at making technical topics accessible.",
    llm=llm,
)

research_task = Task(
    description="Research key facts about {topic}",
    expected_output="A list of 5 facts with sources",
    agent=researcher,
)

writing_task = Task(
    description="Write a 400-word article about {topic} using the research provided",
    expected_output="A complete article in markdown",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"topic": "quantum computing breakthroughs in 2025"})
print(result.raw)
```

---

## 3. Agent Specialization Patterns

### 3.1 Common specialization roles

A well-designed multi-agent system mirrors how a human team would divide the same work.
Common roles in content/research pipelines:

| Role | Responsibility | Key trait in backstory |
|---|---|---|
| Researcher | Gather facts, verify sources | Skeptical, citation-focused |
| Writer | Produce first draft | Clear, engaging prose |
| Editor | Refine for quality | Critical, detail-oriented |
| Fact-checker | Verify specific claims | Rigorous, evidence-demanding |
| Publisher/Formatter | Adapt to specific channels | Format-aware, audience-aware |
| Reviewer/QA | Final approval gate | Holistic judgment |

---

### 3.2 Designing focused, non-overlapping responsibilities

The most common mistake when building multi-agent systems is creating agents with
overlapping or vague responsibilities, which causes redundant or conflicting work.

```python
# BAD: vague, overlapping responsibilities
bad_writer = Agent(
    role="Writer",
    goal="Write good content",   # too vague
    backstory="You write things.",   # adds no behavioral signal
)

bad_editor = Agent(
    role="Editor",
    goal="Make the content better",   # overlaps with writer's job — what specifically?
    backstory="You edit content.",
)


# GOOD: specific, non-overlapping responsibilities
good_writer = Agent(
    role="Content Writer",
    goal="Transform research into a complete first draft with clear structure. "
         "Focus on getting the ideas down clearly — polish comes later.",
    backstory="You write fast, clear first drafts. You don't worry about "
               "perfecting every sentence; that's the editor's job.",
)

good_editor = Agent(
    role="Senior Editor",
    goal="Improve clarity, fix structural issues, strengthen weak transitions, "
         "and ensure claims are properly supported. Do NOT rewrite from scratch — "
         "refine what exists.",
    backstory="You have a sharp eye for unclear sentences and unsupported claims. "
               "You give specific, actionable edits, not vague feedback.",
)
```

**The key technique:** explicitly state in the goal what this agent should NOT do, in
addition to what it should do. This prevents role bleed where two agents redundantly
perform the same work.

---

### 3.3 Tool specialization

Just as roles should be focused, tool access should be focused. Give each agent only
the tools relevant to its job: this both improves focus and reduces unnecessary cost
(unused tool schemas still consume tokens in every prompt).

```python
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool

search_tool  = SerperDevTool()        # web search
scrape_tool  = ScrapeWebsiteTool()    # fetch and read a page
write_tool   = FileWriterTool()       # save output to a file

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate facts using web search and source verification",
    backstory="...",
    tools=[search_tool, scrape_tool],   # research tools only
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging content from provided research",
    backstory="...",
    tools=[],   # no tools needed — writer works purely from context
    llm=llm,
)

publisher = Agent(
    role="Content Publisher",
    goal="Format and save final content to appropriate files",
    backstory="...",
    tools=[write_tool],   # only the publisher needs file-writing access
    llm=llm,
)
```

---

### 3.4 Custom tools for specialized agents

You can build CrewAI-compatible tools the same way as LangChain tools from Phase 05,
using the `@tool` decorator from `crewai.tools` (or wrap existing LangChain tools).

```python
from crewai.tools import tool

@tool("Character Counter")
def character_counter(text: str) -> str:
    """Count the characters in a piece of text. Useful for checking if content
    fits a platform's character limit (e.g. Twitter's 280 characters)."""
    count = len(text)
    return f"Character count: {count}"


@tool("Hashtag Generator")
def hashtag_generator(topic: str, count: int = 3) -> str:
    """Generate relevant hashtags for a given topic. Use for social media formatting."""
    # In production, this might call an API or use the LLM itself
    words = topic.lower().replace(",", "").split()
    hashtags = ["#" + "".join(w.capitalize() for w in words[:2])]
    hashtags += [f"#{w.capitalize()}" for w in words[:count-1] if len(w) > 3]
    return " ".join(hashtags[:count])


social_media_agent = Agent(
    role="Social Media Specialist",
    goal="Adapt content for Twitter and LinkedIn while respecting platform constraints",
    backstory="You understand each platform's culture: Twitter rewards brevity "
               "and wit; LinkedIn rewards professional insight and depth.",
    tools=[character_counter, hashtag_generator],
    llm=llm,
)
```

---

## 4. Sequential vs Parallel Execution

### 4.1 Sequential process (the default)

In `Process.sequential`, tasks execute strictly in the order they appear in the `tasks`
list. Each task can access the outputs of any previous tasks listed in its `context`.

```python
from crewai import Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,   # research → writing → editing, strictly in order
)
```

```
research_task ──► writing_task ──► editing_task
   (1st)             (2nd)             (3rd)
```

**When to use:** When later steps genuinely depend on earlier steps' output, which is
the most common case (you cannot write before researching, cannot edit before writing).

---

### 4.2 Hierarchical process — a manager agent delegates

`Process.hierarchical` introduces a manager agent that decides which agent should handle
each task dynamically, rather than following a fixed order. This is useful when the
right sequence of work is not known in advance.

```python
from crewai import Crew, Process, LLM

manager_llm = LLM(model="gpt-4o", temperature=0)  # often use a stronger model for managing

crew = Crew(
    agents=[researcher, writer, editor, fact_checker],
    tasks=[content_creation_task],  # a single high-level task
    process=Process.hierarchical,
    manager_llm=manager_llm,        # the manager decides which agent does what, and when
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "AI regulation in the EU"})
```

In hierarchical mode, the manager agent looks at the overall task, decides which
specialist agent should act first, evaluates their output, and decides what to do next,
potentially looping back to an agent multiple times. This is more flexible but also
less predictable and more expensive (the manager itself makes LLM calls to coordinate).

**When to use hierarchical:** Open-ended tasks where the right sequence of specialist
involvement cannot be predetermined, e.g. "produce a high-quality article" where the
manager might decide the draft needs another research pass before editing.

---

### 4.3 True parallel execution

Neither `sequential` nor `hierarchical` natively runs independent tasks in parallel:
they orchestrate one task at a time. For genuinely parallel work (e.g., formatting the
same approved content for 3 different platforms simultaneously, where none depends on
the others), you run separate crew/task executions concurrently using Python's
concurrency tools.

```python
import asyncio
from crewai import Agent, Task, Crew, Process

blog_formatter = Agent(
    role="Blog Formatter",
    goal="Format content as a polished blog post with SEO-friendly headers",
    backstory="An expert in blog formatting and SEO best practices.",
    llm=llm,
)

linkedin_formatter = Agent(
    role="LinkedIn Formatter",
    goal="Format content as an engaging LinkedIn post (1300 char limit, "
         "professional tone, with a thought-provoking question at the end)",
    backstory="An expert in LinkedIn's algorithm and professional audience engagement.",
    llm=llm,
)

twitter_formatter = Agent(
    role="Twitter Thread Formatter",
    goal="Format content as a Twitter/X thread (5-7 tweets, each under 280 "
         "characters, with a strong hook tweet)",
    backstory="An expert in Twitter's fast-paced, hook-driven format.",
    llm=llm,
)

def format_for_platform(formatter_agent: Agent, content: str, platform: str) -> str:
    """Run a single-agent, single-task mini-crew for one platform."""
    task = Task(
        description=f"Format this content for {platform}:\n\n{content}",
        expected_output=f"Content formatted appropriately for {platform}",
        agent=formatter_agent,
    )
    mini_crew = Crew(agents=[formatter_agent], tasks=[task], process=Process.sequential)
    result = mini_crew.kickoff()
    return result.raw


async def format_all_platforms(content: str) -> dict:
    """Run all three formatting crews concurrently using a thread pool,
    since CrewAI's kickoff() is synchronous/blocking."""
    
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(None, format_for_platform, blog_formatter, content, "a blog post"),
        loop.run_in_executor(None, format_for_platform, linkedin_formatter, content, "LinkedIn"),
        loop.run_in_executor(None, format_for_platform, twitter_formatter, content, "Twitter"),
    ]
    
    blog, linkedin, twitter = await asyncio.gather(*tasks)
    
    return {"blog": blog, "linkedin": linkedin, "twitter": twitter}


# Usage
results = asyncio.run(format_all_platforms(edited_article_text))
print(results["blog"][:200])
print(results["linkedin"][:200])
print(results["twitter"][:200])
```

**Why this matters for production:** The research → write → edit phase is inherently
sequential (each step needs the previous step's output). But the final formatting step,
adapting the same approved content for 3 platforms, has no dependency between the
three outputs. Running them in parallel cuts the formatting phase's latency by roughly
3x, exactly like the async patterns from Phase 01.

---

### 4.4 Async kickoff with CrewAI directly

Recent CrewAI versions support `kickoff_async()` natively, which is cleaner than wrapping
in a thread executor:

```python
import asyncio

async def format_all_platforms_native(content: str) -> dict:
    blog_task = Task(
        description=f"Format for blog:\n\n{content}",
        expected_output="Blog-formatted content",
        agent=blog_formatter,
    )
    linkedin_task = Task(
        description=f"Format for LinkedIn:\n\n{content}",
        expected_output="LinkedIn-formatted content",
        agent=linkedin_formatter,
    )
    twitter_task = Task(
        description=f"Format for Twitter:\n\n{content}",
        expected_output="Twitter thread",
        agent=twitter_formatter,
    )

    blog_crew     = Crew(agents=[blog_formatter],     tasks=[blog_task])
    linkedin_crew = Crew(agents=[linkedin_formatter], tasks=[linkedin_task])
    twitter_crew  = Crew(agents=[twitter_formatter],  tasks=[twitter_task])

    blog_result, linkedin_result, twitter_result = await asyncio.gather(
        blog_crew.kickoff_async(),
        linkedin_crew.kickoff_async(),
        twitter_crew.kickoff_async(),
    )

    return {
        "blog": blog_result.raw,
        "linkedin": linkedin_result.raw,
        "twitter": twitter_result.raw,
    }
```

---

## 5. Inter-Agent Context Passing

### 5.1 The `context` parameter — automatic passing

As shown in Section 2.3, the `context=[task1, task2]` parameter on a `Task` automatically
includes the listed tasks' outputs in the new task's prompt. This is the simplest and
most common mechanism.

```python
research_task = Task(
    description="Research {topic}",
    expected_output="5+ facts with sources",
    agent=researcher,
)

writing_task = Task(
    description="Write an article about {topic}",
    expected_output="A complete draft",
    agent=writer,
    context=[research_task],   # writer's prompt automatically includes research_task's output
)

editing_task = Task(
    description="Edit the draft for clarity and accuracy",
    expected_output="A polished final article",
    agent=editor,
    context=[writing_task, research_task],   # editor sees BOTH the draft AND original research
                                              # (useful for fact-checking during editing)
)
```

**What actually happens internally:** CrewAI appends the referenced tasks' output text
into the new task's prompt, formatted as additional context. This is conceptually
identical to the `MessagesPlaceholder` pattern from Phase 03: you are injecting prior
results into a new prompt, but CrewAI automates the bookkeeping of which task's output
goes where.

---

### 5.2 Shared state vs explicit passing

There are two philosophies for inter-agent communication:

**Explicit passing (CrewAI's default approach):** Each task explicitly declares which
prior outputs it needs via `context=[...]`. This is predictable and easy to debug: you
can always trace exactly what information reached each agent.

**Shared state (LangGraph's approach for multi-agent graphs):** All agents read and
write to a common state object. This is more flexible for complex flows with loops and
conditional branching, but requires careful design to avoid agents reading stale or
irrelevant state.

```python
# LangGraph multi-agent state — shared, not explicitly passed per-task
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class ContentPipelineState(TypedDict):
    topic: str
    research_findings: list[str]
    draft: str
    edited_draft: str
    final_outputs: dict[str, str]   # platform -> formatted content

def researcher_node(state: ContentPipelineState) -> dict:
    findings = run_research(state["topic"])
    return {"research_findings": findings}   # written to shared state

def writer_node(state: ContentPipelineState) -> dict:
    # reads research_findings directly from shared state — no explicit passing needed
    draft = write_article(state["topic"], state["research_findings"])
    return {"draft": draft}

def editor_node(state: ContentPipelineState) -> dict:
    edited = edit_article(state["draft"])
    return {"edited_draft": edited}
```

**When to choose which:** For straightforward pipelines (research → write → edit →
publish), CrewAI's explicit `context=[...]` is simpler and sufficiently clear. For
pipelines with loops (e.g., "editor sends draft back to writer if quality is below
threshold"), LangGraph's shared-state graph model with conditional edges is more
natural; you've already learned this pattern in Phase 05.

---

### 5.3 Structured output passing with Pydantic

By default, task outputs are plain text passed as context to the next task. For more
reliable downstream parsing, use `output_pydantic` to get structured output from a task,
exactly like `instructor` from Phase 02.

```python
from pydantic import BaseModel, Field
from crewai import Task

class ResearchFindings(BaseModel):
    facts: list[str] = Field(description="Specific, verifiable facts discovered")
    sources: list[str] = Field(description="Sources for the facts")
    confidence: str = Field(description="high/medium/low confidence in findings")

research_task = Task(
    description="Research {topic} thoroughly",
    expected_output="Structured research findings",
    agent=researcher,
    output_pydantic=ResearchFindings,   # ← enforces structured output
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff(inputs={"topic": "edge AI deployment"})

# Access the structured output of a specific task
research_output = result.tasks_output[0]
findings: ResearchFindings = research_output.pydantic
print(findings.facts)
print(findings.confidence)
```

This combines the reliability benefits of Phase 02's structured extraction with
multi-agent orchestration: downstream agents (and your own code) get validated,
typed data instead of parsing free text.

---

### 5.4 Avoiding context window bloat across many agents

As pipelines grow longer (researcher → writer → editor → fact-checker → publisher),
each subsequent agent's context grows if it receives every prior task's full output.
For long pipelines, be deliberate about what each agent actually needs:

```python
# BAD: every task gets context from every previous task — grows unbounded
fact_check_task = Task(
    description="...",
    agent=fact_checker,
    context=[research_task, writing_task, editing_task],  # editing_task already
                                                            # incorporates writing_task's
                                                            # content — redundant
)

# GOOD: only pass what's actually needed
fact_check_task = Task(
    description="Verify the claims in the edited draft against the original research",
    agent=fact_checker,
    context=[editing_task, research_task],  # need the final draft + original sources,
                                             # not the intermediate writing_task draft
)
```

**Rule of thumb:** Pass the most "downstream" version of any given piece of information
(e.g., the edited draft rather than both the draft and the edited draft), plus any
"source of truth" data (like original research) that should be cross-checked
independently of how content evolved.

---

## 6. Lightweight Alternatives — Swarm & Assistants API

### 6.1 When CrewAI/LangGraph are overkill

CrewAI and LangGraph are full frameworks. For simpler handoff patterns, like "this
agent decides whether to escalate to a different specialized agent," a lighter-weight
approach may be preferable, especially for learning the underlying mechanism or for
projects that need minimal dependencies.

---

### 6.2 OpenAI Swarm — minimal multi-agent handoffs

Swarm (OpenAI's experimental, educational framework, since evolved into the
**Agents SDK**) demonstrates multi-agent handoff with almost no abstraction. Its core
idea: an agent's response can include a "handoff" to a different agent, which then
takes over the conversation.

```python
# pip install openai-agents   (the Agents SDK — Swarm's production successor,
# released March 2025; Swarm itself is archived)
# Conceptual example showing the handoff pattern:

from dataclasses import dataclass
from typing import Callable

@dataclass
class SimpleAgent:
    name: str
    instructions: str
    functions: list[Callable]   # tools, including possible handoff functions


def transfer_to_writer():
    """Call this when research is complete and writing should begin."""
    return writer_agent   # returning an Agent object signals a handoff

def transfer_to_editor():
    """Call this when the draft is ready for editing."""
    return editor_agent


researcher_agent = SimpleAgent(
    name="Researcher",
    instructions="Research the topic. When you have enough facts, call transfer_to_writer.",
    functions=[web_search_tool, transfer_to_writer],
)

writer_agent = SimpleAgent(
    name="Writer",
    instructions="Write a draft from the research. When done, call transfer_to_editor.",
    functions=[transfer_to_editor],
)

editor_agent = SimpleAgent(
    name="Editor",
    instructions="Edit the draft for clarity. This is the final step.",
    functions=[],
)

# A Swarm-style runner loop (simplified):
def run_swarm(starting_agent: SimpleAgent, message: str, max_turns: int = 10):
    current_agent = starting_agent
    history = [{"role": "user", "content": message}]

    for _ in range(max_turns):
        # Call the LLM with current_agent's instructions and available functions
        response = call_llm_with_tools(current_agent, history)
        history.append(response)

        if response.get("handoff_agent"):
            current_agent = response["handoff_agent"]   # switch control
            continue

        if not response.get("tool_calls"):
            return response["content"]   # no more tools/handoffs — done

    return "Max turns reached"
```

**The key insight of the Swarm pattern:** instead of a central orchestrator (CrewAI's
Crew, LangGraph's graph) deciding which agent runs next, **agents decide for themselves**
by calling a special "handoff" function that transfers control. This is a peer-to-peer
model rather than a hub-and-spoke model.

**When to consider this pattern:** Customer service-style routing, where the "current"
agent is the one best suited to handle the conversation right now, and which agent that
is can change dynamically based on what the user asks. For example: a general support
agent that hands off to a billing specialist agent when billing topics come up.

---

### 6.3 OpenAI Assistants API — managed multi-step agents

The Assistants API is OpenAI's managed service for building agents without managing
conversation state, tool execution loops, or memory yourself: OpenAI's servers handle
persistence and orchestration.

```python
from openai import OpenAI

client = OpenAI()

# Create an assistant (this persists on OpenAI's servers — done once, reused)
assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions="You are a research assistant. Use web search to find facts "
                 "and cite your sources.",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}],   # built-in tools: code_interpreter, file_search
)

# Create a thread (a conversation — persists across multiple interactions)
thread = client.beta.threads.create()

# Add a message to the thread
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Research the adoption rate of electric vehicles in 2024.",
)

# Run the assistant on the thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Get the response
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            print(msg.content[0].text.value)
            break
```

**Assistants API vs LangGraph/CrewAI:**

| Aspect | Assistants API | LangGraph / CrewAI |
|---|---|---|
| State management | Managed by OpenAI (threads persist server-side) | You manage it (checkpointers, your own DB) |
| Multi-agent orchestration | Not built-in — you compose multiple assistants yourself | Native support |
| Vendor lock-in | OpenAI only | Provider-agnostic |
| Setup complexity | Low for single-agent use | Higher, but more control |
| Best for | Quick prototypes, OpenAI-only products | Production multi-agent systems, multi-provider |

> **Note on the Assistants API's status:** OpenAI announced in August 2025 that the
> Assistants API is deprecated, with a hard sunset date of **August 26, 2026**. After
> that date, calls like the ones above stop working entirely, not just "not recommended."
> It's being replaced by the **Responses API** combined with the **Conversations API**
> and the **Agents SDK**, which consolidate single-agent and multi-agent patterns into
> one interface. If you're building new production systems on OpenAI's primitives
> directly (rather than LangGraph/CrewAI), use the Responses API, not the code above,
> and check OpenAI's migration guide if you have an existing Assistants integration.

---

### 6.4 Choosing your multi-agent framework

| Framework | Best for | Tradeoff |
|---|---|---|
| **CrewAI** | Role-based pipelines (research→write→edit) | Less flexible for loops/branching |
| **LangGraph** | Complex flows with loops, conditionals, HITL | More code to write |
| **Swarm-style handoff** | Dynamic routing between peer agents | You build the orchestration loop yourself |
| **OpenAI Assistants API** | Quick OpenAI-only prototypes | Vendor lock-in, less orchestration control |

For this phase's project (Researcher → Writer → Editor → Publisher), CrewAI's
sequential process is the natural fit: the roles are clear, the order is fixed, and
the final step parallelizes cleanly.

---

## 7. Key Takeaways

1. **Multi-agent systems trade latency/cost for specialization quality.** A focused
   "Senior Editor" persona produces better editing than a generalist agent doing five
   jobs. Only add agents when this quality gain is worth the added latency and cost.

2. **CrewAI's three primitives: Agent (who), Task (what), Crew (how they're
   orchestrated).** Role/goal/backstory shape behavior exactly like Phase 02's persona
   pattern; write them as specific behavioral instructions, not vague labels.

3. **Specify what an agent should NOT do, not just what it should do.** This is the
   single most effective technique for preventing role overlap between agents.

4. **Sequential process for fixed pipelines; hierarchical for dynamic delegation;
   manual `asyncio.gather` for genuinely independent parallel work.** Most content/
   research pipelines are sequential up to a fan-out point (e.g., final formatting),
   which can then run in parallel.

5. **`context=[task1, task2]` is explicit, traceable context passing.** Be deliberate
   about what each task actually needs: passing every prior task's output to every
   subsequent task causes context bloat without benefit.

6. **`output_pydantic` brings Phase 02's structured extraction into multi-agent
   pipelines.** Use it whenever a downstream agent or your own code needs to parse a
   task's output reliably, rather than relying on free-text passing.

7. **Lighter alternatives exist for simpler needs.** Swarm-style handoff is ideal for
   dynamic peer-to-peer routing (e.g., support ticket routing). The Assistants API
   offloads state management to OpenAI but sacrifices multi-provider flexibility and
   fine-grained orchestration control.

---

## 8. Practice Exercises

### Exercise 1 — Add a Fact-Checker Agent (Easy)
Extend the project's crew with a `fact_checker` agent that runs after the `editor` and
before the `publisher`. Its task should verify that claims in the edited draft are
consistent with the original research findings (give it `context=[editing_task,
research_task]`), and flag any unsupported claims. If it finds issues, the publisher's
task should incorporate the fact-checker's flags into a final disclaimer if needed.

### Exercise 2 — Hierarchical Process Comparison (Medium)
Rebuild the content pipeline using `Process.hierarchical` instead of `Process.sequential`,
with a single high-level task ("Produce a polished article on {topic}") and a manager
LLM coordinating the researcher, writer, and editor agents. Run both versions on the same
topic and compare: total LLM calls, wall-clock time, and output quality.

### Exercise 3 — Structured Multi-Platform Output (Medium-Hard)
Define a Pydantic model `PlatformContent` with fields for `platform`, `content`, and
`character_count`. Modify the three formatter agents (blog/LinkedIn/Twitter) to use
`output_pydantic=PlatformContent`. Write a validator that rejects (and triggers a retry
with feedback) any Twitter output exceeding 280 characters per tweet.

### Exercise 4 — Swarm-Style Dynamic Routing (Hard)
Build a customer support triage system using the Swarm handoff pattern (Section 6.2):
a `Triage` agent that reads an incoming support message and hands off to either a
`BillingAgent`, `TechnicalAgent`, or `GeneralAgent` based on content. Each specialist
agent should be able to hand back to `Triage` if it determines the message was
misrouted. Test with 5 different support messages covering different categories.

---

*Next: Phase 07, Fine-tuning & Customization*  
*You will learn when prompting and RAG aren't enough, and how to adapt open-source
models to your domain using LoRA and QLoRA.*
