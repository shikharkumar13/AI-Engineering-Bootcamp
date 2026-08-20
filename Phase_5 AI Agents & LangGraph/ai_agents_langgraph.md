# Phase 05 — AI Agents & LangGraph

> **Prerequisites:** Phases 01-04 complete. You can call LLMs, engineer prompts, build
> LangChain pipelines, and implement RAG.  
> **What you'll learn:** The ReAct pattern; building and registering tools; LangGraph's
> graph-based agent model; human-in-the-loop workflows; debugging and evaluating agents.  
> **Project:** An AI Research Agent that searches the web, reads sources, and produces a
> structured research report.

---

## Table of Contents

1. [The Big Picture — From RAG to Agents](#1-the-big-picture--from-rag-to-agents)
2. [The ReAct Pattern](#2-the-react-pattern)
3. [Tool Creation & Registration](#3-tool-creation--registration)
4. [LangGraph — Stateful Agent Graphs](#4-langgraph--stateful-agent-graphs)
5. [Human-in-the-Loop Workflows](#5-human-in-the-loop-workflows)
6. [Agent Debugging & Evaluation](#6-agent-debugging--evaluation)
7. [Key Takeaways](#7-key-takeaways)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. The Big Picture — From RAG to Agents

### 1.1 What RAG cannot do

RAG (Phase 04) is a **fixed, single-pass pipeline**: retrieve once, generate once.

```
Question → retrieve chunks → generate answer → done
```

This works when the answer lives entirely in your indexed documents. But consider these
requests:

- "What's the weather in Tokyo right now, and should I pack an umbrella for my trip
  next week?" (needs a live weather API, not a static document)
- "Find the three highest-rated Python web frameworks released in the last year and
  compare their GitHub star counts" (needs web search AND multiple lookups)
- "Debug this error message, try a fix, run the tests, and tell me if it passes"
  (needs code execution and a feedback loop based on the result)
- "Research this topic across multiple sources and write me a report" (needs several
  rounds of searching, reading, and synthesizing)

None of these fit the retrieve-once-generate-once pattern. They require the model to
**decide what action to take next based on what happened previously**, potentially many
times in a row, using different tools, until the task is actually complete.

---

### 1.2 What an agent is

An agent is an LLM wrapped in a loop with access to tools, where the LLM decides:
1. What action to take next (which tool, with what arguments)
2. When it has enough information to produce a final answer

```
┌──────────────────────────────────────────────────────┐
│                                                        │
│   ┌─────────┐    decides action    ┌────────────┐    │
│   │   LLM   │ ───────────────────► │    Tool     │    │
│   │ "brain" │                      │  execution  │    │
│   └────┬────┘                      └──────┬──────┘    │
│        ▲                                  │           │
│        │         observes result          │           │
│        └──────────────────────────────────┘           │
│                                                        │
│   Loop continues until LLM decides task is complete    │
└──────────────────────────────────────────────────────┘
```

This is fundamentally different from Phase 02's function calling, where the model
generated arguments once and your code used them once. Here, the model calls a tool,
sees the result, and decides what to do *next*, potentially calling more tools, in a
loop, until it has enough information.

**From your ML background:** Think of this as analogous to an RL agent's
perceive-decide-act loop, except the "policy" is implemented by prompting an LLM rather
than learned weights, and the "environment" is the set of tools you provide.

---

### 1.3 When to use an agent vs a fixed pipeline

| Use a fixed pipeline (Phase 02-04) | Use an agent |
|---|---|
| The steps are known in advance | The steps depend on what is discovered along the way |
| One or two LLM calls suffice | The task may need a variable, unknown number of steps |
| Latency must be predictable | Some unpredictability in latency is acceptable |
| The task is well-defined extraction/RAG | The task is open-ended research/automation |

**Important caution:** Agents are slower, more expensive, and less predictable than
fixed pipelines. Always ask "could this be a simple chain instead?" before reaching for
an agent. Many production systems that claim to need agents actually need a well-designed
3-step LCEL chain.

---

## 2. The ReAct Pattern

### 2.1 What ReAct is

ReAct (Reasoning + Acting), introduced by Yao et al. (2022), is the foundational pattern
behind nearly all LLM agents. The model alternates between three things, written out
as text:

```
Thought: <reasoning about what to do next>
Action: <tool name>
Action Input: <arguments for the tool>
Observation: <result returned by the tool>

... (repeats as many times as needed) ...

Thought: I now have enough information to answer.
Final Answer: <the answer>
```

This loop is called the **agent scratchpad**. Each Thought/Action/Observation cycle is
appended to the model's context, so the next decision is informed by everything that
happened before.

---

### 2.2 Why writing out "Thought" improves accuracy

This connects directly to Chain-of-Thought from Phase 02. By forcing the model to write
its reasoning before choosing an action, you get:

1. **Better tool selection:** the model reasons about which tool fits before picking one
2. **Debuggability:** you can read the Thought to understand *why* the model did something
3. **Error recovery:** if an Observation reveals a mistake, the next Thought can correct course

```python
# Example of a full ReAct trace for: "What is the population of the capital of France?"

"""
Thought: I need to find the capital of France first, then its population.
Action: web_search
Action Input: "capital of France"
Observation: Paris is the capital and most populous city of France.

Thought: The capital is Paris. Now I need its population.
Action: web_search
Action Input: "Paris population 2024"
Observation: Paris had a population of approximately 2.1 million in the city
proper, and 12.5 million in the metropolitan area, as of 2024.

Thought: I now have enough information to answer.
Final Answer: The capital of France is Paris, which has a population of
approximately 2.1 million in the city proper (12.5 million in the metro area).
"""
```

---

### 2.3 Manual ReAct implementation (understanding the mechanism)

Before using a framework, implement ReAct manually once. This demystifies what LangGraph
and other frameworks automate for you.

```python
from openai import OpenAI
import json, re

client = OpenAI()

# Define available tools as plain Python functions
def web_search(query: str) -> str:
    """Simulated search — in production this calls a real API (see Section 3)."""
    fake_results = {
        "capital of France": "Paris is the capital and most populous city of France.",
        "Paris population 2024": "Paris has approximately 2.1 million residents in the city proper.",
    }
    return fake_results.get(query, f"No results found for '{query}'")

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # In production, use a safe eval library, not raw eval()
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}

REACT_SYSTEM_PROMPT = """You are an assistant that solves tasks step by step using tools.

Available tools:
- web_search(query: str): search the web for information
- calculator(expression: str): evaluate a math expression

Use this EXACT format for each step:

Thought: <your reasoning about what to do next>
Action: <tool name, one of: web_search, calculator>
Action Input: <the input to the tool>

After you receive an Observation, continue with more Thought/Action/Action Input
steps as needed. When you have enough information, respond with:

Thought: I now have enough information to answer.
Final Answer: <your final answer>

Begin!"""


def run_react_agent(question: str, max_steps: int = 5) -> str:
    """
    Manual ReAct loop. This is what LangGraph automates for you.
    """
    scratchpad = f"Question: {question}\n"

    for step in range(max_steps):
        # Ask the model what to do next, given everything so far
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": scratchpad},
            ],
            temperature=0,
            stop=["Observation:"],   # stop generation before the model fakes an observation
        )
        text = response.choices[0].message.content
        print(f"--- Step {step+1} ---\n{text}\n")

        scratchpad += text

        # Check if the model produced a final answer
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # Parse out the action and input
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match  = re.search(r"Action Input:\s*(.+)", text)

        if not action_match or not input_match:
            return "Agent failed to produce a valid action."

        action_name  = action_match.group(1).strip()
        action_input = input_match.group(1).strip().strip('"')

        # Execute the tool
        if action_name in TOOLS:
            observation = TOOLS[action_name](action_input)
        else:
            observation = f"Unknown tool: {action_name}"

        # Append the observation and loop again
        scratchpad += f"\nObservation: {observation}\n"

    return "Max steps reached without a final answer."


# Run it
answer = run_react_agent("What is the population of the capital of France?")
print(f"\nFINAL: {answer}")
```

**What this manual loop demonstrates:**
- The "Action" is just structured text the model generates, the same idea as function
  calling from Phase 02, but here it loops
- Your code is responsible for parsing the action, executing it, and feeding the
  observation back
- The `stop=["Observation:"]` parameter prevents the model from hallucinating fake
  tool results. This is critical: otherwise the model might "imagine" an Observation
  instead of waiting for the real one

---

### 2.4 ReAct with native function calling (more reliable)

Rather than parsing free text, use native function calling (from Phase 02) for the
Action step. This is far more reliable because the model produces structured JSON
instead of text you have to regex out.

```python
import json

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

def run_function_calling_agent(question: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. "
                                       "Use tools as needed to answer accurately."},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",   # model decides whether to call a tool or answer directly
        )
        message = response.choices[0].message
        messages.append(message)

        # If the model didn't call a tool, it's done — return the answer
        if not message.tool_calls:
            return message.content

        # Execute each requested tool call and append results
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  [Step {step+1}] Calling {fn_name}({fn_args})")

            if fn_name in TOOLS:
                result = TOOLS[fn_name](**fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

            # Tool results go back as a "tool" role message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return "Max steps reached."


answer = run_function_calling_agent("What is the population of the capital of France?")
print(answer)
```

**Why this is more reliable than text-parsing ReAct:** The model is constrained by the
function calling schema to produce valid JSON arguments: no regex parsing, no malformed
text, no hallucinated formats. This is the foundation that LangGraph's `ToolNode` uses
internally.

---

## 3. Tool Creation & Registration

### 3.1 The `@tool` decorator

LangChain provides a `@tool` decorator that turns any Python function into an
agent-usable tool. It automatically extracts the function signature and docstring to
build the JSON schema the LLM sees.

```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.
    
    Use this when you need up-to-date facts, news, or information
    not available in your training data.
    """
    # Implementation goes here (see Tavily example below)
    return f"Search results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    
    Use this for any arithmetic, percentages, or numeric calculations.
    Example inputs: "15 * 23", "(100 - 20) / 4", "2 ** 10"
    """
    try:
        import numexpr
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Inspect what the LLM actually sees
print(web_search.name)         # → "web_search"
print(web_search.description)  # → "Search the web for current information..."
print(web_search.args)         # → {'query': {'title': 'Query', 'type': 'string'}}
```

**The docstring is not documentation: it is the tool's description sent to the LLM.**
The model decides whether and how to use a tool based entirely on its name, docstring,
and argument schema. Write docstrings as if you are instructing a new employee on when
to use this specific tool.

---

### 3.2 Tools with Pydantic argument schemas

For tools with multiple or complex arguments, define an explicit Pydantic schema. This
gives you the same validation benefits as Phase 02's `instructor`.

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Tokyo' or 'New York'")
    units: str = Field(
        default="celsius",
        description="Temperature units: 'celsius' or 'fahrenheit'"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    # Call a real weather API here
    return f"Weather in {city}: 22°{'C' if units == 'celsius' else 'F'}, partly cloudy"


print(get_weather.args)
# → {'city': {'description': "City name, e.g. 'Tokyo'...", 'title': 'City', 'type': 'string'},
#    'units': {'default': 'celsius', 'description': '...', 'title': 'Units', 'type': 'string'}}
```

---

### 3.3 Real-world tool: Tavily web search

Tavily is a search API purpose-built for LLM agents: it returns clean, summarized
results instead of raw HTML, which dramatically reduces the tokens needed and improves
result quality.

```bash
pip install tavily-python
```

```python
import os
from tavily import TavilyClient
from langchain_core.tools import tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information. Returns titles, URLs, and
    summaries of the most relevant results.
    
    Use this for: current events, recent data, fact-checking, or any
    information that might not be in your training data.
    """
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",   # "basic" is faster; "advanced" digs deeper
    )

    results = response.get("results", [])
    if not results:
        return f"No results found for: {query}"

    formatted = []
    for r in results:
        formatted.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content'][:300]}"
        )
    return "\n\n---\n\n".join(formatted)


# Test it directly (outside an agent)
result = tavily_search.invoke({"query": "latest developments in LLM agents 2025"})
print(result)
```

**Why a search-specific API instead of raw web scraping?** Tavily, Exa, and similar
services are built for LLM consumption: they return clean text (no ads, navigation,
cookie banners), they rank results for relevance to the query, and they often include
a synthesized answer alongside the raw results. This saves enormous amounts of token
budget compared to feeding the model raw HTML.

---

### 3.4 Tool: fetch and read a web page

```python
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def fetch_page_content(url: str) -> str:
    """Fetch and extract the text content of a specific web page URL.
    
    Use this after web_search to read the full content of a promising result,
    not for general searching (use web_search for that).
    """
    try:
        response = httpx.get(url, timeout=10.0, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style tags — they add noise, not content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid blowing the context window with one page
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        return text

    except httpx.HTTPStatusError as e:
        return f"Failed to fetch page: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Failed to fetch page: {type(e).__name__}: {e}"
```

---

### 3.5 Tool: code execution (sandboxed)

For agents that need to run code (data analysis, calculations, generated scripts),
never use raw `exec()`; execute in an isolated sandbox instead. E2B is a managed
sandbox service designed for this.

```bash
pip install e2b-code-interpreter
```

```python
from e2b_code_interpreter import Sandbox
from langchain_core.tools import tool

@tool
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandbox and return the output.
    
    Use this for calculations, data analysis, or any task requiring code
    execution. The sandbox has common libraries pre-installed (pandas, numpy).
    Print results explicitly — only stdout is captured.
    """
    try:
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code)

            output_parts = []
            if execution.logs.stdout:
                output_parts.append("STDOUT:\n" + "".join(execution.logs.stdout))
            if execution.logs.stderr:
                output_parts.append("STDERR:\n" + "".join(execution.logs.stderr))
            if execution.error:
                output_parts.append(f"ERROR: {execution.error.name}: {execution.error.value}")

            return "\n".join(output_parts) if output_parts else "Code executed with no output."

    except Exception as e:
        return f"Sandbox execution failed: {e}"


# Test
result = execute_python.invoke({
    "code": "import statistics\ndata = [23, 45, 12, 67, 34]\nprint(f'Mean: {statistics.mean(data)}')"
})
print(result)  # → "STDOUT:\nMean: 36.2"
```

**Why sandboxing matters:** If your agent generates code based on LLM output (which can
be influenced by untrusted input, e.g., content scraped from a webpage), executing it
directly on your machine is a critical security risk: prompt injection in fetched
content could instruct the model to generate malicious code. A sandbox isolates execution
from your host system.

---

### 3.6 Binding tools to a model

To let a LangChain chat model use tools, bind them with `.bind_tools()`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [tavily_search, fetch_page_content, calculator]
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("What is the latest version of Python?")

# Check if the model wants to call a tool
print(response.tool_calls)
# → [{'name': 'tavily_search', 'args': {'query': 'latest Python version'}, 'id': 'call_abc123'}]
```

This is the building block LangGraph uses internally, but LangGraph manages the full
loop (calling tools, feeding results back, deciding when to stop) for you.

---

## 4. LangGraph — Stateful Agent Graphs

### 4.1 Why LangGraph instead of manual loops

The manual ReAct loop in Section 2 works, but does not scale to complex agents:
- No built-in way to visualize the flow
- Hard to add conditional branches ("if the search fails, try a different tool")
- Hard to add human approval steps
- Hard to persist state across sessions
- Hard to resume from a specific point after a crash

LangGraph models the agent as a **graph**: nodes are processing steps (LLM calls, tool
execution, custom logic), edges define how control flows between them, and a shared
**state** object is passed between nodes and updated as it goes.

---

### 4.2 Core concepts: State, Nodes, Edges

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# 1. STATE: the data that flows through the graph
class AgentState(TypedDict):
    # add_messages is a "reducer" — it tells LangGraph how to merge updates.
    # Instead of overwriting the messages list, new messages are appended.
    messages: Annotated[list[BaseMessage], add_messages]


# 2. NODES: functions that take state, return a partial state update
def call_model(state: AgentState) -> dict:
    """Node: call the LLM with the current message history."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}   # appended via the add_messages reducer


# 3. Build the graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

# 4. Run it
result = graph.invoke({"messages": [("human", "What is 15% of 240?")]})
print(result["messages"][-1].content)
```

**Why `Annotated[list, add_messages]`?** Without a reducer, each node's return value
would *replace* the state field. With `add_messages` as the reducer, returned messages
are *appended* to the existing list. This is how the conversation history accumulates
across multiple node executions in the graph.

---

### 4.3 Building a full ReAct agent graph

This is the canonical agent pattern in LangGraph: an `agent` node that calls the LLM,
and a `tools` node that executes any requested tool calls, with a conditional edge that
loops between them until the LLM stops requesting tools.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Tools (from Section 3)
tools = [tavily_search, fetch_page_content, calculator]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """The 'brain' — decides what to do next given the conversation so far."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode is a prebuilt LangGraph node that executes whatever tool calls
# are present in the last message — no manual parsing needed
tool_node = ToolNode(tools)


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")

# Conditional edge: tools_condition checks if the last AI message has tool_calls.
# If yes → route to "tools" node. If no → route to END.
builder.add_conditional_edges(
    "agent",
    tools_condition,   # prebuilt function: inspects state, returns "tools" or END
)

# After tools execute, route back to the agent to decide the next step
builder.add_edge("tools", "agent")

graph = builder.compile()
```

```
       START
         │
         ▼
     ┌────────┐
  ┌─►│ agent  │
  │  └───┬────┘
  │      │ tools_condition checks: did the LLM request a tool?
  │      │
  │   ┌──┴───┐
  │   │      │
  │  YES    NO
  │   │      │
  │   ▼      ▼
  │ ┌──────┐ END
  └─┤tools │
    └──────┘
```

This is the same loop as the manual ReAct implementation, but LangGraph:
- Handles state accumulation automatically (the reducer)
- Provides `ToolNode` so you don't write tool execution/parsing code
- Provides `tools_condition` so you don't write the stopping logic
- Gives you a graph you can visualize, modify, and extend

---

### 4.4 Running the agent and inspecting the trace

```python
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="What is the population of Tokyo, and what is "
                                        "that number divided by 1000?")]
})

# Print the full conversation, including tool calls and results
for msg in result["messages"]:
    role = type(msg).__name__
    content = msg.content if msg.content else f"[tool_calls: {msg.tool_calls}]" \
              if hasattr(msg, 'tool_calls') and msg.tool_calls else ""
    print(f"{role}: {content[:150]}")

# Final answer
print(f"\nFinal answer: {result['messages'][-1].content}")
```

**Streaming the graph execution** (see each step as it happens):

```python
for chunk in graph.stream({
    "messages": [HumanMessage(content="What is the latest LangChain version?")]
}, stream_mode="values"):
    last_message = chunk["messages"][-1]
    print(f"[{type(last_message).__name__}] {str(last_message.content)[:100]}")
```

---

### 4.5 Adding custom routing logic

Beyond the standard ReAct loop, LangGraph lets you add custom conditional edges for
more sophisticated control flow, for example retry logic or routing to specialized
sub-agents based on the type of question.

```python
from typing import Literal

def route_by_complexity(state: AgentState) -> Literal["simple_agent", "research_agent"]:
    """
    Custom routing: decide which agent path to use based on the question.
    This is a node that examines state and returns the name of the next node.
    """
    last_message = state["messages"][-1].content.lower()

    research_keywords = ["compare", "research", "analyze", "report", "comprehensive"]
    if any(kw in last_message for kw in research_keywords):
        return "research_agent"
    return "simple_agent"


# Add this as a conditional entry point
builder = StateGraph(AgentState)
builder.add_node("simple_agent", simple_agent_node)
builder.add_node("research_agent", research_agent_node)

builder.add_conditional_edges(
    START,
    route_by_complexity,
    {"simple_agent": "simple_agent", "research_agent": "research_agent"}
)
```

---

### 4.6 Adding memory with checkpointers

LangGraph's `MemorySaver` (or persistent checkpointers for production) automatically
saves the state after each node execution, allowing conversations to persist across
multiple `invoke()` calls, and even across process restarts with persistent backends.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph_with_memory = builder.compile(checkpointer=memory)

# Every invoke() needs a thread_id to identify the conversation
config = {"configurable": {"thread_id": "user-alice-session-1"}}

# Turn 1
result1 = graph_with_memory.invoke(
    {"messages": [HumanMessage(content="What is the capital of Japan?")]},
    config=config,
)
print(result1["messages"][-1].content)

# Turn 2 — the graph remembers Turn 1 because of the same thread_id
result2 = graph_with_memory.invoke(
    {"messages": [HumanMessage(content="What is its population?")]},  # "its" refers to Tokyo
    config=config,
)
print(result2["messages"][-1].content)

# Inspect the full state at any point
state = graph_with_memory.get_state(config)
print(f"Messages in this thread: {len(state.values['messages'])}")
```

For production, replace `MemorySaver` (in-RAM) with a persistent backend:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Persists to a SQLite file — survives process restarts
with SqliteSaver.from_conn_string("agent_memory.db") as memory:
    graph_with_memory = builder.compile(checkpointer=memory)
```

---

## 5. Human-in-the-Loop Workflows

### 5.1 Why agents sometimes need human approval

Fully autonomous agents are risky for high-stakes actions: sending an email, executing
a financial transaction, deleting data, or making an irreversible API call. Human-in-
the-loop (HITL) lets the agent pause before such an action and wait for explicit approval.

---

### 5.2 Interrupting before a tool call

LangGraph supports interrupting graph execution before specific nodes, using the
`interrupt_before` parameter at compile time.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Interrupt the graph BEFORE the "tools" node runs —
# giving a human the chance to review/approve/reject the tool call
graph_with_approval = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)

config = {"configurable": {"thread_id": "approval-demo"}}

# Run until the interrupt point
result = graph_with_approval.invoke(
    {"messages": [HumanMessage(content="Search for the latest AI safety research")]},
    config=config,
)

# Execution paused before the tool call. Inspect what it's about to do:
state = graph_with_approval.get_state(config)
last_message = state.values["messages"][-1]
print("Agent wants to call:")
for tool_call in last_message.tool_calls:
    print(f"  {tool_call['name']}({tool_call['args']})")

# Human decides: approve or reject
user_approval = input("Approve this action? (yes/no): ")

if user_approval.lower() == "yes":
    # Resume execution — pass None to continue from where it stopped
    result = graph_with_approval.invoke(None, config=config)
    print(result["messages"][-1].content)
else:
    print("Action rejected. Agent execution halted.")
```

---

### 5.3 Modifying state during a human review

A more advanced pattern: let the human edit the proposed tool arguments before
continuing, not just approve or reject.

```python
state = graph_with_approval.get_state(config)
last_message = state.values["messages"][-1]

print(f"Agent wants to search for: {last_message.tool_calls[0]['args']['query']}")
new_query = input("Edit the search query (or press Enter to keep it): ").strip()

if new_query:
    # Modify the pending tool call's arguments directly
    last_message.tool_calls[0]['args']['query'] = new_query
    
    # Update the graph's state with the edited message
    graph_with_approval.update_state(config, {"messages": [last_message]})

# Resume with the (possibly edited) action
result = graph_with_approval.invoke(None, config=config)
```

---

### 5.4 Use cases for HITL in production

| Scenario | Why human approval matters |
|---|---|
| Sending an email on the user's behalf | Irreversible, reputational risk |
| Making a purchase or financial transaction | Real money, hard to undo |
| Deleting files or database records | Destructive, irreversible |
| Posting to social media | Public, reputational risk |
| Executing arbitrary code | Security risk |
| Low-confidence classifications | Quality control before downstream action |

**Design principle:** Add HITL checkpoints at the boundary between "the agent gathered
information" and "the agent takes an irreversible action." Read-only actions (search,
fetch, calculate) rarely need approval; write/destructive/financial actions usually do.

---

## 6. Agent Debugging & Evaluation

### 6.1 Why agents are hard to debug

Unlike a fixed pipeline where you can trace exactly what happened, agents make dynamic
decisions. The same question might take a different path through your tools depending
on temperature, model version, or even random variation. You need to observe:

- Which tools were called, in what order, with what arguments
- What each tool returned
- How many steps the agent took before answering
- Whether it got stuck in a loop (calling the same tool repeatedly with similar args)
- The token cost of the full trace (agents can be expensive: many LLM calls per task)

---

### 6.2 LangSmith tracing for agents

Exactly as in Phase 03, enabling LangSmith automatically traces every node execution in
your LangGraph, including the full Thought→Action→Observation sequence.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "research-agent-debug"

# Now every graph.invoke() call is fully traced in LangSmith:
# - Each node execution (agent, tools)
# - Each tool call with its exact arguments
# - Each tool's return value
# - Token usage and latency per LLM call
# - The full message history at each step

result = graph.invoke({"messages": [HumanMessage(content="Research quantum computing breakthroughs in 2025")]})
```

In the LangSmith UI, you can see the entire execution tree, click into any node to see
exactly what was sent and received, and compare runs side-by-side.

---

### 6.3 Adding custom logging to track agent behavior

For lightweight debugging without LangSmith, instrument your nodes directly:

```python
import time

class AgentTracer:
    """Lightweight tracer to track agent steps without external dependencies."""

    def __init__(self):
        self.steps: list[dict] = []

    def log_step(self, node_name: str, input_summary: str, output_summary: str, duration: float):
        self.steps.append({
            "step": len(self.steps) + 1,
            "node": node_name,
            "input": input_summary[:100],
            "output": output_summary[:100],
            "duration_s": round(duration, 2),
        })

    def report(self):
        print(f"\n{'='*60}\nAGENT EXECUTION TRACE ({len(self.steps)} steps)\n{'='*60}")
        total_time = sum(s["duration_s"] for s in self.steps)
        for s in self.steps:
            print(f"  [{s['step']}] {s['node']} ({s['duration_s']}s)")
            print(f"      in:  {s['input']}")
            print(f"      out: {s['output']}")
        print(f"\n  Total steps: {len(self.steps)}, Total time: {total_time:.2f}s")


tracer = AgentTracer()

def traced_agent_node(state: AgentState) -> dict:
    t0 = time.time()
    last_msg = state["messages"][-1].content if state["messages"] else ""
    response = llm_with_tools.invoke(state["messages"])
    tracer.log_step(
        "agent",
        input_summary=str(last_msg),
        output_summary=response.content or f"[calling tools: {[t['name'] for t in response.tool_calls]}]",
        duration=time.time() - t0,
    )
    return {"messages": [response]}
```

---

### 6.4 Detecting infinite loops and runaway agents

A common failure mode: the agent keeps calling the same tool with slightly different
arguments, never converging on an answer. Guard against this with a max-step limit and
loop detection.

```python
class LoopGuard:
    """Detects when an agent is stuck calling the same tool repeatedly."""

    def __init__(self, max_steps: int = 10, max_repeated_calls: int = 3):
        self.max_steps = max_steps
        self.max_repeated_calls = max_repeated_calls
        self.call_history: list[tuple[str, str]] = []  # (tool_name, args_str)

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Returns (should_stop, reason)."""
        self.call_history.append((tool_name, str(args)))

        if len(self.call_history) > self.max_steps:
            return True, f"Exceeded max steps ({self.max_steps})"

        # Count how many times the exact same (tool, args) pair appeared
        current = self.call_history[-1]
        repeat_count = self.call_history.count(current)
        if repeat_count >= self.max_repeated_calls:
            return True, f"Tool '{tool_name}' called {repeat_count}x with identical args — likely stuck"

        return False, ""


guard = LoopGuard(max_steps=10, max_repeated_calls=3)

def guarded_tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        should_stop, reason = guard.check(tool_call["name"], tool_call["args"])
        if should_stop:
            return {"messages": [{"role": "tool", "tool_call_id": tool_call["id"],
                                   "content": f"STOPPED: {reason}"}]}
    # otherwise proceed normally with ToolNode logic
    return tool_node.invoke(state)
```

---

### 6.5 Evaluating agent quality

Evaluating agents is harder than evaluating a single LLM call because correctness
depends on the whole trajectory, not just the final text. Three useful evaluation
dimensions:

**1. Task success rate:** did the agent achieve the goal?
```python
def evaluate_task_success(question: str, expected_answer_contains: list[str]) -> bool:
    """Simple evaluation: does the final answer mention the expected key facts?"""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    final_answer = result["messages"][-1].content.lower()
    return all(kw.lower() in final_answer for kw in expected_answer_contains)

test_cases = [
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is 15% of 200?", ["30"]),
]

passed = sum(evaluate_task_success(q, exp) for q, exp in test_cases)
print(f"Passed: {passed}/{len(test_cases)}")
```

**2. Tool efficiency:** how many steps/tool calls did it take?
```python
def count_tool_calls(result: dict) -> int:
    return sum(
        1 for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

# Fewer tool calls for the same correct answer = more efficient agent
```

**3. Cost per task:** what did this trajectory cost in tokens?
```python
def estimate_trajectory_cost(result: dict, cost_per_1k_input=0.00015, cost_per_1k_output=0.0006) -> float:
    # In production, sum actual usage from each LLM call's response metadata
    total_tokens = sum(
        len(str(msg.content)) // 4   # rough token estimate
        for msg in result["messages"]
    )
    return (total_tokens / 1000) * cost_per_1k_input  # simplified
```

---

## 7. Key Takeaways

1. **Agents extend RAG with a decision loop.** RAG retrieves once and answers. Agents
   decide what to do next, repeatedly, using tools, until the task is complete.

2. **ReAct is Thought → Action → Observation, repeated.** Writing reasoning before
   acting (just like Chain-of-Thought) improves tool selection and gives you a
   debuggable trace of why the agent did what it did.

3. **Function calling is more reliable than text-parsing for actions.** Native tool
   calling (Phase 02) produces structured JSON; regex-parsing free text ReAct output
   is fragile. Always prefer function calling when available.

4. **A tool is a docstring + a function.** The LLM decides whether and how to use a
   tool based entirely on its name, docstring, and argument schema; write these as
   instructions to the model, not just code comments.

5. **LangGraph models agents as graphs: State + Nodes + Edges.** `StateGraph` with a
   reducer (like `add_messages`) accumulates conversation history automatically.
   `ToolNode` and `tools_condition` implement the ReAct loop without custom parsing code.

6. **Checkpointers add memory; `interrupt_before` adds human oversight.** Use
   `MemorySaver` for conversation persistence across calls. Use `interrupt_before`
   to pause before risky actions and require explicit approval.

7. **Agents need different debugging tools than chains.** Track the full trajectory
   (tool calls, arguments, observations), guard against infinite loops with step limits
   and repeat detection, and evaluate based on task success + efficiency + cost, not
   just whether the final text looks reasonable.

---

## 8. Practice Exercises

### Exercise 1 — Custom Tool Set (Easy)
Build three new `@tool`-decorated functions: a `unit_converter` (e.g., miles to km), a
`current_datetime` tool (returns today's date and time), and a `word_count` tool. Bind
them to an LLM and verify the model picks the right tool for different questions.

### Exercise 2 — Manual ReAct vs LangGraph Comparison (Medium)
Implement the same agent (with the same 2 tools) both manually (Section 2.4 style) and
with LangGraph (Section 4.3 style). Run both on 5 identical questions. Compare: lines of
code, ease of adding a third tool, and ease of adding memory.

### Exercise 3 — Approval-Gated Research Agent (Medium-Hard)
Extend the project's research agent with `interrupt_before=["tools"]` specifically for
the `fetch_page_content` tool (but not `tavily_search`, which is safe/read-only at the
search level but you want to control which pages get fully fetched and consume tokens).
Print the proposed URL before fetching and require user confirmation.

### Exercise 4 — Loop Detection and Recovery (Hard)
Extend `LoopGuard` so that when it detects a stuck agent, instead of just stopping, it
injects a system message telling the LLM "You have called this tool with identical
arguments multiple times. Try a different approach or provide your best answer with
the information you have." Test this recovery behavior with a deliberately ambiguous
question that might confuse a naive agent.

---

*Next: Phase 06, Multi-Agent Systems*  
*You will orchestrate teams of specialized agents (researcher, writer, editor) that
collaborate on tasks too complex for a single agent to handle well alone.*
