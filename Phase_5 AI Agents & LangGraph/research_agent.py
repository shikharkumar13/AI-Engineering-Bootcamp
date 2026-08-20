"""
research_agent.py — AI Research Agent built with LangGraph

Takes a topic, searches the web, reads sources, tracks findings, and produces
a structured research report with citations.

Demonstrates all Phase 05 concepts:
  - ReAct pattern via LangGraph's prebuilt ToolNode, with a custom step-limit-aware
    routing function in place of the prebuilt tools_condition (see _should_continue)
  - Custom tool registration (tools.py)
  - StateGraph with reducers for message accumulation
  - Checkpointer-based memory across turns
  - Human-in-the-loop approval before web fetches
  - Loop detection / step limiting
  - Structured final report generation
"""

from typing import Annotated, TypedDict, Literal
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS, get_notes_store, reset_notes

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME      = "gpt-4o-mini"
MAX_AGENT_STEPS = 12     # safety limit — prevents runaway loops
MAX_REPEAT_CALLS = 3     # same tool+args called this many times → stop


# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


# ── Structured report schema ───────────────────────────────────────────────────

class ResearchReport(BaseModel):
    title:          str = Field(description="A clear, descriptive title for the research")
    summary:        str = Field(description="2-3 sentence executive summary of findings")
    key_findings:   list[str] = Field(description="5-8 specific, factual findings discovered")
    sources:        list[str] = Field(description="URLs or sources referenced during research")
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions that remain unanswered or warrant further research"
    )


# ── System prompt ──────────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """You are a meticulous research assistant. Your job is to \
research the given topic thoroughly using the available tools, then synthesize your \
findings into a clear report.

Process:
1. Search the web for information on the topic (use web_search)
2. For especially relevant results, fetch the full page content (use fetch_page_content)
3. Save every important fact you discover with save_finding, including its source
4. Use calculator for any numeric comparisons or computations needed
5. Use current_date if recency matters for the topic
6. Before concluding, use review_notes to check you have comprehensive coverage
7. Once you have enough well-sourced findings, summarize your research

Guidelines:
- Prioritize accuracy over speed — verify important claims with a second source if possible
- Save findings as you discover them, don't wait until the end
- Aim for at least 5 distinct, well-sourced findings before concluding
- If initial searches don't surface enough information, try different search queries
- When you have sufficient research, say "Research complete" and summarize what you found
"""


# ── Loop guard ──────────────────────────────────────────────────────────────────

@dataclass
class LoopGuard:
    """Detects repeated identical tool calls — a sign the agent is stuck."""
    max_repeated: int = MAX_REPEAT_CALLS
    call_history: list[tuple[str, str]] = field(default_factory=list)

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        key = (tool_name, str(sorted(args.items())))
        self.call_history.append(key)
        count = self.call_history.count(key)
        if count >= self.max_repeated:
            return True, f"'{tool_name}' called {count}x with identical arguments — stopping to avoid a loop"
        return False, ""

    def reset(self):
        self.call_history = []


# ── ResearchAgent class ────────────────────────────────────────────────────────

class ResearchAgent:
    """
    AI Research Agent — searches, reads, tracks findings, produces a report.

    Usage:
        agent = ResearchAgent()
        result = agent.research("Recent breakthroughs in quantum error correction")
        print(result["transcript"])
        report = agent.generate_report("Recent breakthroughs in quantum error correction")
        print(report.model_dump_json(indent=2))
    """

    def __init__(self, model: str = MODEL_NAME, require_fetch_approval: bool = False):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.loop_guard = LoopGuard()
        self.require_fetch_approval = require_fetch_approval

        self._memory = MemorySaver()
        self.graph = self._build_graph()

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        builder.add_edge("tools", "agent")

        interrupt_before = ["tools"] if self.require_fetch_approval else []
        return builder.compile(checkpointer=self._memory, interrupt_before=interrupt_before)

    def _agent_node(self, state: AgentState) -> dict:
        """The reasoning step — decide what to do next."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
        }

    def _tools_node(self, state: AgentState) -> dict:
        """Execute requested tools, with loop guard protection."""
        last_message = state["messages"][-1]

        # Check every requested call for a stuck loop before executing anything.
        stuck_reasons: dict[str, str] = {}
        for tool_call in last_message.tool_calls:
            stuck, reason = self.loop_guard.check(tool_call["name"], tool_call["args"])
            if stuck:
                stuck_reasons[tool_call["id"]] = reason

        if not stuck_reasons:
            return ToolNode(ALL_TOOLS).invoke(state)

        # At least one call is stuck. The model still expects a ToolMessage for
        # *every* tool_call_id in the preceding AIMessage — leaving any of them
        # unanswered makes the next API call to OpenAI fail — so run the
        # non-stuck calls directly and synthesize a message for the stuck ones.
        from langchain_core.messages import ToolMessage
        tools_by_name = {t.name: t for t in ALL_TOOLS}
        messages = []
        for tool_call in last_message.tool_calls:
            if tool_call["id"] in stuck_reasons:
                messages.append(ToolMessage(
                    content=f"LOOP DETECTED: {stuck_reasons[tool_call['id']]}. Please try "
                            f"a different approach or conclude with available information.",
                    tool_call_id=tool_call["id"],
                ))
            else:
                tool_fn = tools_by_name[tool_call["name"]]
                result = tool_fn.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                ))

        return {"messages": messages}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide whether to keep looping or stop."""
        if state.get("step_count", 0) >= MAX_AGENT_STEPS:
            return "end"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    # ── Public interface ─────────────────────────────────────────────────────

    def research(
        self,
        topic: str,
        thread_id: str = "default-research",
        verbose: bool = True,
    ) -> dict:
        """
        Run the agent on a research topic.

        Args:
            topic:     What to research
            thread_id: Conversation thread ID (for memory persistence)
            verbose:   Print each step as it happens

        Returns:
            dict with 'transcript' (list of steps), 'final_message', 'step_count'
        """
        reset_notes()
        self.loop_guard.reset()

        config = {"configurable": {"thread_id": thread_id}}
        transcript = []

        initial_state = {
            "messages": [
                SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=f"Research this topic: {topic}"),
            ],
            "step_count": 0,
        }

        if verbose:
            print(f"\n{'─'*60}\nResearching: {topic}\n{'─'*60}")

        for event in self.graph.stream(initial_state, config=config, stream_mode="values"):
            last_msg = event["messages"][-1]
            step_info = self._summarize_message(last_msg)
            transcript.append(step_info)

            if verbose and step_info["summary"]:
                print(f"  {step_info['type']}: {step_info['summary'][:150]}")

        return {
            "topic": topic,
            "transcript": transcript,
            "final_message": event["messages"][-1].content,
            "step_count": event.get("step_count", 0),
            "notes": get_notes_store().all(),
        }

    def _summarize_message(self, msg: BaseMessage) -> dict:
        """Create a readable summary of a message for the transcript."""
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            calls = ", ".join(
                f"{tc['name']}({', '.join(f'{k}={v!r}' for k,v in tc['args'].items())[:60]})"
                for tc in msg.tool_calls
            )
            return {"type": "AGENT→TOOL", "summary": f"Calling: {calls}"}

        elif msg_type == "ToolMessage":
            return {"type": "TOOL RESULT", "summary": str(msg.content)}

        elif msg_type == "AIMessage":
            return {"type": "AGENT", "summary": msg.content}

        return {"type": msg_type, "summary": str(getattr(msg, "content", ""))}

    def generate_report(self, topic: str, thread_id: str | None = None) -> ResearchReport:
        """
        Run research (if not already done) and synthesize findings into a
        structured report using the saved notes.
        """
        if thread_id is None:
            thread_id = f"report-{topic[:30]}"
            self.research(topic, thread_id=thread_id, verbose=True)

        notes = get_notes_store().all()
        notes_text = "\n".join(
            f"- {n['finding']} (source: {n['source']})" for n in notes
        )

        if not notes_text:
            notes_text = "(No findings were saved during research.)"

        import instructor
        from openai import OpenAI

        client = instructor.from_openai(OpenAI())

        report = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research report writer. Synthesize the "
                               "findings below into a structured, well-organized report."
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nFindings:\n{notes_text}\n\n"
                               f"Write a structured research report based on these findings."
                }
            ],
            response_model=ResearchReport,
            temperature=0,
        )

        return report

    def resume_after_approval(self, thread_id: str) -> dict:
        """Resume execution after a human-in-the-loop interrupt approves the action."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(None, config=config)
        return result

    def get_pending_action(self, thread_id: str) -> dict | None:
        """Check if the graph is paused waiting for approval, and what it wants to do."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)

        if not state.next:  # no pending nodes — not interrupted
            return None

        last_message = state.values["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return {
                "pending_node": state.next,
                "tool_calls": last_message.tool_calls,
            }
        return None
