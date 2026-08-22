# Project 5 — Travel Itinerary Planner

## Context

The fourth Practice Project (Recipe & Meal Planner) combined a RAG phase with
a structured-extraction phase. This one is a step up in difficulty on
purpose, per the user's own framing when this project was first sketched:
"the only one here that touches agents, so it's a notch harder than the
other Practice Projects." It combines **Phase 5** (`ResearchAgent`, a
LangGraph ReAct agent with live web search) and **Phase 2** (`DataExtractor`,
structured extraction), the same "compose two phase classes" shape as every
other Practice Project, imported directly rather than duplicated.

Concept: give it a destination and a number of days, it researches
attractions/food/practical tips via live web search, then structures the
findings into a real day-by-day itinerary. It can also answer one-off
follow-up questions about the destination.

Following Project 1/2/4's precedent: **script only** (core module + `demo.py`,
no FastAPI/Docker/Streamlit), no `evaluation.py` (only Phase 8 and Project 3
have eval gates). This wasn't re-confirmed with the user this round since it
matches the pattern chosen for every prior project — flagged here so it's
visible before implementation starts.

## Verified API surface (re-read from the actual code, not memory)

```python
# Phase 5 research_agent.py
class ResearchAgent:
    def __init__(self, model: str = MODEL_NAME, require_fetch_approval: bool = False): ...
    def research(self, topic: str, thread_id: str = "default-research", verbose: bool = True) -> dict:
        # returns {"topic", "transcript", "final_message", "step_count", "notes"}
        # notes = [{"finding": str, "source": str}, ...]
    def generate_report(self, topic: str, thread_id: str | None = None) -> ResearchReport: ...
    # NOT used by this project — if thread_id is None it silently runs research() a
    # second time internally. Avoided entirely by design.

# Phase 2 extractor.py
class DataExtractor:
    def __init__(self, model: str = "gpt-4o-mini"): ...
    def extract(self, text: str, output_model: Type[T], extra_instructions: str = "",
                examples=None, use_cot: bool = False, temperature: float = 0) -> T: ...
```

`MAX_AGENT_STEPS = 12` hard-caps every `.research()` call. `ResearchNotes` is
a **module-level singleton** — `reset_notes()` runs at the start of every
single `.research()` call, so notes from an earlier call are gone the
moment the next call starts (must be captured into a local variable
immediately after each call returns). `MemorySaver` is in-memory,
process-lifetime only, keyed by `thread_id`.

## Three real gotchas found while pressure-testing this design (and how they're handled)

1. **`thread_id` collision causes silent state bleed across calls.**
   `AgentState.messages` uses `Annotated[list[BaseMessage], add_messages]` —
   a reducer that *appends* to a thread's existing checkpoint rather than
   replacing it. Confirmed directly against Phase 5's own `demo.py` (Demo 4,
   "Memory Across Multiple Turns"), which deliberately reuses a `thread_id`
   to prove turn 2 remembers turn 1. If `TravelPlanner.plan_trip()` reused a
   fixed `thread_id` pattern like `f"trip-{destination}-{num_days}d"`, a
   second call for the same destination+day-count would silently inherit
   the first call's full conversation history instead of starting fresh —
   not a crash, a quiet correctness bug. **Fix:** an instance-level call
   counter makes every `thread_id` unique per call, no matter how many times
   `plan_trip()`/`ask_about_destination()` are called on the same
   `TravelPlanner`.

2. **One `.research()` call per trip risks an empty `final_message`.**
   `_should_continue` checks `step_count >= MAX_AGENT_STEPS` *before*
   checking whether the last message still has pending tool calls — so a
   call cut off mid-loop at step 12 can return an `AIMessage` with
   `content=""` (tool-calling messages usually carry no text) as
   `final_message`. Asking one call to cover attractions + food + practical
   tips for up to 5 days in a single 12-step budget makes this a real risk,
   not hypothetical. **Fix:** split into **two** `.research()` calls per
   trip — attractions/food, then practical/logistics — each getting its own
   full step budget, plus a same-call fallback: if `final_message` comes
   back empty, fall back to the joined `notes` findings for that call
   instead of feeding empty text to the extractor.

3. **Two calls, not `num_days` calls.** Splitting by day (one `.research()`
   per day) would re-trigger the notes-singleton-reset gotcha per day,
   requires building day-to-day dedup logic so day 2 doesn't repeat day 1's
   suggestions, and multiplies cost/latency by up to 5x — disproportionate
   complexity for an "intermediate, a notch harder" project (that's
   Project 3's multi-agent territory, not this one's). Two calls, with
   `DataExtractor.extract()` doing the "organize into day_number 1..N" work
   as a pure text-reorganization task, is cheaper, more reliable, and still
   honestly demonstrates the singleton-capture pattern once per call.

## Other design decisions

- **`num_days` is capped at 1-5**, not higher — bounded by what two research
  passes can realistically cover in reasonable depth. Documented in the
  README, not silently degraded.
- **No separate `models.py`** — inline `Activity`/`DayPlan`/`Itinerary` into
  `itinerary_planner.py`, matching Project 1 and Project 4's precedent.
- **No sample data folder** — unlike Project 4's `sample_recipes/`, there's
  nothing local to ingest; input is just a destination string and a day
  count.
- **`.env.example` needs both `OPENAI_API_KEY` and `TAVILY_API_KEY`**,
  matching Project 2's precedent — without Tavily, `web_search` returns a
  literal error string to the LLM and the whole premise (live travel
  research) falls apart, not just degrades gracefully the way Project 4's
  unused-key omissions did.
- **`requirements.txt` is the literal union of Phase 5's and Phase 2's own
  `requirements.txt`**, verified by reading both files directly, no pruning
  to only-what's-imported (matches every prior project's convention, e.g.
  Project 4 also carries Phase 2's unused `anthropic`/`tiktoken` verbatim).

## Files to create

```
Project_5 Travel Itinerary Planner/
├── README.md
├── itinerary_planner.py   ← TravelPlanner + Activity/DayPlan/Itinerary (Pydantic)
├── demo.py                 ← single main(), calls plan_trip() twice to prove
│                               the thread_id fix, plus one ask_about_destination()
├── requirements.txt        ← union of Phase 5 + Phase 2's requirements
└── .env.example            ← OPENAI_API_KEY + TAVILY_API_KEY
```

Plus edits to the root `README.md` (Repository Structure tree, Practice
Projects table, Progress checklist, "four" → "five") and `CLAUDE.md`
(Project list, Phase 2/Phase 5 architecture-note bullets).

## `itinerary_planner.py` — method-by-method

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_5 AI Agents & LangGraph"))
sys.path.insert(0, str(_REPO_ROOT / "Phase_2 Prompt Engineering"))
from research_agent import ResearchAgent   # Phase 5
from extractor import DataExtractor          # Phase 2

# ── This project's own schema ──────────────────────────────────────────
class Activity(BaseModel):
    name: str = Field(description="Specific named place or activity, not a generic category")
    description: str
    category: str = Field(description="e.g. 'sightseeing', 'food', 'practical'")
    estimated_time: str | None = Field(None, description="e.g. '2 hours', 'morning'")

class DayPlan(BaseModel):
    day_number: int
    theme: str | None = Field(None, description="A short theme for the day, e.g. 'Old Town & History'")
    activities: list[Activity]

class Itinerary(BaseModel):
    destination: str
    num_days: int
    days: list[DayPlan]
    practical_tips: list[str] = []
    sources: list[str] = []

def _render_notes(notes: list[dict]) -> str:
    return "\n".join(f"- {n['finding']} (source: {n['source']})" for n in notes)

class TravelPlanner:
    def __init__(self):
        self.agent = ResearchAgent()
        self.extractor = DataExtractor()
        self._call_count = 0

    def _next_thread_id(self, label: str) -> str:
        """Every agent call gets a unique thread_id, scoped to this instance.
        MemorySaver appends to a thread's history rather than replacing it,
        so reusing a thread_id across calls would silently leak the earlier
        call's conversation into the new one."""
        self._call_count += 1
        return f"{label}-{self._call_count}"

    # ── Planning ──────────────────────────────────────────────────────
    def plan_trip(self, destination: str, num_days: int, interests: list[str] | None = None) -> Itinerary:
        if not 1 <= num_days <= 5:
            raise ValueError("plan_trip() supports 1-5 day trips (bounded by two research passes per call).")

        interests_clause = f" Focus especially on: {', '.join(interests)}." if interests else ""

        attractions_topic = (
            f"Research attractions, sightseeing, and food/dining recommendations for a "
            f"{num_days}-day trip to {destination}. Find specific named places, not generic "
            f"categories, with enough detail to plan a day-by-day schedule.{interests_clause}"
        )
        attractions_result = self.agent.research(
            attractions_topic, thread_id=self._next_thread_id(f"trip-{destination}-attractions"), verbose=False
        )
        attractions_notes = attractions_result["notes"]  # captured now, before the next call resets the singleton
        attractions_summary = attractions_result["final_message"] or _render_notes(attractions_notes)

        practical_topic = (
            f"Research practical travel tips for a {num_days}-day trip to {destination}: "
            f"getting around/transport, approximate costs, safety notes, and other logistics."
        )
        practical_result = self.agent.research(
            practical_topic, thread_id=self._next_thread_id(f"trip-{destination}-practical"), verbose=False
        )
        practical_notes = practical_result["notes"]
        practical_summary = practical_result["final_message"] or _render_notes(practical_notes)

        blob = (
            f"Attractions & food research:\n{attractions_summary}\n\n"
            f"Practical tips research:\n{practical_summary}\n\n"
            f"Research notes:\n{_render_notes(attractions_notes + practical_notes)}"
        )
        return self.extractor.extract(
            blob, Itinerary,
            extra_instructions=(
                f"This is travel research for a {num_days}-day trip to {destination}. Organize the "
                f"attractions/food findings into exactly {num_days} days (day_number 1 to {num_days}), "
                f"each with 2-4 activities, spreading different activities across different days "
                f"rather than repeating them. Put logistics/transport/safety findings into "
                f"practical_tips, not into any day's activities. Preserve source URLs from the "
                f"research notes into the sources field where available."
            ),
        )

    # ── Follow-up Q&A ─────────────────────────────────────────────────
    def ask_about_destination(self, destination: str, question: str) -> str:
        """Thin passthrough to ResearchAgent.research() for a one-off
        question, e.g. 'what's the best way to get around in <destination>?'"""
        result = self.agent.research(question, thread_id=self._next_thread_id(f"qa-{destination}"), verbose=False)
        return result["final_message"] or _render_notes(result["notes"])
```

## Build order

1. **`itinerary_planner.py`** first — everything else depends on it,
   especially the two-call split and the thread_id counter. sys.path order:
   Phase 5 before Phase 2 (matches the descending-phase-number convention
   Project 2 and Project 4 both already use). Run
   `python -m py_compile itinerary_planner.py` as soon as it parses.
2. **`requirements.txt` + `.env.example`** — needed immediately after to
   actually exercise the module.
3. **`demo.py`** — single `main()`. Deliberately calls `plan_trip()` twice
   with overlapping destination/day-count (e.g. the same trip twice, or two
   similar trips) specifically to prove the thread_id fix works, not just
   to show a single happy path — this is the cheapest regression check
   available for gotcha #1. Also one `ask_about_destination()` call.
4. **Verify**: `python -m py_compile` on both files, then check what's
   possible without a live run (dependency install, construction) the same
   way Project 4 was verified when no API keys were available, and ask the
   user the same question Project 4's build did about live verification
   before writing the README's example output.
5. **`README.md`** last — same section structure as Project 4's (What this
   project does / Why this pairing / Project structure / Quick start / API
   reference / What this exercises from each phase / Known limitations
   worth knowing / Adapting this further), with the `num_days` cap, the
   TAVILY_API_KEY requirement, and the possible-empty-research-pass edge
   case named as limitations, not hidden.
6. **Root `README.md`** — insert `Project_5 Travel Itinerary Planner/` into
   the Repository Structure tree, add its row to the Practice Projects
   table, "these four" → "these five," update the Progress checklist.
7. **`CLAUDE.md`** — add Project 5 to the Practice Projects list/intro,
   append a clause to the existing Phase 2 and Phase 5 architecture-note
   bullets.

## Verification

- `python -m py_compile itinerary_planner.py demo.py` as each is written.
- If a real `OPENAI_API_KEY`/`TAVILY_API_KEY` are available: run
  `python demo.py` end-to-end and confirm both `plan_trip()` calls produce
  a real `Itinerary` (not an empty-activities result from the
  empty-`final_message` edge case), the day count matches `num_days`, and
  the second `plan_trip()` call for an overlapping destination does *not*
  show any bleed-through from the first call's research (the thread_id fix
  actually working, not just present in the code).
- If no real keys are available: verify what's possible without live LLM
  calls (construction, `python -m py_compile`) and say so plainly rather
  than claiming a full run happened.
