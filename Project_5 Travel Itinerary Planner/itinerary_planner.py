"""
itinerary_planner.py — TravelPlanner

Combines Phase 5's ResearchAgent (live web research via a ReAct agent) and
Phase 2's DataExtractor (structured extraction) into one project: research
a destination via live web search, then structure the findings into a
real day-by-day itinerary. Also answers one-off follow-up questions.

Imports both phases' code directly rather than duplicating it.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

# ── Wire up sibling phase folders so we can import their code directly ────
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
        ResearchAgent's checkpointer appends to a thread's message history
        rather than replacing it, so reusing a thread_id across calls would
        silently leak an earlier call's conversation into a new one."""
        self._call_count += 1
        return f"{label}-{self._call_count}"

    # ── Planning ──────────────────────────────────────────────────────
    def plan_trip(
        self, destination: str, num_days: int, interests: list[str] | None = None
    ) -> Itinerary:
        """Researches a destination in two passes (attractions/food, then
        practical logistics) and structures the findings into a day-by-day
        itinerary. Two passes, not one and not num_days: one call risks an
        empty result if the agent's step budget runs out mid-search, and
        one call per day would re-trigger a notes reset on every call and
        multiply cost for no real benefit — see PLAN.md."""
        if not 1 <= num_days <= 5:
            raise ValueError(
                "plan_trip() supports 1-5 day trips (bounded by two research passes per call)."
            )

        interests_clause = f" Focus especially on: {', '.join(interests)}." if interests else ""

        attractions_topic = (
            f"Research attractions, sightseeing, and food/dining recommendations for a "
            f"{num_days}-day trip to {destination}. Find specific named places, not generic "
            f"categories, with enough detail to plan a day-by-day schedule.{interests_clause}"
        )
        attractions_result = self.agent.research(
            attractions_topic,
            thread_id=self._next_thread_id(f"trip-{destination}-attractions"),
            verbose=False,
        )
        # Captured now, before the next research() call resets the shared notes store.
        attractions_notes = attractions_result["notes"]
        attractions_summary = attractions_result["final_message"] or _render_notes(attractions_notes)

        practical_topic = (
            f"Research practical travel tips for a {num_days}-day trip to {destination}: "
            f"getting around/transport, approximate costs, safety notes, and other logistics."
        )
        practical_result = self.agent.research(
            practical_topic,
            thread_id=self._next_thread_id(f"trip-{destination}-practical"),
            verbose=False,
        )
        practical_notes = practical_result["notes"]
        practical_summary = practical_result["final_message"] or _render_notes(practical_notes)

        blob = (
            f"Attractions & food research:\n{attractions_summary}\n\n"
            f"Practical tips research:\n{practical_summary}\n\n"
            f"Research notes:\n{_render_notes(attractions_notes + practical_notes)}"
        )
        return self.extractor.extract(
            blob,
            Itinerary,
            extra_instructions=(
                f"This is travel research for a {num_days}-day trip to {destination}. Organize "
                f"the attractions/food findings into exactly {num_days} days (day_number 1 to "
                f"{num_days}), each with 2-4 activities, spreading different activities across "
                f"different days rather than repeating them. Put logistics/transport/safety "
                f"findings into practical_tips, not into any day's activities. Preserve source "
                f"URLs from the research notes into the sources field where available."
            ),
        )

    # ── Follow-up Q&A ─────────────────────────────────────────────────
    def ask_about_destination(self, destination: str, question: str) -> str:
        """Thin passthrough to ResearchAgent.research() for a one-off
        question, e.g. 'what's the best way to get around in <destination>?'"""
        result = self.agent.research(
            question, thread_id=self._next_thread_id(f"qa-{destination}"), verbose=False
        )
        return result["final_message"] or _render_notes(result["notes"])
