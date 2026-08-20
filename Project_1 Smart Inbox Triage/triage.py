"""
Project 1 — Smart Inbox Triage
================================
Combines two things from earlier phases instead of rebuilding them:

  - Phase 2's `DataExtractor` (instructor + Pydantic) to pull structured
    fields out of a raw support ticket: category, priority, sentiment,
    a one-line summary, and concrete action items.
  - Phase 1's `LLMClient` (multi-provider, with fallback and cost tracking)
    to draft a reply based on those extracted fields — falling back from
    OpenAI to Claude to Gemini if the primary provider is unavailable,
    and recording the cost of every call along the way.

Nothing here reimplements retry logic, cost math, or the extraction
prompt templates — those already exist in Phase 1 and Phase 2 and are
imported directly.

Usage:
    from triage import InboxTriager

    triager = InboxTriager()
    result = triager.triage(raw_ticket_text)
    print(result.extraction.category, result.extraction.priority)
    print(result.draft_reply)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

# ── Wire up sibling phase folders so we can import their code directly ────
# (their directory names have spaces, so a normal package import won't
# reach them — this makes the two sibling projects importable as modules)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_2 Prompt Engineering"))
sys.path.insert(0, str(_REPO_ROOT / "Phase_1 LLM APIs"))

from models import ActionItem, Priority, Sentiment          # Phase 2
from extractor import DataExtractor                          # Phase 2
from llm_client import LLMClient, Provider                   # Phase 1


# ── This project's own schema — a support-ticket-shaped sibling of ────────
# Phase 2's EmailData, reusing its shared enums/ActionItem rather than
# redefining them.

class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    COMPLAINT = "complaint"
    GENERAL = "general"


class TicketExtraction(BaseModel):
    summary: str = Field(description="One sentence describing what the customer needs")
    category: TicketCategory
    priority: Priority = Field(
        description="High: outage/billing error/angry customer. "
        "Medium: needs a response but not urgent. Low: informational or minor."
    )
    sentiment: Sentiment
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Concrete follow-up tasks for support/eng to do. Empty if none.",
    )


@dataclass
class TriageResult:
    ticket_text: str
    extraction: TicketExtraction
    draft_reply: str
    draft_reply_provider: str
    draft_reply_cost_usd: float


class InboxTriager:
    """
    Reads a raw support ticket, extracts structured fields (Phase 2), then
    drafts a reply grounded in those fields using a multi-provider client
    with fallback (Phase 1).
    """

    def __init__(self, extraction_model: str = "gpt-4o-mini"):
        self.extractor = DataExtractor(model=extraction_model)
        self.llm = LLMClient()

    def triage(self, ticket_text: str) -> TriageResult:
        extraction = self.extractor.extract(
            ticket_text,
            TicketExtraction,
            extra_instructions=(
                "priority=high only for outages, billing errors, or clearly "
                "angry/urgent customers. action_items should be specific things "
                "a support agent or engineer needs to do, not the customer."
            ),
        )

        reply = self.llm.chat_with_fallback(
            prompt=(
                f"A customer support ticket was categorized as "
                f"'{extraction.category.value}' with priority "
                f"'{extraction.priority.value}'.\n\n"
                f"Ticket summary: {extraction.summary}\n"
                f"Sentiment: {extraction.sentiment.value}\n\n"
                f"Original ticket:\n{ticket_text}\n\n"
                "Draft a short, professional reply (3-5 sentences) that "
                "acknowledges the issue and states the next step. Do not "
                "invent policy details you don't have."
            ),
            system="You are a helpful, concise customer support agent.",
            max_tokens=300,
        )

        return TriageResult(
            ticket_text=ticket_text,
            extraction=extraction,
            draft_reply=reply.text,
            draft_reply_provider=reply.provider,
            draft_reply_cost_usd=reply.cost_usd,
        )

    def batch_triage(self, tickets: List[str]) -> List[TriageResult]:
        results = []
        for i, ticket in enumerate(tickets):
            print(f"  [{i + 1}/{len(tickets)}] triaging...", end=" ")
            try:
                results.append(self.triage(ticket))
                print("done")
            except Exception as e:
                print(f"failed ({type(e).__name__}: {e})")
        return results

    def report(self):
        """Prints cumulative cost/usage across the reply-drafting calls only
        (extraction calls are billed separately by Phase 2's DataExtractor)."""
        self.llm.stats.report()
