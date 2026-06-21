"""
models.py — Pydantic models for structured data extraction.

Each model represents the expected schema for a specific document type.
Field descriptions are instructions to the LLM about what to extract.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# ── Shared enums ──────────────────────────────────────────────────────────────

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"

class Priority(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"

class WorkType(str, Enum):
    REMOTE  = "remote"
    HYBRID  = "hybrid"
    ONSITE  = "on-site"
    UNKNOWN = "unknown"


# ── Email extraction ───────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str = Field(description="A clear, specific, actionable task (start with a verb)")
    assignee: Optional[str] = Field(
        default=None,
        description="Person responsible for this task, if mentioned"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline for this task in natural language (e.g. 'Friday EOD', 'Oct 28')"
    )

class EmailData(BaseModel):
    sender_name: Optional[str] = Field(
        default=None,
        description="Full name of the email sender. Null if not stated."
    )
    subject_summary: str = Field(
        description="One sentence describing what the email is about"
    )
    intent: str = Field(
        description="Primary intent. One of: request, complaint, update, question, "
                    "approval, rejection, introduction, follow_up"
    )
    sentiment: Sentiment
    priority: Priority = Field(
        description="High: financial impact / time-sensitive / executive. "
                    "Medium: needs response but not urgent. Low: informational."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Specific tasks or follow-ups required. Empty list if none."
    )
    key_points: List[str] = Field(
        description="Top 3-5 key points from the email, each as one sentence"
    )


# ── Job posting extraction ─────────────────────────────────────────────────────

class JobPosting(BaseModel):
    job_title: str = Field(description="Exact job title as listed")
    company: str = Field(description="Company name")
    location: Optional[str] = Field(
        default=None,
        description="City/state/country or 'Remote' or 'Hybrid'"
    )
    work_type: WorkType
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum annual salary in USD as integer. Null if not mentioned."
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum annual salary in USD as integer. Null if not mentioned."
    )
    experience_years: Optional[str] = Field(
        default=None,
        description="Required experience range, e.g. '3-5 years', '2+ years'"
    )
    required_skills: List[str] = Field(
        description="Skills listed as required or must-have. Be specific (e.g. 'Python 3.x', "
                    "not just 'programming')"
    )
    nice_to_have_skills: List[str] = Field(
        default_factory=list,
        description="Skills listed as preferred, bonus, or nice-to-have"
    )
    responsibilities: List[str] = Field(
        description="Key job responsibilities, each as one concise bullet"
    )
    benefits: List[str] = Field(
        default_factory=list,
        description="Benefits or perks mentioned (health insurance, 401k, etc.)"
    )


# ── News article extraction ────────────────────────────────────────────────────

class Entity(BaseModel):
    name: str = Field(description="Full name of the entity")
    type: str = Field(
        description="Entity type: person, organization, location, product, event, or other"
    )
    role: Optional[str] = Field(
        default=None,
        description="Role in the article (e.g. 'CEO of Apple', 'defendant', 'study author')"
    )

class NewsArticle(BaseModel):
    headline: str = Field(
        description="A clear, factual headline capturing the main news event"
    )
    category: str = Field(
        description="Article category: technology, politics, business, science, "
                    "health, sports, entertainment, or world"
    )
    entities: List[Entity] = Field(
        description="Key people, organizations, and locations mentioned. Max 8."
    )
    key_facts: List[str] = Field(
        description="Top 5 specific, verifiable facts from the article (dates, numbers, quotes)"
    )
    sentiment: Sentiment = Field(
        description="Overall tone: positive (good news), negative (bad news/crisis), "
                    "neutral (informational)"
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date in ISO format (YYYY-MM-DD) if mentioned"
    )


# ── Meeting notes extraction ───────────────────────────────────────────────────

class Decision(BaseModel):
    decision: str = Field(description="What was decided, stated clearly")
    rationale: Optional[str] = Field(
        default=None,
        description="Why this decision was made, if explained"
    )

class MeetingNotes(BaseModel):
    meeting_title: str = Field(description="Title or topic of the meeting")
    date: Optional[str] = Field(
        default=None,
        description="Meeting date in ISO format (YYYY-MM-DD)"
    )
    attendees: List[str] = Field(
        default_factory=list,
        description="Names of people who attended the meeting"
    )
    key_decisions: List[Decision] = Field(
        description="Decisions made during the meeting"
    )
    action_items: List[ActionItem] = Field(
        description="Tasks assigned during the meeting, with owner and deadline"
    )
    next_meeting: Optional[str] = Field(
        default=None,
        description="Date/time of next meeting if mentioned"
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Unresolved questions or items deferred to a later date"
    )


# ── Receipt / invoice extraction ───────────────────────────────────────────────

class LineItem(BaseModel):
    name: str = Field(description="Item name or description")
    quantity: Optional[float] = Field(default=None, description="Quantity if specified")
    unit_price: Optional[float] = Field(default=None, description="Price per unit in USD")
    total_price: float = Field(description="Total price for this line item in USD")

class Receipt(BaseModel):
    merchant: str = Field(description="Store or vendor name")
    date: Optional[str] = Field(
        default=None,
        description="Transaction date in ISO format (YYYY-MM-DD)"
    )
    items: List[LineItem] = Field(description="All line items on the receipt")
    subtotal: Optional[float] = Field(default=None, description="Subtotal before tax in USD")
    tax: Optional[float] = Field(default=None, description="Tax amount in USD")
    total: float = Field(description="Final total amount in USD")
    payment_method: Optional[str] = Field(
        default=None,
        description="Payment method: cash, credit, debit, check, or other"
    )
