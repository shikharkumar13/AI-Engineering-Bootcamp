"""
extractor.py — Universal Structured Data Extractor

Demonstrates all Phase 02 concepts:
  - instructor for clean Pydantic-based extraction
  - Jinja2 for dynamic prompt templates
  - Few-shot prompting
  - Chain-of-Thought extraction
  - Raw function calling (for understanding)
  - Auto document-type detection
"""

import json
from typing import TypeVar, Type, Any
from pydantic import BaseModel
from jinja2 import Template
from dotenv import load_dotenv
import instructor
from openai import OpenAI

from models import (
    EmailData, JobPosting, NewsArticle,
    MeetingNotes, Receipt, Sentiment, Priority
)

load_dotenv()

T = TypeVar("T", bound=BaseModel)


# ── Prompt templates ───────────────────────────────────────────────────────────

# Template for standard extraction with optional few-shot examples and instructions
EXTRACTION_TEMPLATE = Template("""
You are a precise data extraction engine.

Instructions:
- Extract ONLY information explicitly present in the text
- Do not infer or assume missing information — use null for absent optional fields
- Field descriptions in the schema are your instructions for what to extract
{% if extra_instructions %}
- {{ extra_instructions }}
{% endif %}

{% if examples %}
Examples of correct extractions:
{% for ex in examples %}
---
Input: "{{ ex.input }}"
Correct extraction: {{ ex.output }}
{% endfor %}
---
{% endif %}

Now extract from this text:

{{ text }}
""".strip())


# Template for Chain-of-Thought extraction (complex documents)
COT_TEMPLATE = Template("""
You are an expert analyst extracting structured information from a document.

First, reason through the document:
1. What type of document is this?
2. Who are the key people/entities involved?
3. What are the main topics and events?
4. What information is explicitly stated vs implied?
5. What important details should not be missed?

Then extract the structured data based on your analysis.

Document:
{{ text }}

Reason through it step by step before extracting.
""".strip())


# Template for document type detection
DETECT_TYPE_TEMPLATE = Template("""
Identify the document type. Output ONLY one of these exact strings:
email, job_posting, news_article, meeting_notes, receipt

Rules:
- email: a message from one person/entity to another
- job_posting: a listing for an open job position
- news_article: a journalistic report on events or topics
- meeting_notes: notes or minutes from a meeting or call
- receipt: a proof of purchase or invoice

Document:
{{ text }}

Document type:""".strip())


# ── DataExtractor class ────────────────────────────────────────────────────────

class DataExtractor:
    """
    Extracts structured data from unstructured text using LLMs.

    Core method:
        extractor.extract(text, OutputModel)  → validated OutputModel instance

    Specialized methods:
        extractor.extract_email(text)
        extractor.extract_job_posting(text)
        extractor.extract_news_article(text)
        extractor.extract_meeting_notes(text)
        extractor.extract_receipt(text)
        extractor.auto_extract(text)           → auto-detects type
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # instructor patches the client to return Pydantic models directly
        self.client = instructor.from_openai(OpenAI())
        # Raw OpenAI client for demonstrations
        self._raw_client = OpenAI()

    # ── Core extraction method ─────────────────────────────────────────────

    def extract(
        self,
        text: str,
        output_model: Type[T],
        extra_instructions: str = "",
        examples: list[dict] | None = None,
        use_cot: bool = False,
        temperature: float = 0,
    ) -> T:
        """
        Extract structured data into a Pydantic model.

        Args:
            text:               The unstructured text to extract from
            output_model:       Pydantic model class to populate
            extra_instructions: Additional field-level instructions
            examples:           Few-shot examples [{"input": "...", "output": "..."}]
            use_cot:            Use Chain-of-Thought reasoning for complex documents
            temperature:        0 for deterministic, higher for variation
        """
        if use_cot:
            prompt = COT_TEMPLATE.render(text=text)
        else:
            prompt = EXTRACTION_TEMPLATE.render(
                text=text,
                extra_instructions=extra_instructions,
                examples=examples or [],
            )

        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=output_model,
            temperature=temperature,
            max_retries=3,  # instructor retries if validation fails
        )

    # ── Specialized extractors ─────────────────────────────────────────────

    def extract_email(self, text: str) -> EmailData:
        """Extract structured data from an email."""
        return self.extract(
            text,
            EmailData,
            extra_instructions=(
                "For action_items: make each task specific and actionable (start with a verb). "
                "For priority: High means financial impact or time < 24h, "
                "Medium means response needed within a week, Low means informational."
            ),
        )

    def extract_job_posting(self, text: str) -> JobPosting:
        """Extract structured data from a job posting."""
        return self.extract(
            text,
            JobPosting,
            extra_instructions=(
                "salary_min and salary_max should be annual USD integers (no currency symbols). "
                "If hourly rate given, multiply by 2080 for annual. "
                "Separate required vs nice-to-have skills carefully."
            ),
        )

    def extract_news_article(self, text: str) -> NewsArticle:
        """Extract structured data from a news article."""
        return self.extract(
            text,
            NewsArticle,
            use_cot=True,  # articles benefit from CoT to catch all entities
            extra_instructions=(
                "key_facts should be specific and verifiable — include numbers, "
                "dates, and direct quotes where available."
            ),
        )

    def extract_meeting_notes(self, text: str) -> MeetingNotes:
        """Extract structured data from meeting notes."""
        return self.extract(
            text,
            MeetingNotes,
            use_cot=True,
        )

    def extract_receipt(self, text: str) -> Receipt:
        """Extract structured data from a receipt or invoice."""
        return self.extract(
            text,
            Receipt,
            extra_instructions=(
                "All monetary values should be floats in USD (no currency symbols). "
                "If a currency symbol is present, strip it."
            ),
        )

    # ── Auto-detect document type ──────────────────────────────────────────

    def auto_extract(self, text: str) -> dict[str, Any]:
        """
        Automatically detect document type, then extract appropriate fields.
        Returns dict with 'document_type' and 'data' (the Pydantic model).
        """
        # Step 1: detect type
        doc_type = self._detect_type(text)

        # Step 2: extract with the right model
        type_map = {
            "email":         (self.extract_email,        EmailData),
            "job_posting":   (self.extract_job_posting,  JobPosting),
            "news_article":  (self.extract_news_article, NewsArticle),
            "meeting_notes": (self.extract_meeting_notes, MeetingNotes),
            "receipt":       (self.extract_receipt,      Receipt),
        }

        if doc_type not in type_map:
            raise ValueError(f"Unknown document type detected: '{doc_type}'")

        extractor_fn, _ = type_map[doc_type]
        extracted_data   = extractor_fn(text)

        return {
            "document_type": doc_type,
            "data":          extracted_data,
        }

    def _detect_type(self, text: str) -> str:
        """Classify document type using zero-shot prompting."""
        prompt = DETECT_TYPE_TEMPLATE.render(text=text[:1500])  # truncate for speed

        response = self._raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().lower()

    # ── Raw function calling demo ──────────────────────────────────────────
    # This shows what instructor does under the hood.
    # In practice, use extract() instead.

    def extract_raw_function_calling(self, text: str) -> dict:
        """
        Demonstrates raw OpenAI function calling — what instructor abstracts away.
        Useful to understand the mechanism before using the library.
        """
        tool_schema = {
            "type": "function",
            "function": {
                "name": "extract_email",
                "description": "Extract structured information from an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_summary": {
                            "type": "string",
                            "description": "One sentence summary of what the email is about"
                        },
                        "intent": {
                            "type": "string",
                            "enum": ["request", "complaint", "update", "question",
                                     "approval", "rejection", "follow_up"],
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "action_items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of action items or tasks",
                        },
                    },
                    "required": ["subject_summary", "intent", "sentiment",
                                 "priority", "action_items"],
                },
            },
        }

        response = self._raw_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract email information."},
                {"role": "user",   "content": text},
            ],
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": "extract_email"}},
            temperature=0,
        )

        # The model generated arguments — parse them from JSON string
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)

    # ── Batch extraction ───────────────────────────────────────────────────

    def batch_extract(
        self,
        texts: list[str],
        output_model: Type[T],
        **kwargs,
    ) -> list[T | Exception]:
        """
        Extract from multiple texts sequentially.
        Failed extractions return the Exception instead of raising.
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.extract(text, output_model, **kwargs)
                results.append(result)
                print(f"  [{i+1}/{len(texts)}] ✓")
            except Exception as e:
                results.append(e)
                print(f"  [{i+1}/{len(texts)}] ✗ {type(e).__name__}: {e}")
        return results
