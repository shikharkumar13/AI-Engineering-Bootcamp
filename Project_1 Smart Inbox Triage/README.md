# Project 1 — Smart Inbox Triage

**Exercises:** Phase 1 (Universal LLM Client) + Phase 2 (Structured Data Extractor)

## What this project does

Takes raw support tickets and, for each one:
1. **Extracts structured fields:** category, priority, sentiment, a summary,
   and concrete action items, using Phase 2's `DataExtractor` (instructor +
   Pydantic), reusing its shared `Priority`/`Sentiment`/`ActionItem` types.
2. **Drafts a reply** grounded in those extracted fields using Phase 1's
   `LLMClient`, with automatic fallback across OpenAI → Claude → Gemini if
   the primary provider fails, and full cost tracking per call.

Nothing here reimplements retry logic, prompt templates, or cost math:
`triage.py` imports `DataExtractor` and `LLMClient` directly from the Phase 1
and Phase 2 folders and composes them.

## Why this pairing

Phase 1 and Phase 2 are usually practiced in isolation: Phase 1 teaches you
to call an LLM reliably, Phase 2 teaches you to get structured output back.
A real feature almost always needs both in the same request path: extract
what you need to make a decision, then generate something conditioned on
that decision, resiliently. This project is the smallest realistic example
of that pattern.

## Project structure
```
Project_1 Smart Inbox Triage/
├── triage.py         ← TicketExtraction model + InboxTriager class
├── demo.py            ← 3 sample tickets, run end-to-end
├── requirements.txt
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python demo.py
```

> Note: Phase 1's `LLMClient` constructs all three provider clients
> (OpenAI, Anthropic, Gemini) on init, even if you only ever call one, so
> all three keys need to be present in `.env` for this project to start,
> not just the one(s) actually used by `chat_with_fallback`.

## API reference

```python
from triage import InboxTriager

triager = InboxTriager()

result = triager.triage(raw_ticket_text)
print(result.extraction.category)      # TicketCategory.BILLING
print(result.extraction.priority)      # Priority.HIGH
print(result.extraction.action_items)  # [ActionItem(task="...", ...), ...]
print(result.draft_reply)              # drafted reply text
print(result.draft_reply_provider)     # "openai", or "claude"/"gemini" if it fell back
print(result.draft_reply_cost_usd)

# Batch, same pattern as Phase 2's batch_extract
results = triager.batch_triage([ticket1, ticket2, ticket3])
triager.report()   # cumulative cost/usage report (Phase 1's UsageStats)
```

## What this exercises from each phase

| From | What's reused |
|---|---|
| Phase 2 | `DataExtractor.extract()`, the Jinja2 extraction prompt templates, `Priority`/`Sentiment`/`ActionItem` Pydantic types |
| Phase 1 | `LLMClient.chat_with_fallback()`, per-call cost tracking, `UsageStats.report()` |

## Adapting this further

Swap `SAMPLE_TICKETS` in `demo.py` for a real ticket source (a CSV export,
an inbox API), and swap the reply-drafting prompt for whatever tone/policy
your use case needs. The extraction schema (`TicketExtraction`) is a good
template for adding your own document-specific fields the same way Phase 2's
`models.py` does.
