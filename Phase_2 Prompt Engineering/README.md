# Phase 02 — Structured Data Extractor

## What this project does
A universal pipeline that reads any unstructured text (emails, job postings,
news articles, meeting notes, receipts) and returns clean, validated JSON using
LLMs + Pydantic + instructor.

## Project structure
```
Phase_2 Prompt Engineering/
├── models.py      ← Pydantic models for each document type
├── extractor.py   ← DataExtractor class (the library)
├── demo.py        ← All demos
├── requirements.txt
└── .env.example   ← copy to .env and fill in your API key
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
python demo.py
```

## API reference

```python
from extractor import DataExtractor
from models import EmailData, JobPosting, NewsArticle, MeetingNotes

extractor = DataExtractor()

# Typed extractors
email_data   = extractor.extract_email(email_text)
job_data     = extractor.extract_job_posting(job_text)
article_data = extractor.extract_news_article(article_text)
notes_data   = extractor.extract_meeting_notes(notes_text)
receipt_data = extractor.extract_receipt(receipt_text)

# Access fields as Python objects (not dicts)
print(email_data.priority)      # Priority.HIGH
print(email_data.action_items)  # [ActionItem(task="...", deadline="...")]
print(email_data.model_dump())  # → dict
print(email_data.model_dump_json(indent=2))  # → JSON string

# Auto-detect document type
result = extractor.auto_extract(any_text)
print(result["document_type"])  # "email", "job_posting", etc.
print(result["data"])           # the appropriate Pydantic model

# Generic extraction with any Pydantic model
from pydantic import BaseModel
class MyModel(BaseModel):
    field_a: str
    field_b: int

result = extractor.extract(text, MyModel)

# Batch processing
results = extractor.batch_extract([text1, text2, text3], EmailData)
```

## Concepts demonstrated

| Demo | Concept |
|------|---------|
| Demo 1 | Zero-shot, few-shot, CoT, self-consistency |
| Demo 2 | Email extraction with nested Pydantic models |
| Demo 3 | Job posting with type coercion and enums |
| Demo 4 | News article with Chain-of-Thought |
| Demo 5 | Meeting notes with complex nested structures |
| Demo 6 | Receipt extraction with numeric line items |
| Demo 7 | Auto document-type detection |
| Demo 8 | Raw function calling vs instructor comparison |
| Demo 9 | Batch extraction with error handling |
