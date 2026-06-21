# Phase 01 — Universal LLM Client

## What you built
A production-ready Python library that wraps OpenAI, Claude, and Gemini behind a
single unified interface, with streaming, cost tracking, retries, and async support.

## Project structure
```
phase01_project/
├── llm_client.py    ← the library (this is your project)
├── phase01_demo.py          ← shows all features
├── phase01_requirements.txt
└── .env             ← your API keys (create this!)
```

## Quick start

### 1. Install dependencies
```bash
pip install -r phase01_requirements.txt
```

### 2. Create your .env file
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### 3. Run the demo
```bash
python phase01_demo.py
```

## API reference

```python
from llm_client import LLMClient, Provider

client = LLMClient()

# Basic call (OpenAI by default)
resp = client.chat("What is backpropagation?")
print(resp.text)
print(resp.cost_usd)

# Choose provider
resp = client.chat("...", provider=Provider.CLAUDE)
resp = client.chat("...", provider=Provider.GEMINI)

# Streaming
resp = client.chat("...", stream=True)

# Custom system prompt
resp = client.chat("...", system="You are a Python expert.")

# Fallback: try OpenAI, fall back to Claude if it fails
resp = client.chat_with_fallback("...", providers=[Provider.OPENAI, Provider.CLAUDE])

# Parallel async (5 calls in the time of 1)
import asyncio
results = asyncio.run(client.parallel(["q1", "q2", "q3"]))

# Session stats
client.stats.report()
```

## What each file teaches

| File | Concepts covered |
|------|-----------------|
| `llm_client.py` | All 5 Phase 01 topics integrated |
| `phase01_demo.py` | How to use the client in real scenarios |

## Skills demonstrated (for your portfolio/resume)
- Multi-provider LLM integration (OpenAI, Anthropic, Google)
- Streaming API responses
- Token counting and cost management
- Automatic retry with exponential backoff (tenacity)
- Async/parallel API calls (asyncio)
- Production patterns: fallback, semaphore rate-limiting, session tracking
