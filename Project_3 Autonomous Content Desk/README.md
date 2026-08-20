# Project 3 — Autonomous Content Desk

**Exercises:** Phase 6 (Multi-Agent Systems) + Phase 7 (Fine-Tuning & Customization) + Phase 8 (Production & Deployment)

## What this project does

A small FastAPI service with two capabilities:

1. **`POST /generate`:** a topic in, publication-ready blog/LinkedIn/Twitter
   content out, via Phase 6's `ContentFactory` (research → write → edit →
   parallel platform formatting), run behind Phase 8's production shell:
   API-key auth, rate limiting, health/readiness checks.
2. **`POST /support-compare`:** a support message in; runs it through
   Phase 7's fine-tuned support-ticket model (if reachable) *and* a
   general-purpose crew agent, side by side.

Plus `evaluation.py`, a content-quality regression gate that follows Phase
8's own baseline/regression pattern, adapted the way Phase 8's README
itself suggests for non-RAG services ("swap RAGAS for DeepEval custom
metrics"): here that's an LLM-as-judge coherence score per platform plus a
structural length check, instead of RAGAS's retrieval-grounded metrics.

## Why this pairing

`/support-compare` exists to make Phase 7's own decision framework
concrete instead of just stating it: fine-tuning is worth it for a narrow,
structured, repeated task (support tickets forced into a fixed
Issue/Resolution/Next Steps shape); a general-purpose agent is what you'd
reach for on an open-ended task like drafting marketing content. Running
the *same* support message through both and returning them side by side is
that comparison, not just an assertion of it.

Phase 7's model is deliberately **not imported as Python**: `train.py` and
`inference_api.py` depend on CUDA-only packages (`unsloth`, `bitsandbytes`)
that won't install on a non-GPU machine. `crew.py` calls it over HTTP
instead, exactly the way Phase 7's own `demo_client.py` does, and degrades
gracefully (skips that side of the comparison) if `FINETUNED_MODEL_URL`
isn't set or isn't reachable, which is the common case, since it means
spinning up Phase 7's Colab notebook and tunneling it.

## Project structure
```
Project_3 Autonomous Content Desk/
├── crew.py             ← ContentDesk: wraps Phase 6's ContentFactory,
│                          adds the fine-tune-vs-crew comparison
├── main.py              ← FastAPI backend (auth, rate limiting, endpoints)
├── evaluation.py         ← content-quality regression gate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick start

### Option A — Run locally (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, TAVILY_API_KEY at minimum

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B — Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

### Run the evaluation gate

```bash
python evaluation.py
# or: pytest evaluation.py -v
```

## API reference

```bash
curl -X POST http://localhost:8000/generate \
     -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
     -d '{"topic": "The rise of small, efficient language models"}'

curl -X POST http://localhost:8000/support-compare \
     -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
     -d '{"message": "I was charged twice this month, please help."}'
```

`/support-compare`'s response always includes `crew_response`;
`finetuned_response` is `null` and `finetuned_available` is `false` unless
`FINETUNED_MODEL_URL` is configured and Phase 7's `inference_api.py` is
actually running and reachable at that URL.

## What this exercises from each phase

| From | What's reused |
|---|---|
| Phase 6 | `ContentFactory.run()` in full — sequential research/write/edit, parallel platform formatting |
| Phase 7 | The fine-tuned model, called over HTTP exactly like `demo_client.py`; its own "fine-tune vs. general agent" decision framework, made concrete via `/support-compare` |
| Phase 8 | The production-shell pattern: API-key auth, `slowapi` rate limiting, health/readiness endpoints, Docker + docker-compose, and the baseline/regression evaluation-gate structure (metrics swapped for content-appropriate ones) |

## Adapting this further

To actually exercise the fine-tuned side of `/support-compare`: follow
Phase 7's README to train and serve the model in Colab, tunnel it (ngrok or
similar), and set `FINETUNED_MODEL_URL` to that tunnel's address before
starting this service.
