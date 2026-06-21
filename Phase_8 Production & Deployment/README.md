# Phase 08 — Production AI Service (Capstone)

## What this project does
Takes the RAG concept from Phase 04 and wraps it in everything a real product
needs: a FastAPI backend with auth/rate-limiting/streaming, Docker
containerization, automated RAGAS evaluation, Langfuse observability, and a
Streamlit frontend that talks to the backend over HTTP — not direct function
calls.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Streamlit  │ HTTP │   FastAPI    │      │   RAGService    │
│  (frontend) │ ───► │  (main.py)   │ ───► │ (rag_service.py)│
└─────────────┘      └──────┬───────┘      └─────────────────┘
                             │
                      ┌──────┴───────┐
                      │   Langfuse   │  (observability.py)
                      │   RAGAS      │  (evaluation.py)
                      └──────────────┘

All wired together with Docker + docker-compose
```

## Project structure
```
phase08_project/
├── rag_service.py        ← Core AI logic (lightweight RAG engine)
├── main.py                 ← FastAPI backend (auth, rate limit, streaming, health)
├── observability.py          ← Langfuse tracing wrapper (graceful no-op if unconfigured)
├── evaluation.py               ← RAGAS regression-test suite
├── streamlit_app.py              ← Frontend (calls main.py over HTTP)
├── phase08_demo_client.py           ← Exercises every API endpoint
├── Dockerfile                        ← Backend container
├── Dockerfile.streamlit                ← Frontend container
├── docker-compose.yml                    ← Orchestrates both, with healthchecks
├── phase08_requirements.txt
└── .env.example
```

## Quick start

### Option A — Run locally (no Docker)

```bash
pip install -r phase08_requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum

# Terminal 1: backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: frontend
streamlit run streamlit_app.py

# Terminal 3: exercise every endpoint
python phase08_demo_client.py
```

### Option B — Run with Docker Compose (recommended — this is the actual point of this phase)

```bash
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
docker-compose up --build
```

- Backend: http://localhost:8000 (interactive docs at `/docs`)
- Frontend: http://localhost:8501

### Run the RAGAS evaluation suite

```bash
python evaluation.py
```
First run establishes a baseline. Subsequent runs flag regressions if any
metric drops more than 0.05 below baseline — this is what you'd wire into CI.

## API reference

```bash
# Health & readiness
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Index a document
curl -X POST http://localhost:8000/index \
     -H "X-API-Key: demo-key-123" \
     -F "file=@handbook.pdf"

# Chat (non-streaming, returns sources + trace_id)
curl -X POST http://localhost:8000/chat \
     -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
     -d '{"message": "What is the refund policy?", "k": 4}'

# Chat (streaming)
curl -X POST http://localhost:8000/chat/stream \
     -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
     -d '{"message": "What is the refund policy?"}'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
     -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
     -d '{"trace_id": "...", "is_positive": true}'
```

## Concepts demonstrated

| File | Concept |
|------|---------|
| `main.py` | async endpoints, `StreamingResponse`, API key auth via `Depends()`, `slowapi` rate limiting, exception→HTTP status mapping, liveness vs readiness |
| `rag_service.py` | Self-contained business logic, cleanly separated from the API layer |
| `observability.py` | Langfuse `@observe()` tracing, graceful no-op fallback, feedback recording |
| `evaluation.py` | RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall), baseline comparison, regression detection |
| `Dockerfile` / `Dockerfile.streamlit` | Layer caching order, non-root user, `HEALTHCHECK` |
| `docker-compose.yml` | Multi-service orchestration, `depends_on: condition: service_healthy`, named volumes for persistence |
| `streamlit_app.py` | Frontend calling backend over HTTP (architectural separation), streaming consumption, feedback UI |
| `phase08_demo_client.py` | Exercises auth failures, rate limiting, streaming, and feedback end-to-end |

## What makes this "production" rather than another prototype

1. **It's a service, not a script.** Other systems call it over HTTP; it doesn't
   require a human watching a terminal.
2. **It rejects bad requests properly.** Missing auth gets 401. No docs indexed
   gets 400. Upstream LLM overloaded gets 503. Not a stack trace leaked to the client.
3. **It's reproducible.** `docker-compose up` produces the same environment
   anywhere — no "works on my machine."
4. **It's measurable.** `evaluation.py` gives you a quantitative, regression-
   detecting answer to "did my change make this better or worse?" instead of
   "I tried 3 questions and it seemed fine."
5. **It's observable.** Every request is traced with cost, latency, and
   retrieved sources — and real users can flag bad answers via `/feedback`.

## Adapting this to your own capstone

Swap `rag_service.py` for the agent from Phase 05, the multi-agent crew from
Phase 06, or the fine-tuned model API from Phase 07 — `main.py`,
`observability.py`, the Docker setup, and the evaluation pattern (swap RAGAS
for DeepEval custom metrics if not RAG-specific) all transfer directly. This
is the production shell for any AI service you build going forward.
