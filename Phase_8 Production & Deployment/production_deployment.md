# Phase 08 — Production & Deployment

> **Prerequisites:** Phases 01–07 complete. You can build LLM apps, RAG systems,
> agents, and fine-tune models.  
> **What you'll learn:** FastAPI backends for LLM apps; rapid UIs with Streamlit/Gradio;
> Docker containerization; automated evaluation with RAGAS/DeepEval; observability
> with Langfuse and LangSmith.  
> **Project:** A containerized, evaluated, observable AI service — wrapping the RAG
> engine from Phase 04 in a production-grade deployment.

---

## Table of Contents

1. [The Big Picture — The Prototype-to-Production Gap](#1-the-big-picture--the-prototype-to-production-gap)
2. [FastAPI LLM Backends](#2-fastapi-llm-backends)
3. [Streamlit & Gradio — Rapid UIs](#3-streamlit--gradio--rapid-uis)
4. [Docker & Containerization](#4-docker--containerization)
5. [LLM Evaluation — RAGAS & DeepEval](#5-llm-evaluation--ragas--deepeval)
6. [Observability — Langfuse & LangSmith](#6-observability--langfuse--langsmith)
7. [Key Takeaways](#7-key-takeaways)
8. [Practice Exercises](#8-practice-exercises)

---

## 1. The Big Picture — The Prototype-to-Production Gap

### 1.1 What "it works on my machine" is missing

Every phase so far has produced working code — a RAG pipeline, an agent, a fine-tuned
model. But "works when I run the Python script" and "works as a product other people
rely on" are very different bars. The gap consists of:

| Prototype | Production |
|---|---|
| Runs in your terminal | Runs as a service other systems call over HTTP |
| Single user (you) | Many concurrent users |
| You watch the output | No one is watching — you need logs/traces |
| If it breaks, you notice and fix it | If it breaks, you need to be alerted automatically |
| "It seemed to work in my tests" | Quantitative evaluation on a held-out test set |
| Runs on your laptop with your Python version | Runs identically on any machine, any time |
| API keys in a `.env` file you trust | Secrets management, access control |

This phase closes that gap using four pieces: a real API server (FastAPI), a
container that makes deployment reproducible (Docker), an automated way to know if a
change made quality better or worse (RAGAS/DeepEval), and visibility into what's
happening in production (Langfuse/LangSmith).

---

### 1.2 The production stack for this phase

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Streamlit  │ ───► │   FastAPI    │ ───► │  Your AI Logic  │
│  (frontend) │ HTTP │  (backend)   │      │ (RAG/Agent/etc) │
└─────────────┘      └──────┬───────┘      └────────┬────────┘
                             │                       │
                             ▼                       ▼
                      ┌─────────────┐        ┌──────────────┐
                      │  Langfuse   │        │    RAGAS     │
                      │ (traces in  │        │ (evaluation, │
                      │  production)│        │  pre-deploy) │
                      └─────────────┘        └──────────────┘

         Everything packaged in Docker containers,
         orchestrated with docker-compose
```

---

## 2. FastAPI LLM Backends

### 2.1 Why FastAPI specifically

FastAPI is the dominant choice for LLM backends in Python for three concrete reasons:
native async support (essential for I/O-bound LLM calls, see Phase 01), automatic
request/response validation via Pydantic (the same library you've used since Phase 02),
and automatic interactive API documentation generated from your code.

```bash
pip install fastapi uvicorn[standard]
```

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="My AI Service")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # request.message is already validated as a string by Pydantic —
    # if the client sends malformed JSON or wrong types, FastAPI
    # automatically returns a 422 error before your code even runs
    answer = await call_your_llm_logic(request.message)
    return ChatResponse(response=answer)

# Run with: uvicorn main:app --reload
# Interactive docs automatically available at http://localhost:8000/docs
```

**Why `async def`:** Every LLM call is I/O-bound (Phase 01, Section 7) — your server
process spends most of its time waiting for the LLM provider's response, not using CPU.
With `async def` and `await`, FastAPI can handle other incoming requests while waiting
for a slow LLM call to return, dramatically increasing the number of concurrent users
one server process can serve. A synchronous (`def` instead of `async def`) endpoint
blocks the entire worker while waiting.

---

### 2.2 Streaming responses

Just as in Phase 01 and Phase 03, streaming dramatically improves perceived latency.
FastAPI supports this via `StreamingResponse`.

```python
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_tokens(message: str):
    """An async generator — yields chunks as they arrive from the LLM."""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta   # each yield sends a chunk to the client immediately

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_tokens(request.message),
        media_type="text/event-stream",   # Server-Sent Events (SSE) content type
    )
```

**Consuming a streaming endpoint from a client:**

```python
import httpx

async def consume_stream(message: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", "http://localhost:8000/chat/stream",
            json={"message": message},
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
```

---

### 2.3 Authentication — API keys

Production APIs need to control who can call them. The simplest robust pattern is an
API key checked via a FastAPI dependency.

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import os

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = set(os.getenv("VALID_API_KEYS", "").split(","))

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    A FastAPI dependency — runs before the endpoint function, can reject
    the request before any of your business logic executes.
    """
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),   # ← runs automatically before chat()
):
    answer = await call_your_llm_logic(request.message)
    return ChatResponse(response=answer)
```

**Why a `Depends()` function instead of checking inside the endpoint:** Dependencies
compose cleanly across many endpoints — write `verify_api_key` once, apply it to every
protected route. They also run before the request body is fully processed in some cases,
rejecting unauthorized requests as early as possible, and they're independently testable.

For production systems with per-user rate limits or usage tracking, replace the simple
set-membership check with a database lookup that also returns the user's plan/quota.

---

### 2.4 Rate limiting

LLM API calls cost money per request. Without rate limiting, a single misbehaving
client (or a bug in your own frontend retrying in a loop) can generate a large bill
very quickly.

```python
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)   # rate limit per client IP
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")   # max 10 requests per minute per IP
async def chat(request: Request, chat_request: ChatRequest):
    # Note: the route function needs a `request: Request` parameter
    # for slowapi to inspect when rate-limiting by IP
    answer = await call_your_llm_logic(chat_request.message)
    return ChatResponse(response=answer)
```

For per-API-key (rather than per-IP) rate limiting, key the limiter on the API key
extracted by your auth dependency instead of the client's IP address — more accurate
for clients behind shared NATs or proxies.

---

### 2.5 Error handling

LLM calls fail in many ways covered in Phase 01 (rate limits, timeouts, bad requests).
A production API needs to translate these into proper HTTP responses rather than
leaking raw exceptions or hanging.

```python
from openai import RateLimitError, APITimeoutError, APIError

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    try:
        answer = await call_your_llm_logic(request.message)
        return ChatResponse(response=answer)

    except RateLimitError:
        # 503: the upstream provider is rate-limiting US, not the client's fault
        raise HTTPException(status_code=503, detail="Service temporarily overloaded. Try again shortly.")

    except APITimeoutError:
        raise HTTPException(status_code=504, detail="The request took too long. Try again.")

    except APIError as e:
        # Log the real error internally, but don't leak provider-specific details to the client
        logger.error(f"LLM provider error: {e}")
        raise HTTPException(status_code=502, detail="Upstream AI service error.")

    except Exception as e:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error.")
```

**Why translate exceptions instead of letting them propagate:** An unhandled exception
in FastAPI returns a generic 500 with (by default, in debug mode) a stack trace —
leaking internal implementation details to clients and giving them no actionable
information. Mapping specific exceptions to specific status codes lets clients
distinguish "retry me later" (503/504) from "you sent something wrong" (400/422) from
"something is broken, page someone" (500/502).

---

### 2.6 Health checks

Container orchestrators (Docker, Kubernetes) and load balancers need a way to know if
your service is alive and ready to handle traffic.

```python
from datetime import datetime

@app.get("/health")
async def health_check():
    """Liveness check — is the process running at all?"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready")
async def readiness_check():
    """
    Readiness check — is the service actually able to serve traffic?
    Check dependencies: can we reach the LLM provider, vector DB, etc.
    """
    checks = {}
    try:
        # A cheap check that doesn't cost much (e.g., a models.list() call)
        await client.models.list()
        checks["llm_provider"] = "ok"
    except Exception as e:
        checks["llm_provider"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(content={"ready": all_ok, "checks": checks}, status_code=status_code)
```

**Liveness vs readiness, the distinction that matters:** Liveness answers "should this
container be restarted?" (if it fails, something is fundamentally broken — restart).
Readiness answers "should traffic be routed here right now?" (if it fails, the process
is fine but a dependency is temporarily down — don't restart, just stop sending traffic
until it recovers). Conflating the two causes unnecessary restarts during transient
upstream outages.

---

## 3. Streamlit & Gradio — Rapid UIs

### 3.1 When to use a rapid UI framework

Streamlit and Gradio let you build a usable web UI in Python, with no HTML/CSS/JS,
in far less time than a full frontend framework (React, Vue). They are the right choice
for: internal tools, demos, prototypes you want stakeholders to click through, and
data-science-adjacent products where the user base doesn't need a highly custom UI.
They are the wrong choice for a polished consumer product with custom branding and
complex interactions — for that, use the `frontend-design` patterns from a real
frontend framework.

---

### 3.2 Streamlit basics (you've used this since Phase 04)

You already built a Streamlit RAG app in Phase 04. The production addition here is
connecting Streamlit to your FastAPI backend over HTTP, rather than calling your AI
logic directly inside the Streamlit process — this separates the UI from the backend
the same way a real product separates frontend and backend services.

```python
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.title("My AI Service")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Stream from the FastAPI backend, not a direct LLM call —
        # this is the key production difference from earlier phases
        with httpx.stream(
            "POST", f"{API_URL}/chat/stream",
            json={"message": prompt},
            headers={"X-API-Key": st.secrets["API_KEY"]},
            timeout=60.0,
        ) as response:
            for chunk in response.iter_text():
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

**Why call the backend over HTTP instead of importing your AI logic directly:** This
mirrors real production architecture — the UI and the AI service can be deployed,
scaled, and updated independently. If you later replace Streamlit with a React frontend,
or add a mobile app, the FastAPI backend doesn't change at all.

---

### 3.3 Gradio — the alternative

Gradio is Streamlit's main competitor, with a slightly different philosophy: instead of
a script that re-runs top-to-bottom on every interaction (Streamlit's model), Gradio
wires specific functions to specific UI components, closer to a traditional event-driven
UI model. Gradio is especially popular for ML model demos (it's what most Hugging Face
Spaces use) and has excellent built-in support for streaming and chat interfaces.

```bash
pip install gradio
```

```python
import gradio as gr
import httpx

API_URL = "http://localhost:8000"

def chat_fn(message: str, history: list) -> str:
    """Gradio calls this function with the new message and the conversation history."""
    response = httpx.post(
        f"{API_URL}/chat",
        json={"message": message},
        headers={"X-API-Key": "your-api-key"},
        timeout=30.0,
    )
    return response.json()["response"]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="My AI Service",
    description="Ask me anything about the documents I've indexed.",
    examples=["What is the main topic?", "Summarize the key points."],
)

demo.launch(server_name="0.0.0.0", server_port=7860)
```

**Gradio's streaming chat pattern:**

```python
def chat_fn_streaming(message: str, history: list):
    """A generator function — Gradio recognizes this and streams the output."""
    partial_response = ""
    with httpx.stream(
        "POST", f"{API_URL}/chat/stream",
        json={"message": message},
        headers={"X-API-Key": "your-api-key"},
    ) as response:
        for chunk in response.iter_text():
            partial_response += chunk
            yield partial_response   # Gradio updates the UI on each yield

demo = gr.ChatInterface(fn=chat_fn_streaming)
```

**Streamlit vs Gradio:**

| | Streamlit | Gradio |
|---|---|---|
| Execution model | Script reruns top-to-bottom per interaction | Function-to-component wiring |
| Best for | Dashboards, multi-page apps, data exploration tools | Model demos, chat interfaces, ML showcases |
| Hugging Face Spaces | Supported | The default/most common choice |
| Learning curve | Slightly steeper for complex state | Very fast for simple chat/demo UIs |

---

## 4. Docker & Containerization

### 4.1 Why containers

"It works on my machine" fails in production because of differences in: Python version,
installed system libraries, environment variables, file paths, and OS-level dependencies.
A Docker container packages your application together with its exact runtime
environment — the same container that runs on your laptop runs identically on a cloud
server, a teammate's machine, or a Kubernetes cluster.

---

### 4.2 A Dockerfile for a FastAPI LLM service

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) — Docker caches this layer,
# so rebuilding after a code change (not a dependency change) is fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code
COPY . .

# Document the port the app listens on (informational; doesn't actually publish it)
EXPOSE 8000

# Run as a non-root user for security
RUN useradd -m appuser
USER appuser

# The command that runs when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `COPY requirements.txt .` before `COPY . .`:** Docker builds images in layers,
and caches each layer if its inputs haven't changed. If you copy all your code first
and `pip install` after, ANY code change invalidates the cache and forces a full
dependency reinstall on every rebuild. By installing dependencies from `requirements.txt`
first (which changes rarely) and copying code after (which changes often), you only
re-run `pip install` when dependencies actually change — dramatically speeding up
iterative development.

---

### 4.3 Building and running

```bash
# Build the image
docker build -t my-ai-service:latest .

# Run the container, mapping host port 8000 to container port 8000
docker run -p 8000:8000 \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e VALID_API_KEYS=$VALID_API_KEYS \
    my-ai-service:latest

# Run in the background (detached) with a name for easy reference
docker run -d --name ai-service -p 8000:8000 \
    --env-file .env \
    my-ai-service:latest

# View logs
docker logs -f ai-service

# Stop and remove
docker stop ai-service && docker rm ai-service
```

**Never bake secrets into the image.** Pass API keys via `-e` flags, `--env-file`, or
your orchestrator's secrets management — never `COPY` a `.env` file into the image or
hardcode keys in the Dockerfile. Anyone who can pull the image can extract anything
baked into its layers.

---

### 4.4 docker-compose for multi-service apps

A real product is rarely one container — you typically have the API backend, a frontend,
maybe a vector database, maybe a Redis cache. `docker-compose` defines and runs all of
them together.

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VALID_API_KEYS=${VALID_API_KEYS}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db   # persist vector DB data outside the container
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000   # service name "api" resolves via Docker's internal DNS
    depends_on:
      api:
        condition: service_healthy   # wait for the API's healthcheck to pass first
```

```bash
# Start everything
docker-compose up -d

# View logs from all services
docker-compose logs -f

# Stop everything
docker-compose down
```

**The key insight of `depends_on: condition: service_healthy`:** Without this, Docker
Compose starts containers in dependency order but does NOT wait for a service to be
actually ready — just for its process to have started. The frontend container could
start and immediately fail trying to reach an API that's still loading its model or
connecting to its vector database. The healthcheck-gated dependency ensures the frontend
only starts once the API genuinely reports itself ready.

---

### 4.5 Multi-stage builds for smaller images

For production, multi-stage builds keep the final image lean by discarding build-time
dependencies that aren't needed at runtime.

```dockerfile
# Stage 1: build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: final runtime image — only copies the installed packages, not build tools
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 5. LLM Evaluation — RAGAS & DeepEval

### 5.1 Why automated evaluation matters in production

Phase 04 (RAG) introduced manual evaluation: spot-checking a few questions, eyeballing
answers. This does not scale, and crucially does not catch **regressions** — if you
change your chunk size, your retrieval strategy, or your prompt, how do you know if
quality went up or down across your whole use case, not just the 3 examples you happened
to try by hand? Automated evaluation frameworks solve this by running a fixed test set
through your pipeline and computing quantitative metrics every time you make a change.

---

### 5.2 RAGAS — RAG-specific evaluation

RAGAS (RAG Assessment) is purpose-built for evaluating RAG pipelines, and computes
metrics using an LLM as the underlying judge (similar to the LLM-as-judge pattern from
Phase 07), but with carefully designed, decomposed metrics rather than a single holistic
score.

```bash
pip install ragas
```

**The core RAGAS metrics:**

| Metric | What it measures | Computed from |
|---|---|---|
| `faithfulness` | Is the answer supported by the retrieved context (no hallucination)? | answer + retrieved context |
| `answer_relevancy` | Does the answer actually address the question asked? | question + answer |
| `context_precision` | Are the retrieved chunks relevant (not noisy/irrelevant)? | question + retrieved context |
| `context_recall` | Did retrieval find everything needed to answer correctly? | retrieved context + ground truth |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Build an evaluation dataset: for each test question, run YOUR pipeline
# and collect what it retrieved and what it answered
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],       # list of retrieved chunk texts, per question
    "ground_truth": [],   # the correct/reference answer, for context_recall
}

test_questions = [
    {"question": "What is the boiling point of water at sea level?",
     "ground_truth": "100 degrees Celsius (212 degrees Fahrenheit)."},
    {"question": "What causes the seasons on Earth?",
     "ground_truth": "The tilt of Earth's axis relative to its orbit around the Sun."},
    # ... more test cases, ideally 20-50+ for stable metrics
]

for test_case in test_questions:
    # Run YOUR actual RAG pipeline (from Phase 04)
    response = rag_engine.ask(test_case["question"], k=5, strategy="hybrid")

    eval_data["question"].append(test_case["question"])
    eval_data["answer"].append(response.answer)
    eval_data["contexts"].append([chunk.text for chunk in response.sources])
    eval_data["ground_truth"].append(test_case["ground_truth"])

eval_dataset = Dataset.from_dict(eval_data)

# Run RAGAS evaluation
result = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.92, 'answer_relevancy': 0.87, 
#  'context_precision': 0.81, 'context_recall': 0.89}

df = result.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy"]])
```

---

### 5.3 Interpreting RAGAS scores

All four core metrics range from 0 to 1, where higher is better. But they diagnose
**different** problems, which is the key to using RAGAS effectively:

```
Low faithfulness, high context_precision
  → Retrieval found the right chunks, but the LLM is hallucinating beyond them.
  → Fix: tighten the generation prompt, lower temperature, add stricter "only use context" instructions.

High faithfulness, low context_precision
  → The LLM is faithfully reporting from context, but the context itself is noisy/irrelevant.
  → Fix: improve retrieval (re-ranking, better chunking) — see Phase 04, Section 6.

Low context_recall
  → Retrieval is missing information needed to answer correctly.
  → Fix: increase k, improve chunking strategy, try hybrid retrieval or HyDE.

Low answer_relevancy (even if faithfulness is high)
  → The answer is truthful given the context, but doesn't actually address the question.
  → Fix: improve the generation prompt to focus more directly on answering the specific question.
```

This decomposition is RAGAS's main value over a single holistic "is this answer good"
score — it tells you **which component of your pipeline** to fix.

---

### 5.4 DeepEval — broader LLM evaluation

DeepEval covers a wider range of evaluation needs beyond RAG specifically — general
answer correctness, safety/bias checks, conversational quality, and custom metrics — and
integrates with pytest, so evaluation can run as part of your CI pipeline.

```bash
pip install deepeval
```

```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

def test_rag_answer_quality():
    """A pytest-compatible test — run with `pytest test_eval.py`"""
    
    response = rag_engine.ask("What is the boiling point of water?")
    
    test_case = LLMTestCase(
        input="What is the boiling point of water?",
        actual_output=response.answer,
        retrieval_context=[chunk.text for chunk in response.sources],
        context=["Water boils at 100°C (212°F) at sea level."],  # ground truth context
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    hallucination_metric = HallucinationMetric(threshold=0.3)  # lower = stricter

    assert_test(test_case, [relevancy_metric, hallucination_metric])
    # This raises an AssertionError (failing the test) if either metric
    # doesn't meet its threshold — exactly like a normal pytest assertion
```

**Running evaluation as part of CI:**

```bash
# In your CI pipeline (GitHub Actions, etc.)
pytest test_eval.py -v

# DeepEval also has a CLI for generating a readable report
deepeval test run test_eval.py
```

**RAGAS vs DeepEval — when to use which:**

| | RAGAS | DeepEval |
|---|---|---|
| Specialization | Deep, RAG-specific metrics | Broader: RAG, agents, chatbots, custom metrics |
| CI/CD integration | Manual scripting | Native pytest integration |
| Best for | Diagnosing RAG pipeline components | General test-suite-style evaluation gates |

In practice, many production teams use both: RAGAS for deep RAG pipeline diagnosis
during development, DeepEval (or a custom pytest suite calling RAGAS metrics) as an
automated gate in CI that blocks deploys if quality regresses.

---

### 5.5 Building a regression-test evaluation suite

The production pattern: maintain a fixed set of test questions with known-good
expectations, and run them automatically whenever the pipeline changes.

```python
import json
from pathlib import Path

class EvaluationSuite:
    """
    A reusable, versioned test set for regression testing your RAG/agent pipeline.
    Run this after ANY change to chunking, retrieval, prompts, or models.
    """

    def __init__(self, test_file: str = "eval_suite.json"):
        self.test_file = test_file
        self.test_cases = self._load_or_create()

    def _load_or_create(self) -> list[dict]:
        if Path(self.test_file).exists():
            return json.loads(Path(self.test_file).read_text())
        return []

    def add_case(self, question: str, ground_truth: str, tags: list[str] = None):
        self.test_cases.append({
            "question": question,
            "ground_truth": ground_truth,
            "tags": tags or [],
        })
        Path(self.test_file).write_text(json.dumps(self.test_cases, indent=2))

    def run(self, rag_engine, metrics: list) -> dict:
        """Run the full suite and return aggregate + per-case results."""
        from ragas import evaluate
        from datasets import Dataset

        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        for case in self.test_cases:
            response = rag_engine.ask(case["question"])
            eval_data["question"].append(case["question"])
            eval_data["answer"].append(response.answer)
            eval_data["contexts"].append([c.text for c in response.sources])
            eval_data["ground_truth"].append(case["ground_truth"])

        result = evaluate(Dataset.from_dict(eval_data), metrics=metrics)
        return result


# Build the suite once
suite = EvaluationSuite()
suite.add_case(
    "What is the boiling point of water?",
    "100 degrees Celsius at sea level.",
    tags=["factual", "easy"],
)

# Run it any time you change something
from ragas.metrics import faithfulness, answer_relevancy
results = suite.run(rag_engine, metrics=[faithfulness, answer_relevancy])
print(results)

# Compare against a saved baseline to detect regressions
baseline_faithfulness = 0.90
if results["faithfulness"] < baseline_faithfulness - 0.05:
    print("⚠ REGRESSION DETECTED: faithfulness dropped significantly")
```

---

## 6. Observability — Langfuse & LangSmith

### 6.1 Why production observability differs from development tracing

You used LangSmith in Phase 03 and Phase 05 for development-time debugging — seeing
what a single chain or agent run did. Production observability has additional
requirements: aggregating metrics across thousands of users and requests, tracking cost
over time, alerting when error rates spike, and correlating traces with specific user
sessions for support/debugging purposes.

---

### 6.2 Langfuse — open-source LLM observability

Langfuse is a popular open-source alternative/complement to LangSmith, with strong
support for production-scale tracing, cost tracking, and prompt management.

```bash
pip install langfuse
```

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",   # or self-hosted URL
)

@observe()   # automatically traces this function's inputs, outputs, and timing
async def answer_question(question: str, user_id: str) -> str:
    response = await rag_engine.ask(question)
    return response.answer

# Usage — tracing happens automatically via the decorator
answer = await answer_question("What is RAG?", user_id="user-123")
```

**Manual, fine-grained tracing for multi-step pipelines:**

```python
from langfuse.decorators import observe, langfuse_context

@observe()
async def rag_pipeline(question: str):
    # Nested @observe() calls automatically create a trace tree,
    # exactly like LangSmith's nested spans
    chunks = await retrieve_step(question)
    answer = await generate_step(question, chunks)

    # Attach custom metadata/scores to the current trace
    langfuse_context.update_current_observation(
        metadata={"num_chunks_retrieved": len(chunks)},
    )
    return answer

@observe()
async def retrieve_step(question: str):
    return rag_engine.retrieve(question, k=5)

@observe()
async def generate_step(question: str, chunks: list):
    return await call_llm(question, chunks)
```

---

### 6.3 Tracking cost and usage per user

A critical production need: knowing not just "what happened" but "what did it cost,
and for whom" — essential for usage-based billing, abuse detection, and cost
optimization.

```python
from langfuse.decorators import observe, langfuse_context

@observe(as_type="generation")
async def call_llm_with_cost_tracking(prompt: str, user_id: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # Explicitly log token usage and cost — Langfuse aggregates this
    # across all calls, queryable by user, time range, or model
    langfuse_context.update_current_observation(
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
        metadata={"user_id": user_id},
    )

    return response.choices[0].message.content
```

In the Langfuse dashboard, this enables queries like "total cost per user this month"
or "which users are driving the most token usage" — essential for any product with
usage-based pricing or a need to detect anomalous/abusive usage patterns.

---

### 6.4 Scoring and feedback collection

Production systems need to capture quality signals from real usage, not just
pre-deployment test sets. Langfuse supports attaching scores to traces — from automated
evaluators (e.g., RAGAS run periodically on production traces) or from explicit user
feedback (thumbs up/down).

```python
from langfuse import Langfuse

langfuse = Langfuse()

# After a user gives feedback (e.g., clicking thumbs up/down in your UI)
def record_user_feedback(trace_id: str, is_positive: bool, comment: str = None):
    langfuse.score(
        trace_id=trace_id,
        name="user_feedback",
        value=1 if is_positive else 0,
        comment=comment,
    )

# In your FastAPI endpoint:
@app.post("/feedback")
async def submit_feedback(trace_id: str, is_positive: bool):
    record_user_feedback(trace_id, is_positive)
    return {"status": "recorded"}
```

```python
# Periodically score a sample of production traces with RAGAS automatically,
# so quality monitoring doesn't depend entirely on users bothering to click feedback buttons
def score_production_sample(traces: list, metrics: list):
    from ragas import evaluate
    for trace in traces:
        # Re-run RAGAS metrics against the actual production input/output
        score = compute_ragas_score(trace.input, trace.output, trace.context, metrics)
        langfuse.score(trace_id=trace.id, name="ragas_faithfulness", value=score)
```

---

### 6.5 Alerting on quality or cost anomalies

Most observability platforms (Langfuse, LangSmith, or a custom solution built on their
APIs/webhooks) support exporting metrics to standard monitoring stacks (Grafana,
Datadog) where you can set alerts.

```python
import logging

logger = logging.getLogger("ai_service")

async def chat_with_monitoring(message: str, user_id: str) -> str:
    """A production-instrumented endpoint with explicit alerting hooks."""
    try:
        response = await rag_engine.ask(message)

        # Log a structured event for anomaly detection downstream
        if response.num_chunks == 0:
            logger.warning(f"Zero chunks retrieved for user={user_id}, query={message[:50]}")

        return response.answer

    except Exception as e:
        # In production, this log line would be picked up by your log aggregator
        # (e.g. Datadog, CloudWatch) and trigger a PagerDuty alert if error rate spikes
        logger.error(f"chat_with_monitoring failed for user={user_id}: {e}", exc_info=True)
        raise
```

**The production observability checklist:**
1. ✅ Every request traced with input, output, latency, and cost
2. ✅ Errors logged with enough context to reproduce (user, input, stack trace)
3. ✅ Token usage and cost aggregable per user/time period
4. ✅ A feedback mechanism (explicit thumbs up/down, or implicit signals like retry rate)
5. ✅ Automated quality scoring on a sample of production traffic, not just pre-deploy tests
6. ✅ Alerting wired to a real notification channel (Slack, PagerDuty) for error rate
   or cost spikes — not just a dashboard nobody checks

---

## 7. Key Takeaways

1. **FastAPI + async is the standard for LLM backends.** Use `async def` for every
   LLM-calling endpoint (I/O-bound work benefits from async concurrency). Use
   `StreamingResponse` for streaming. Use `Depends()` for auth that composes across
   routes. Map provider exceptions to meaningful HTTP status codes.

2. **Liveness ≠ readiness.** `/health` answers "is the process alive" (restart if not).
   `/ready` answers "can dependencies be reached right now" (don't restart — wait).

3. **Streamlit/Gradio should call your backend over HTTP, not embed your AI logic
   directly.** This separation is what lets you swap frontends, scale them
   independently, and support multiple clients (web UI, mobile app) against one backend.

4. **Docker layer caching rewards installing dependencies before copying code.**
   `COPY requirements.txt . && RUN pip install` before `COPY . .` means code changes
   don't force dependency reinstalls. Never bake secrets into images — inject via
   environment variables at runtime.

5. **RAGAS decomposes RAG quality into diagnosable components.** Low faithfulness
   with high context_precision points at generation (hallucination); low context_recall
   points at retrieval. Don't just track one holistic score — track the four core
   metrics separately so you know *what* to fix.

6. **DeepEval brings evaluation into your test suite via pytest**, letting you gate
   deploys on quality thresholds the same way you'd gate on unit test failures.
   Maintain a fixed, versioned evaluation suite and re-run it on every pipeline change
   to catch regressions automatically.

7. **Production observability needs cost-per-user tracking, feedback collection, and
   automated alerting — not just request tracing.** A trace tells you what happened in
   one request; aggregated metrics, scoring, and alerts tell you whether the system is
   healthy across all your users, over time, without you having to look.

---

## 8. Practice Exercises

### Exercise 1 — Add Authentication and Rate Limiting (Easy)
Extend the project's FastAPI service with API key authentication (Section 2.3) and a
rate limit of 20 requests/minute per key (Section 2.4, adapted to key on the API key
rather than IP). Write a test that confirms a request with an invalid key is rejected
with 401, and that the 21st request within a minute is rejected with 429.

### Exercise 2 — Multi-Stage Docker Build (Medium)
Convert the project's single-stage Dockerfile into a multi-stage build (Section 4.5).
Compare the final image size before and after using `docker images`. Document the size
reduction.

### Exercise 3 — Automated Regression Gate (Medium-Hard)
Build a GitHub Actions workflow (or a local pre-commit script if you don't have a GitHub
repo) that runs the `EvaluationSuite` from Section 5.5 on every code change, fails the
build if `faithfulness` or `answer_relevancy` drops more than 0.05 below a saved
baseline, and otherwise updates the baseline file with the new scores.

### Exercise 4 — Full Observability Dashboard (Hard)
Integrate Langfuse into the project's FastAPI service so every `/chat` request is traced
with input, output, retrieved chunks, token usage, and cost. Add a `/feedback` endpoint
that records explicit thumbs up/down against the corresponding trace. Build a small
Streamlit "admin dashboard" page that queries Langfuse's API to show: total cost today,
average faithfulness score (if you've wired in periodic RAGAS scoring), and the 5 most
recent negative feedback traces with their full input/output for review.

---

## Course Complete

You've now built, end to end:
- **Phase 01:** A universal LLM client across 3 providers
- **Phase 02:** A structured data extraction pipeline
- **Phase 03:** A document chat system with memory
- **Phase 04:** A hybrid RAG pipeline with re-ranking
- **Phase 05:** An autonomous research agent
- **Phase 06:** A multi-agent content production system
- **Phase 07:** A fine-tuned domain-expert model
- **Phase 08:** The production wrapper around all of it — API, containers, evaluation,
  and observability

This is a complete AI engineering portfolio. The next step is yours: pick the project
that excites you most, extend it into something more substantial and personally
distinctive, deploy it somewhere real (Fly.io, Render, AWS, GCP), write up how you
built it, and put it in front of people.
