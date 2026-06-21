"""
main.py — Production FastAPI Backend

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
    docker build -t ai-service .
    docker run -p 8000:8000 --env-file .env ai-service

Demonstrates all Phase 08 backend concepts:
  - async endpoints + StreamingResponse
  - API key authentication via Depends()
  - Rate limiting (slowapi)
  - Mapped exception handling (LLM errors -> proper HTTP status codes)
  - Liveness (/health) vs readiness (/ready)
  - Langfuse tracing wired into every request
  - Feedback endpoint
"""

import os
import logging
import time
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Security, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from openai import RateLimitError, APITimeoutError, APIError

from rag_service import RAGService
from observability import traced_ask, record_feedback, get_current_trace_id, flush

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_service")


# ── App state ───────────────────────────────────────────────────────────────────

rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG service once at startup."""
    global rag_service
    logger.info("Starting AI service...")
    rag_service = RAGService(
        collection_name=os.getenv("COLLECTION_NAME", "production_docs"),
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    )
    logger.info(f"RAG service ready. Stats: {rag_service.stats()}")
    yield
    logger.info("Shutting down — flushing observability events...")
    flush()


app = FastAPI(
    title="Production AI Service",
    description="Phase 08 capstone: FastAPI + Docker + RAGAS + Langfuse",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Rate limiting ───────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Authentication ──────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = set(filter(None, os.getenv("VALID_API_KEYS", "demo-key-123").split(",")))


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key header)")
    return api_key


# ── Schemas ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    num_chunks: int
    latency_s: float
    trace_id: str | None = None

class FeedbackRequest(BaseModel):
    trace_id: str
    is_positive: bool
    comment: str | None = None

class IndexResponse(BaseModel):
    source: str
    chunks_indexed: int
    total_in_collection: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ReadyResponse(BaseModel):
    ready: bool
    checks: dict


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness check — is the process running? (always returns 200 if reachable)"""
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())


@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness check — can the service actually serve traffic right now?"""
    checks = {}

    if rag_service is None:
        checks["rag_service"] = "not initialized"
    else:
        try:
            stats = rag_service.stats()
            checks["rag_service"] = "ok"
            checks["indexed_chunks"] = stats["total_chunks"]
        except Exception as e:
            checks["rag_service"] = f"error: {e}"

    all_ok = checks.get("rag_service") == "ok"
    status_code = 200 if all_ok else 503
    return JSONResponse(
        content=ReadyResponse(ready=all_ok, checks=checks).model_dump(),
        status_code=status_code,
    )


@app.post("/index", response_model=IndexResponse)
@limiter.limit("5/minute")
async def index_document(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """Upload and index a document (PDF or TXT) into the RAG knowledge base."""
    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag_service.index(tmp_path)
        return IndexResponse(**result)
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Non-streaming chat endpoint — full RAG answer with sources, traced."""
    try:
        answer = traced_ask(rag_service, chat_request.message, chat_request.k, user_id=api_key)

        return ChatResponse(
            answer=answer.answer,
            sources=[{"source": s.source, "score": round(s.score, 3),
                      "excerpt": s.text[:200]} for s in answer.sources],
            num_chunks=answer.num_chunks,
            latency_s=answer.latency_s,
            trace_id=get_current_trace_id(),
        )

    except RuntimeError as e:
        # e.g. "no documents indexed yet" — a client-fixable problem
        raise HTTPException(status_code=400, detail=str(e))

    except RateLimitError:
        raise HTTPException(status_code=503, detail="Upstream AI service is overloaded. Try again shortly.")

    except APITimeoutError:
        raise HTTPException(status_code=504, detail="The request took too long. Try again.")

    except APIError as e:
        logger.error(f"LLM provider error: {e}")
        raise HTTPException(status_code=502, detail="Upstream AI service error.")

    except Exception:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Streaming chat endpoint — tokens sent as they're generated."""
    if rag_service is None or rag_service.stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Call /index first.")

    async def token_stream():
        try:
            async for token in rag_service.ask_stream(chat_request.message, k=chat_request.k):
                yield token
        except Exception as e:
            logger.exception("Error during streaming")
            yield f"\n[ERROR: {e}]"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, api_key: str = Depends(verify_api_key)):
    """Record user feedback (thumbs up/down) against a specific trace."""
    record_feedback(feedback.trace_id, feedback.is_positive, feedback.comment)
    return {"status": "recorded"}


@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Service statistics — useful for a monitoring dashboard."""
    return rag_service.stats() if rag_service else {"error": "service not initialized"}


@app.get("/")
async def root():
    return {
        "service": "Production AI Service",
        "docs": "/docs",
        "endpoints": ["/health", "/ready", "/index", "/chat", "/chat/stream", "/feedback", "/stats"],
    }
