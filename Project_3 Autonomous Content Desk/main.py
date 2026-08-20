"""
main.py — Content Desk FastAPI backend

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
    docker-compose up --build

Follows Phase 8's production shell pattern (API-key auth, rate limiting,
mapped error handling, liveness/readiness) applied to Phase 6's content
crew instead of Phase 4's RAG service.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from crew import ContentDesk

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("content_desk")


# ── App state ──────────────────────────────────────────────────────────

desk: ContentDesk | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global desk
    logger.info("Starting Content Desk...")
    desk = ContentDesk(verbose=False)
    logger.info("Content Desk ready.")
    yield


app = FastAPI(
    title="Autonomous Content Desk",
    description="Phase 6 (multi-agent crew) + Phase 7 (fine-tune comparison) "
    "behind Phase 8's production shell",
    version="1.0.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = set(filter(None, os.getenv("VALID_API_KEYS", "demo-key-123").split(",")))


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key header)")
    return api_key


# ── Schemas ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=300)


class GenerateResponse(BaseModel):
    topic: str
    edited_article: str
    platform_content: dict[str, str]
    timings: dict[str, float]


class CompareRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class CompareResponse(BaseModel):
    message: str
    finetuned_available: bool
    finetuned_response: dict | None
    crew_response: str
    crew_latency_s: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())


@app.get("/ready")
async def readiness_check():
    ready = desk is not None
    return JSONResponse(
        content={"ready": ready},
        status_code=200 if ready else 503,
    )


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("5/minute")
async def generate(request: Request, body: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Runs the full Phase 6 research -> write -> edit -> multi-platform pipeline."""
    try:
        result = desk.generate(body.topic)
        return GenerateResponse(
            topic=result.topic,
            edited_article=result.edited_article,
            platform_content=result.platform_content,
            timings=result.timings,
        )
    except Exception:
        logger.exception("Content generation failed")
        raise HTTPException(status_code=500, detail="Content generation failed.")


@app.post("/support-compare", response_model=CompareResponse)
@limiter.limit("10/minute")
async def support_compare(request: Request, body: CompareRequest, api_key: str = Depends(verify_api_key)):
    """
    Runs the same support message through Phase 7's fine-tuned model (if
    FINETUNED_MODEL_URL is configured and reachable) and a general-purpose
    crew agent, side by side — see crew.py's module docstring for why.
    """
    try:
        comparison = desk.compare_support_response(body.message)
        return CompareResponse(
            message=comparison.message,
            finetuned_available=comparison.finetuned_available,
            finetuned_response=comparison.finetuned_response,
            crew_response=comparison.crew_response,
            crew_latency_s=comparison.crew_latency_s,
        )
    except Exception:
        logger.exception("Support comparison failed")
        raise HTTPException(status_code=500, detail="Support comparison failed.")


@app.get("/")
async def root():
    return {
        "service": "Autonomous Content Desk",
        "docs": "/docs",
        "endpoints": ["/health", "/ready", "/generate", "/support-compare"],
    }
