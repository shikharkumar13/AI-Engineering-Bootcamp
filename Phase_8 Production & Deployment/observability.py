"""
observability.py — Langfuse Observability Integration

Wraps the RAG service calls with tracing, cost tracking, and a feedback
recording mechanism. Designed to fail gracefully if Langfuse isn't
configured (e.g. local dev without keys) — observability should never
break the actual service.
"""

import os
import logging
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ai_service.observability")

LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

if LANGFUSE_ENABLED:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context

    _langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    logger.info("Langfuse observability ENABLED")
else:
    _langfuse_client = None
    logger.warning(
        "Langfuse not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY missing). "
        "Running WITHOUT production tracing. Set these env vars to enable."
    )

    # No-op decorator so @observe() works even when Langfuse isn't configured
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        # support both @observe and @observe() usage
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    class _NoOpContext:
        def update_current_observation(self, *args, **kwargs):
            pass
        def update_current_trace(self, *args, **kwargs):
            pass

    langfuse_context = _NoOpContext()


# ── Traced service wrapper ─────────────────────────────────────────────────────

@observe(name="rag_ask")
def traced_ask(rag_service, question: str, k: int, user_id: str = "anonymous"):
    """
    Wraps RAGService.ask() with full tracing: input, output, retrieved chunks,
    latency, and (if available) token usage / cost.
    """
    answer = rag_service.ask(question, k=k)

    if LANGFUSE_ENABLED:
        langfuse_context.update_current_observation(
            input={"question": question, "k": k},
            output={"answer": answer.answer, "num_sources": answer.num_chunks},
            metadata={
                "user_id": user_id,
                "latency_s": answer.latency_s,
                "sources": [c.source for c in answer.sources],
            },
        )

    return answer


def record_feedback(trace_id: str, is_positive: bool, comment: str | None = None):
    """Record explicit user feedback (thumbs up/down) against a trace."""
    if not LANGFUSE_ENABLED:
        logger.info(f"[no-op, Langfuse disabled] feedback for {trace_id}: "
                     f"{'positive' if is_positive else 'negative'} — {comment}")
        return

    _langfuse_client.score(
        trace_id=trace_id,
        name="user_feedback",
        value=1 if is_positive else 0,
        comment=comment,
    )


def record_quality_score(trace_id: str, metric_name: str, value: float):
    """Record an automated quality score (e.g. from periodic RAGAS sampling) against a trace."""
    if not LANGFUSE_ENABLED:
        logger.info(f"[no-op, Langfuse disabled] {metric_name}={value:.3f} for {trace_id}")
        return

    _langfuse_client.score(trace_id=trace_id, name=metric_name, value=value)


def get_current_trace_id() -> str | None:
    """Get the current trace ID, for returning to the client (so they can submit feedback)."""
    if not LANGFUSE_ENABLED:
        return None
    return langfuse_context.get_current_trace_id()


def flush():
    """Flush any buffered events to Langfuse before process shutdown."""
    if LANGFUSE_ENABLED:
        _langfuse_client.flush()
