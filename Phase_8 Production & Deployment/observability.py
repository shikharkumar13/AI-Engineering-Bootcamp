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
    # langfuse.decorators (observe, langfuse_context) was removed when the SDK moved
    # to OpenTelemetry-based tracing — observe now lives at the top level, and the
    # per-call trace/span is reached through a client from get_client(), not a
    # separate langfuse_context object.
    #
    # get_client() (not a manual Langfuse(...) call) is deliberate: it auto-configures
    # from the LANGFUSE_* env vars load_dotenv() already populated above, and it's the
    # instance @observe's OTel spans actually attach to. Constructing Langfuse(...)
    # directly creates a second, disconnected instance — update_current_span() and
    # get_current_trace_id() silently no-op against it because @observe's spans were
    # never opened on that instance.
    from langfuse import observe, get_client

    _langfuse_client = get_client()
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

    def get_client():
        return None


# ── Traced service wrapper ─────────────────────────────────────────────────────

@observe(name="rag_ask")
def traced_ask(rag_service, question: str, k: int, user_id: str = "anonymous"):
    """
    Wraps RAGService.ask() with full tracing: input, output, retrieved chunks,
    latency, and (if available) token usage / cost.

    Returns (answer, trace_id). The trace ID must be read here, while this
    function's @observe span is still the OpenTelemetry "current" span — once
    this function returns, that span closes and the trace ID is no longer
    reachable via get_current_trace_id() (it returns None outside an active
    span), so callers can't fetch it after the fact the way the old SDK allowed.
    """
    answer = rag_service.ask(question, k=k)
    trace_id = None

    if LANGFUSE_ENABLED:
        client = get_client()
        client.update_current_span(
            input={"question": question, "k": k},
            output={"answer": answer.answer, "num_sources": answer.num_chunks},
            metadata={
                "user_id": user_id,
                "latency_s": answer.latency_s,
                "sources": [c.source for c in answer.sources],
            },
        )
        trace_id = client.get_current_trace_id()

    return answer, trace_id


def record_feedback(trace_id: str, is_positive: bool, comment: str | None = None):
    """Record explicit user feedback (thumbs up/down) against a trace."""
    if not LANGFUSE_ENABLED:
        logger.info(f"[no-op, Langfuse disabled] feedback for {trace_id}: "
                     f"{'positive' if is_positive else 'negative'} — {comment}")
        return

    _langfuse_client.create_score(
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

    _langfuse_client.create_score(trace_id=trace_id, name=metric_name, value=value)


def flush():
    """Flush any buffered events to Langfuse before process shutdown."""
    if LANGFUSE_ENABLED:
        _langfuse_client.flush()
