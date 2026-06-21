"""
demo_client.py — Exercises every endpoint of the production API

Run after starting the backend (either `uvicorn main:app` or `docker-compose up`):
    python demo_client.py
"""

import os
import time
import httpx
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "demo-key-123")
HEADERS = {"X-API-Key": API_KEY}

DIVIDER = "═" * 62

def section(title: str):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


SAMPLE_DOC = """
Customer Handbook

Refunds: Annual subscriptions are eligible for a prorated refund within the
first 30 days of purchase. Monthly subscriptions are not eligible for
partial-month refunds.

Password Reset: Use the 'Forgot Password' link on the login page.

Data Retention: If you cancel your account, your data is retained for 30
days, after which it is permanently deleted.

Plan Changes: Upgrades take effect immediately with a prorated charge.
Downgrades take effect at the end of the current billing cycle.
"""


def demo_health_and_readiness():
    section("1 — Health & Readiness Checks")

    health = httpx.get(f"{API_URL}/health", timeout=10.0).json()
    print(f"  /health: {health}")

    ready = httpx.get(f"{API_URL}/ready", timeout=10.0).json()
    print(f"  /ready:  {ready}")


def demo_auth():
    section("2 — Authentication")

    print("  Request WITHOUT API key (should fail with 401):")
    response = httpx.post(f"{API_URL}/chat", json={"message": "test"}, timeout=10.0)
    print(f"    Status: {response.status_code} — {response.json().get('detail')}")

    print("\n  Request WITH invalid API key (should fail with 401):")
    response = httpx.post(
        f"{API_URL}/chat", json={"message": "test"},
        headers={"X-API-Key": "wrong-key"}, timeout=10.0,
    )
    print(f"    Status: {response.status_code} — {response.json().get('detail')}")

    print("\n  Request WITH valid API key (should succeed or report no-docs):")
    response = httpx.post(
        f"{API_URL}/chat", json={"message": "test"}, headers=HEADERS, timeout=10.0,
    )
    print(f"    Status: {response.status_code}")


def demo_indexing():
    section("3 — Document Indexing")

    sample_path = Path("/tmp/demo_handbook.txt")
    sample_path.write_text(SAMPLE_DOC)

    with open(sample_path, "rb") as f:
        files = {"file": ("handbook.txt", f.read())}

    print(f"  Uploading {sample_path.name}...")
    response = httpx.post(f"{API_URL}/index", files=files, headers=HEADERS, timeout=60.0)
    response.raise_for_status()
    result = response.json()
    print(f"  ✓ Indexed: {result}")


def demo_chat():
    section("4 — Chat (non-streaming)")

    questions = [
        "What is the refund policy for annual subscriptions?",
        "How long is data retained after cancellation?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        response = httpx.post(
            f"{API_URL}/chat", json={"message": q, "k": 3}, headers=HEADERS, timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()
        print(f"  A: {result['answer'][:200]}")
        print(f"  Sources: {[s['source'] for s in result['sources']]}")
        print(f"  Latency: {result['latency_s']}s")
        if result.get("trace_id"):
            print(f"  Trace ID: {result['trace_id']}")
            return result["trace_id"]


def demo_streaming():
    section("5 — Chat (streaming)")

    question = "What happens if I want to downgrade my plan?"
    print(f"  Q: {question}")
    print(f"  A: ", end="")

    with httpx.stream(
        "POST", f"{API_URL}/chat/stream",
        json={"message": question, "k": 3}, headers=HEADERS, timeout=30.0,
    ) as response:
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)
    print()


def demo_rate_limiting():
    section("6 — Rate Limiting")

    print("  Sending 22 rapid requests (limit is 20/minute)...")
    statuses = []
    for i in range(22):
        response = httpx.post(
            f"{API_URL}/chat", json={"message": "quick test"}, headers=HEADERS, timeout=10.0,
        )
        statuses.append(response.status_code)

    rejected = sum(1 for s in statuses if s == 429)
    print(f"  Results: {statuses}")
    print(f"  {rejected} requests rejected with 429 (rate limited)")


def demo_feedback(trace_id: str | None):
    section("7 — Feedback Submission")

    if not trace_id:
        print("  No trace_id available (Langfuse may not be configured) — skipping.")
        return

    response = httpx.post(
        f"{API_URL}/feedback",
        json={"trace_id": trace_id, "is_positive": True, "comment": "Great answer!"},
        headers=HEADERS, timeout=10.0,
    )
    print(f"  Feedback submission status: {response.status_code} — {response.json()}")


def demo_stats():
    section("8 — Service Stats")
    response = httpx.get(f"{API_URL}/stats", headers=HEADERS, timeout=10.0)
    print(f"  {response.json()}")


if __name__ == "__main__":
    print(f"\n{DIVIDER}\n  Phase 08 — Production API Demo Client\n{DIVIDER}")
    print(f"  Target: {API_URL}")

    try:
        demo_health_and_readiness()
        demo_auth()
        demo_indexing()
        time.sleep(1)
        trace_id = demo_chat()
        demo_streaming()
        demo_feedback(trace_id)
        demo_stats()
        demo_rate_limiting()

        print(f"\n{DIVIDER}\n  ✅ Demo complete.\n{DIVIDER}")

    except httpx.ConnectError:
        print(f"\n  ✗ Could not connect to {API_URL}")
        print(f"  Start the backend first:")
        print(f"    uvicorn main:app --host 0.0.0.0 --port 8000")
        print(f"  or:")
        print(f"    docker-compose up")
