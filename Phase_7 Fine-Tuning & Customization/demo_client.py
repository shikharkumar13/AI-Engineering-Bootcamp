"""
demo_client.py — Test client for the inference API

This script does NOT require a GPU itself — it just sends HTTP requests to
inference_api.py, which must be running separately (on a GPU machine).

Run inference_api.py first:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000

Then run this:
    python demo_client.py
"""

import httpx
import time

API_URL = "http://localhost:8000"


TEST_MESSAGES = [
    "I was charged twice for my subscription this month, can you fix this?",
    "The mobile app won't let me log in even though my password is correct.",
    "How do I export my data before canceling my account?",
    "Your service has been down for an hour, what's going on?",
    "Can I change my billing email address?",
]


def check_health():
    print("Checking API health...")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10.0)
        response.raise_for_status()
        health = response.json()
        print(f"  Status: {health['status']}")
        print(f"  GPU available: {health['gpu_available']}")
        print(f"  Model loaded: {health['model_loaded']}")
        return health["model_loaded"]
    except httpx.ConnectError:
        print(f"  ✗ Could not connect to {API_URL}")
        print(f"  Make sure inference_api.py is running:")
        print(f"    uvicorn inference_api:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False


def send_message(message: str) -> dict:
    response = httpx.post(
        f"{API_URL}/chat",
        json={"message": message},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def run_demo():
    print("=" * 62)
    print("  Fine-tuned Support Assistant — API Demo")
    print("=" * 62)

    if not check_health():
        print("\n  Aborting demo — API not ready.")
        return

    print(f"\n  Sending {len(TEST_MESSAGES)} test messages...\n")

    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"{'─'*62}")
        print(f"  [{i}] Customer: {message}")
        print(f"{'─'*62}")

        try:
            result = send_message(message)
            print(f"  Issue:       {result.get('issue', 'N/A')}")
            print(f"  Resolution:  {result.get('resolution', 'N/A')}")
            print(f"  Next Steps:  {result.get('next_steps', 'N/A')}")
            print(f"  Latency:     {result['latency_s']}s")
        except Exception as e:
            print(f"  ✗ Request failed: {e}")

        print()

    print("=" * 62)
    print("  ✅ Demo complete.")
    print("=" * 62)


if __name__ == "__main__":
    run_demo()
