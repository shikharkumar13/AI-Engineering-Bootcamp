# phase00_part4_api_demo.py
#
# A hands-on demo of everything in Part 4: a GET request, reading a status
# code, reading a JSON response, and sending an API key safely via .env.
#
# Setup before running:
#   1. Activate your virtual environment (you should see (venv) in your prompt)
#   2. Install the two packages this needs:
#        pip install requests python-dotenv
#   3. Make sure phase00_part4_.env.example has been copied to a real ".env"
#      file in this same folder (see the bottom of this file for exactly
#      what it should contain)
#   4. Run it:
#        python phase00_part4_api_demo.py     (Windows)
#        python3 phase00_part4_api_demo.py    (Mac)

import os
import requests
from dotenv import load_dotenv


def demo_basic_get_request():
    """
    The simplest possible HTTP request: ask a server for something,
    look at what comes back.
    """
    print("=" * 60)
    print("  1. A basic GET request")
    print("=" * 60)

    response = requests.get("https://httpbin.org/get")

    print(f"Status code: {response.status_code}")
    print(f"Response (as a Python dict, converted from JSON):")
    print(response.json())
    print()


def demo_status_codes():
    """
    Deliberately trigger a few different status codes, to see the
    difference between success, client error, and server error.
    """
    print("=" * 60)
    print("  2. Triggering different status codes on purpose")
    print("=" * 60)

    for code in [200, 404, 500]:
        response = requests.get(f"https://httpbin.org/status/{code}")
        print(f"Requested status {code}  →  got back: {response.status_code}")
    print()


def demo_hardcoded_auth_BAD_EXAMPLE():
    """
    THIS IS WHAT NOT TO DO. Shown here only so you can see the contrast
    with the next function. Never write a real secret directly in code
    like this.
    """
    print("=" * 60)
    print("  3. Authenticated request (BAD — hardcoded secret, for comparison only)")
    print("=" * 60)

    fake_token = "my-test-token-123"  # if this were a REAL secret, this
                                        # line alone would be a security bug

    headers = {"Authorization": f"Bearer {fake_token}"}
    response = requests.get("https://httpbin.org/bearer", headers=headers)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def demo_proper_env_based_auth():
    """
    THIS is the correct pattern: the secret lives in .env, never in this
    file, and is loaded at runtime with load_dotenv() + os.getenv().
    This is the exact pattern Phase 01 onward uses for every API key.
    """
    print("=" * 60)
    print("  4. Authenticated request (GOOD — secret loaded from .env)")
    print("=" * 60)

    load_dotenv()  # reads the .env file in this folder, if present

    token = os.getenv("DEMO_API_TOKEN")

    if not token:
        print("DEMO_API_TOKEN not found. Did you create a .env file?")
        print("See the bottom of this script for what it should contain.")
        print()
        return

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://httpbin.org/bearer", headers=headers)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def demo_missing_auth():
    """Show what happens when authentication is missing entirely — a 401."""
    print("=" * 60)
    print("  5. What happens with NO auth header at all")
    print("=" * 60)

    response = requests.get("https://httpbin.org/bearer")
    print(f"Status code: {response.status_code}  "
          f"(401 = Unauthorized — exactly what you'll see if your own "
          f".env file is missing an API key in later phases)")
    print()


if __name__ == "__main__":
    demo_basic_get_request()
    demo_status_codes()
    demo_hardcoded_auth_BAD_EXAMPLE()
    demo_proper_env_based_auth()
    demo_missing_auth()

    print("=" * 60)
    print("  Done. Compare sections 3 and 4 above — same result, but only")
    print("  section 4's approach is safe to ever push to GitHub.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────
# To make demo_proper_env_based_auth() work, create a file named exactly
# ".env" (with the leading dot) in this same folder, containing:
#
#   DEMO_API_TOKEN=my-test-token-123
#
# And a file named ".gitignore" in this same folder, containing:
#
#   .env
#
# This ensures that if you ever push this folder to GitHub, your .env
# file (and any real secrets inside it) never gets uploaded.
# ─────────────────────────────────────────────────────────────────────────
