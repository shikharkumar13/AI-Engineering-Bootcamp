"""
fundamentals_demo.py — companion script for Python_fundamentals.md

Runnable, standard-library-only demo of the concepts covered in the
article: data types, control flow, functions, classes, dataclasses, error
handling, file I/O, JSON, type hints, decorators, context managers,
generators, and async/await. Each demo_*() function maps to one section of
the article and can be read independently.

Setup: nothing to install. This script only uses the Python standard
library, so `python fundamentals_demo.py` works right after cloning the
repo (Python 3.10+ for the `str | None` type-hint syntax used below).
"""

import asyncio
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path


def demo_data_types_and_comprehensions():
    print("=" * 60)
    print("1. Core data types & comprehensions (article Sections 2-3)")
    print("=" * 60)

    providers = ["openai", "claude", "gemini"]
    response = {"model": "gpt-4o-mini", "tokens": 42, "cost_usd": 0.0002}

    print(f"list: {providers}")
    print(f"dict: {response}")
    print(f"dict.get with default: {response.get('missing_key', 'n/a')}")

    squares = [n * n for n in range(5)]
    even_squares = [n * n for n in range(10) if n % 2 == 0]
    lengths = {word: len(word) for word in ["a", "bb", "ccc"]}

    print(f"list comprehension: {squares}")
    print(f"filtered comprehension: {even_squares}")
    print(f"dict comprehension: {lengths}")
    print()


def demo_functions():
    print("=" * 60)
    print("2. Functions, defaults, and **kwargs (article Section 4)")
    print("=" * 60)

    def add_tax(price: float, rate: float = 0.08) -> float:
        return price * (1 + rate)

    def chat_with_fallback(prompt: str, **kwargs):
        print(f"  prompt={prompt!r}, extra options={kwargs}")

    print(f"add_tax(100) = {add_tax(100)}")
    print(f"add_tax(100, 0.2) = {add_tax(100, 0.2)}")
    chat_with_fallback("Hello", system="Be concise.", max_tokens=300)
    print()


@dataclass
class TicketResult:
    """A @dataclass — see article Section 5.3. Auto-generates __init__,
    __repr__, and equality from these type-hinted fields."""
    ticket_id: int
    priority: str
    resolved: bool = False


class TicketProcessor:
    """A class bundling data + behavior — see article Section 5.
    __init__ sets up state once; methods below reuse it via self."""

    def __init__(self):
        self.results: list[TicketResult] = []

    def process(self, ticket_id: int, urgency_score: float) -> TicketResult:
        try:
            priority = self._score_to_priority(urgency_score)
        except ValueError as e:
            print(f"  ticket {ticket_id}: {e}")
            priority = "unknown"
        result = TicketResult(ticket_id=ticket_id, priority=priority)
        self.results.append(result)
        return result

    def _score_to_priority(self, score: float) -> str:
        """Leading underscore = internal helper, not part of the public
        interface (article Section 5.2)."""
        if not 0 <= score <= 1:
            raise ValueError(f"score {score} out of range 0-1")
        if score >= 0.7:
            return "high"
        elif score >= 0.3:
            return "medium"
        return "low"


def demo_classes_and_error_handling():
    print("=" * 60)
    print("3. Classes, dataclasses & error handling (article Sections 5-6)")
    print("=" * 60)

    processor = TicketProcessor()
    for ticket_id, score in [(1, 0.9), (2, 0.4), (3, 1.5)]:  # 1.5 is deliberately invalid
        result = processor.process(ticket_id, score)
        print(f"  {result}")
    print()


def demo_file_io_and_json():
    print("=" * 60)
    print("4. File I/O, pathlib & JSON (article Section 7)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "_demo_output"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "results.json"

    processor = TicketProcessor()
    for ticket_id, score in [(1, 0.9), (2, 0.4), (3, 1.5)]:
        processor.process(ticket_id, score)

    payload = [r.__dict__ for r in processor.results]
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {len(payload)} results to {results_path}")

    loaded = json.loads(results_path.read_text())
    print(f"  read back: {loaded}")
    print()


def demo_type_hints():
    print("=" * 60)
    print("5. Type hints (article Section 8)")
    print("=" * 60)

    def double(x: int) -> int:
        return x * 2

    print(f"  double.__annotations__ = {double.__annotations__}")
    print(f"  double(21) = {double(21)}")
    print("  Type hints aren't enforced at runtime: Python still runs this")
    print("  call even though a str isn't an int; it just does whatever")
    print("  str.__mul__ happens to mean instead of raising a type error:")
    print(f"  double('ab') = {double('ab')!r}")
    print()


def timed(func):
    """A simple decorator — see article Section 9.2. Takes a function,
    returns a wrapped version that adds timing around the original call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"  {func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper


@timed
def slow_function():
    time.sleep(0.2)
    return "done"


def demo_decorators():
    print("=" * 60)
    print("6. Decorators (article Section 9)")
    print("=" * 60)
    result = slow_function()
    print(f"  result = {result!r}")
    print()


@contextmanager
def timed_block(label: str):
    """A context manager written with @contextmanager — the code before
    `yield` is the setup, the code after is the guaranteed cleanup. See
    article Section 10; `with open(...)` and `with httpx.stream(...)`
    follow the identical shape under the hood."""
    start = time.time()
    try:
        yield
    finally:
        print(f"  [{label}] finished in {time.time() - start:.3f}s")


def demo_context_managers():
    print("=" * 60)
    print("7. Context managers (article Section 10)")
    print("=" * 60)
    with timed_block("sample work"):
        time.sleep(0.1)
        print("  doing work inside the with block...")
    print()


def count_up_to(n: int):
    """A generator — calling this doesn't run the loop immediately, it
    returns a generator object that produces one value per `yield`,
    on demand. See article Section 11."""
    i = 1
    while i <= n:
        yield i
        i += 1


def stream_reply(tokens: list[str]):
    """Mirrors the streaming pattern used by every phase's chat demo:
    forward each piece the moment it's available instead of waiting for
    the whole thing."""
    for token in tokens:
        yield token


def demo_generators():
    print("=" * 60)
    print("8. Generators & yield (article Section 11)")
    print("=" * 60)
    print(f"  count_up_to(3) -> {list(count_up_to(3))}")

    print("  streamed reply: ", end="")
    for piece in stream_reply(["Hel", "lo", ", ", "wor", "ld", "!"]):
        print(piece, end="", flush=True)
    print()
    print()


async def fetch_one(name: str, delay: float):
    print(f"  starting {name}")
    await asyncio.sleep(delay)
    print(f"  finished {name}")
    return f"{name} result"


async def _run_async_demo():
    results = await asyncio.gather(
        fetch_one("A", 0.2),
        fetch_one("B", 0.2),
        fetch_one("C", 0.2),
    )
    print(f"  results = {results}")


def demo_async_await():
    print("=" * 60)
    print("9. Async & await (article Section 12)")
    print("=" * 60)
    print("  running three 'requests' concurrently, note they all start")
    print("  before any of them finishes:")
    start = time.time()
    asyncio.run(_run_async_demo())
    print(f"  total time: {time.time() - start:.2f}s (not ~0.6s, because they overlapped)")
    print()


if __name__ == "__main__":
    demo_data_types_and_comprehensions()
    demo_functions()
    demo_classes_and_error_handling()
    demo_file_io_and_json()
    demo_type_hints()
    demo_decorators()
    demo_context_managers()
    demo_generators()
    demo_async_await()

    print("=" * 60)
    print("All demos complete. Re-read any section above alongside the")
    print("matching section number in Python_fundamentals.md.")
    print("=" * 60)
