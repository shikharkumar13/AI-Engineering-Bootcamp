# Phase 0, Parts 2 & 3 - Python Fundamentals

> **Who this is for:** You've done Part 1 (dev environment, terminal, Git) and
> want a solid Python foundation before Part 4 (APIs) and the rest of this
> roadmap. This guide assumes no prior programming experience.  
> **What you'll have by the end:** Working comfort with every Python building
> block Phase 01 onward assumes without re-explaining: data types, control
> flow, functions, classes, error handling, file I/O, type hints, decorators,
> context managers, generators, and a first look at `async`/`await` - each
> grounded in the exact pattern you'll meet it in later.  
> **Time:** 6-9 hours total, spread over as many days as you need.

---

## Table of Contents

1. [The Big Picture - Why These Specific Topics](#1-the-big-picture--why-these-specific-topics)
2. [Variables & Core Data Types](#2-variables--core-data-types)
3. [Control Flow & Comprehensions](#3-control-flow--comprehensions)
4. [Functions](#4-functions)
5. [Classes - The Backbone of Every Phase Project](#5-classes--the-backbone-of-every-phase-project)
6. [Error Handling](#6-error-handling)
7. [File I/O, pathlib & JSON](#7-file-io-pathlib--json)
8. [Type Hints](#8-type-hints)
9. [Decorators](#9-decorators)
10. [Context Managers - the `with` statement](#10-context-managers--the-with-statement)
11. [Generators & `yield`](#11-generators--yield)
12. [Async & Await - A First Look](#12-async--await--a-first-look)
13. [Hands-On Walkthrough](#13-hands-on-walkthrough)
14. [Self-Check Before Moving On](#14-self-check-before-moving-on)
15. [Key Takeaways](#15-key-takeaways)
16. [What's Next](#16-whats-next)

---

## 1. The Big Picture - Why These Specific Topics

Python is a huge language, and a full course covers far more than any one
page could. This guide is deliberately narrower: it teaches exactly the
subset of Python that shows up, over and over, in every phase of this
roadmap. Not "Python in general" - *this repo's* Python.

Open almost any file from Phase 01 onward and you'll see the same handful of
patterns repeated: a class wrapping some logic, a `@dataclass` holding a
result, a `try`/`except` around a network call, a `with` block managing a
resource, an `async def` somewhere, a decorator sitting above a function. By
the end of this guide, none of that will look like unfamiliar syntax - it'll
look like five or six ideas you already understand, combined in different
orders.

**A quick preview of what's coming**, taken directly from patterns used
throughout Phases 01-08:

```python
@dataclass
class LLMResponse:              # Section 5 (classes) + a shortcut for it
    text: str                   # Section 8 (type hints)
    cost_usd: float

@retry(stop=stop_after_attempt(5))   # Section 9 (decorators)
def call_api(prompt: str) -> LLMResponse:
    try:                              # Section 6 (error handling)
        ...
    except RateLimitError:
        raise

async def stream_tokens(prompt: str):   # Section 12 (async)
    async with some_client() as client:  # Section 10 (context managers)
        async for chunk in client.stream(prompt):
            yield chunk.text              # Section 11 (generators)
```

If that block reads as "several small, separately-understandable pieces"
rather than "one wall of unfamiliar symbols," you're already most of the way
there. Work through the sections below, then come back and re-read it - it
should look almost boring by then.

---

## 2. Variables & Core Data Types

### 2.1 Variables

A variable is a name that points to a value. Python doesn't require you to
declare a type up front - you just assign:

```python
name = "Claude"
age = 5
is_helpful = True
```

The `=` here isn't mathematical equality, it's assignment: "make the name on
the left point to the value on the right, right now." The value can change
later; the name is just a label you attach to it.

### 2.2 The core built-in types

| Type | Example | Used for |
|---|---|---|
| `str` | `"hello"` | Text |
| `int` | `42` | Whole numbers |
| `float` | `3.14` | Decimal numbers |
| `bool` | `True` / `False` | Yes/no, on/off |
| `list` | `[1, 2, 3]` | An ordered, changeable collection |
| `tuple` | `(1, 2, 3)` | An ordered, *unchangeable* collection |
| `dict` | `{"key": "value"}` | Named values, looked up by key |
| `set` | `{1, 2, 3}` | Unique values, no particular order |

```python
# list — ordered, you can change it after creating it
providers = ["openai", "claude", "gemini"]
providers.append("mistral")
print(providers[0])          # "openai" — indexing starts at 0

# dict — the single most common structure in this entire course.
# Every API response you'll ever handle becomes one of these.
response = {
    "model": "gpt-4o-mini",
    "tokens": 42,
    "cost_usd": 0.0002,
}
print(response["model"])     # "gpt-4o-mini"
print(response.get("missing_key", "default"))  # safe lookup, no crash

# tuple — like a list, but locked once created. Used for values that
# are naturally fixed pairs/groups, e.g. a coordinate or a fixed return shape.
point = (3, 7)

# set — good for "does this exact value already exist," fast membership checks
seen_ids = {"a1", "b2"}
print("a1" in seen_ids)      # True
```

**From this repo:** every `.json()` response from an LLM provider, every row
of a dataset, every config block you'll write from Phase 01 onward is a
`dict`. Getting comfortable reading and building nested dicts (`{"role":
"user", "content": "..."}` inside a `list`, which is the exact shape of every
chat message you'll send) pays off immediately.

### 2.3 f-strings - building text from variables

You'll write these constantly:

```python
name = "Claude"
cost = 0.000195
print(f"Hello, {name}! This call cost ${cost:.6f}")
# → Hello, Claude! This call cost $0.000195
```

The `f` before the opening quote turns on this behavior; anything inside
`{ }` gets evaluated and dropped into the string. `:.6f` after a value is a
**format spec** - here, "show this as a fixed-point number with 6 digits
after the decimal," which is exactly how every phase project prints an API
call's cost.

---

## 3. Control Flow & Comprehensions

### 3.1 if / elif / else

```python
score = 7

if score >= 8:
    print("high confidence")
elif score >= 4:
    print("medium confidence")
else:
    print("low confidence")
```

Python uses **indentation** (consistent spaces, not curly braces) to mark
what's inside a block. This trips up almost everyone at first - a
mismatched indent is the single most common early error, and it's not a
sign anything is wrong with you.

### 3.2 for and while loops

```python
# for — loop over each item in a collection
for provider in ["openai", "claude", "gemini"]:
    print(f"Calling {provider}...")

# while — loop as long as a condition stays true
attempts = 0
while attempts < 3:
    print(f"Attempt {attempts + 1}")
    attempts += 1
```

`for` is by far the more common of the two in this course - almost every
loop you'll see iterates over a known collection (a list of chunks, a list
of messages, a list of tickets) rather than looping on an open-ended
condition.

### 3.3 Comprehensions - building a list/dict in one line

A comprehension is a compact way to build a new collection from an existing
one, instead of writing a `for` loop that appends to an empty list.

```python
# The loop way
squares = []
for n in range(5):
    squares.append(n * n)

# The comprehension way — same result, one line
squares = [n * n for n in range(5)]
# → [0, 1, 4, 9, 16]

# With a filter condition
even_squares = [n * n for n in range(10) if n % 2 == 0]

# Dict comprehension — same idea, building a dict instead of a list
lengths = {word: len(word) for word in ["a", "bb", "ccc"]}
# → {"a": 1, "bb": 2, "ccc": 3}
```

**From this repo:** lines like
`[c.source for c in chunks]` or
`{platform: content for platform, content in results.items()}`
show up in nearly every phase project - pulling one field out of a list of
objects, or reshaping a dict, is one of the most common operations in this
entire codebase. Once comprehensions read naturally, a lot of "real" code
stops looking dense.

---

## 4. Functions

A function is a named, reusable block of code. You define it once with
`def`, then **call** it by name whenever you want that job done.

```python
def add_tax(price: float, rate: float = 0.08) -> float:
    """Return the price with tax added."""
    return price * (1 + rate)

print(add_tax(100))         # 108.0 — uses the default rate
print(add_tax(100, 0.2))    # 120.0 — overrides it
```

Pieces worth naming:
- **Parameters** (`price`, `rate`) are the inputs a function accepts.
- A **default argument** (`rate: float = 0.08`) makes a parameter optional -
  callers can leave it out and get that value automatically.
- `return` hands a value back to whoever called the function. A function
  with no `return` implicitly returns `None`.
- The text in `"""triple quotes"""` right under `def` is a **docstring** - a
  description for humans reading the code. You'll see one under nearly
  every function and class in this course.

### 4.1 `*args` and `**kwargs` - accepting a flexible number of arguments

```python
def chat_with_fallback(prompt, providers=None, **kwargs):
    """kwargs scoops up any extra keyword arguments the caller passes,
    as a dict — so this one function can forward them along without
    listing every possible option by name."""
    print(kwargs)   # e.g. {"system": "...", "max_tokens": 300}

chat_with_fallback("Hello", system="Be concise.", max_tokens=300)
```

`**kwargs` ("keyword arguments") collects any `name=value` pairs the caller
passes that weren't explicitly listed as parameters, bundled into a dict.
`*args` does the same thing for extra *positional* arguments, bundled into a
tuple. You won't write these often as a beginner, but you'll **read** them
constantly - this exact `**kwargs` pattern is how Phase 01's `LLMClient`
forwards options like `system=` and `max_tokens=` through to whichever
provider ends up handling the call.

---

## 5. Classes - The Backbone of Every Phase Project

A class bundles related data and behavior into one reusable package. Where a
function does one job, a class represents a *thing* that has both
information (**attributes**) and abilities (**methods** - just functions
that belong to the class).

```python
class Dog:
    def __init__(self, name):
        self.name = name          # an attribute — data this Dog carries

    def bark(self):                # a method — an ability every Dog has
        print(f"{self.name} says woof!")

rex = Dog("Rex")     # creates a Dog, runs __init__ automatically
rex.bark()             # → "Rex says woof!"
```

`__init__` is a special method that runs automatically the moment you create
a new instance (`Dog("Rex")`) - it's where you set up the object's starting
attributes. `self` always means "this specific instance" - inside `bark()`,
`self` *is* `rex`, so `self.name` is `"Rex"`.

**Every class you'll meet from Phase 01 onward** - `LLMClient`,
`DataExtractor`, `RAGEngine`, `DocChat`, `ResearchAgent` - is built from
exactly this shape:

```python
class RAGEngine:
    def __init__(self, collection_name="rag_collection", persist_dir="./chroma_db"):
        self.collection_name = collection_name   # remembered for every
        self.persist_dir = persist_dir            # method below to use
        self._client = OpenAI()                   # set up once, reused

    def index(self, source):
        """A method — uses self._client, set up back in __init__."""
        ...

    def ask(self, question, k=5):
        """Another method — same instance, same remembered setup."""
        ...
```

### 5.1 Why classes instead of just functions?

You could write `index(client, source)` and `ask(client, question)` as
plain functions, passing `client` in every time. A class exists so you set
up the expensive or repeated stuff once (loading an embedding model,
opening a database connection, reading API keys) in `__init__`, and every
method afterward just uses `self.whatever` - no repetition, and no risk of
accidentally passing the wrong client to the wrong call.

### 5.2 A leading underscore means "internal"

```python
class LLMClient:
    def chat(self, prompt):            # public — meant to be called from outside
        return self._openai_call(prompt)

    def _openai_call(self, prompt):    # internal — an implementation detail
        ...
```

Python doesn't truly enforce privacy (nothing stops you from calling
`_openai_call` directly), but a leading underscore is a strong, universally
understood convention: "this is a helper for the class's own use, not part
of its public interface." You'll see this pattern in every phase project -
public methods with plain names, internal helpers prefixed with `_`.

### 5.3 `@dataclass` - a shortcut for data-holding classes

A huge number of classes in this course exist purely to hold a bundle of
related values - a response, a result, a stat. Writing `__init__` by hand
for these is repetitive, so Python gives you a shortcut:

```python
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str
    provider: str
    cost_usd: float
    latency_s: float = 0.0    # dataclasses support defaults too

response = LLMResponse(text="Hi!", provider="openai", cost_usd=0.0002)
print(response.text)          # "Hi!"
print(response)                # auto-generated, readable printout
```

`@dataclass` is a **decorator** (Section 9) that automatically writes
`__init__`, a readable `__repr__` (what prints when you `print()` it), and
equality comparison for you, just from the type-hinted fields you list.
`LLMResponse`, `RAGResponse`, `RetrievedChunk`, `TriageResult`,
`CopilotTurn` - every "bundle of results" class across this entire codebase
is a `@dataclass`.

---

## 6. Error Handling

Code fails: the network drops, a file doesn't exist, an API key is wrong.
Without handling this, one bad response crashes your entire program.
`try`/`except` lets you say "attempt this risky thing, and if it fails, do
something sensible instead of crashing."

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    result = None
    print("Can't divide by zero!")
```

### 6.1 Catching specific exceptions, not everything

```python
# BAD — catches literally anything, including bugs in your own code,
# and hides what actually went wrong
try:
    response = call_api(prompt)
except:
    print("something broke")

# GOOD — catches exactly the failure modes you expect and know how to handle
try:
    response = call_api(prompt)
except RateLimitError:
    print("Rate limited — wait and retry.")
except APITimeoutError:
    print("Timed out — try again.")
except Exception as e:
    # a deliberate, logged catch-all for anything else — not a silent swallow
    print(f"Unexpected error: {type(e).__name__}: {e}")
    raise
```

Catching a bare, unnamed `except:` (or overly broad `except Exception:`
without re-raising or logging) is a real mistake, not just a style
preference - it silently hides bugs that have nothing to do with the
network call you meant to guard against. Every error-handling block from
Phase 01 onward catches specific exception types on purpose, exactly like
the "GOOD" example above.

### 6.2 else and finally

```python
try:
    data = fetch(url)
except RequestException as e:
    print(f"Failed: {e}")
else:
    # runs only if the try block succeeded — no exception was raised
    print("Fetched successfully.")
finally:
    # always runs, whether it succeeded or failed — cleanup goes here
    print("Done attempting fetch.")
```

### 6.3 Raising your own exceptions

```python
def divide(a, b):
    if b == 0:
        raise ValueError("b cannot be zero")
    return a / b
```

`raise` deliberately triggers an exception - you're the one deciding "this
input is invalid, stop here and tell the caller why," rather than letting
the program continue with bad data.

### 6.4 Exception chaining with `from`

```python
try:
    response = client.chat.completions.create(...)
except APIStatusError as e:
    if e.status_code == 401:
        raise PermissionError("Invalid API key. Check your .env file.") from e
```

`raise NewError(...) from e` re-raises as a different, more meaningful
exception type while keeping the original one attached as context (visible
in the traceback as "The above exception was the direct cause of..."). This
is exactly the pattern Phase 01's error-handling section uses to translate a
raw provider error into something your own code, or a human reading the
crash, can act on directly.

---

## 7. File I/O, pathlib & JSON

### 7.1 Reading and writing files

"I/O" means input/output - reading data in, writing data out.

```python
with open("notes.txt", "w") as f:    # "w" = write (overwrites the file)
    f.write("Hello, file!\n")

with open("notes.txt", "r") as f:    # "r" = read
    contents = f.read()
```

`"w"` overwrites the whole file every time it's opened. `"a"` appends to the
end without erasing what's already there. `"r"` just reads. The `with`
block (Section 10) makes sure the file is properly closed even if something
goes wrong partway through - always prefer it over calling `open()` on its
own.

### 7.2 pathlib - the modern way to work with file paths

```python
from pathlib import Path

data_dir = Path("data")
file_path = data_dir / "tickets.json"   # / joins path segments — works on
                                          # both Mac/Linux and Windows

print(file_path.exists())                # True/False, no crash either way
file_path.write_text("hello")            # shortcut for open() + write()
content = file_path.read_text()          # shortcut for open() + read()
```

`pathlib.Path` largely replaces manually building path strings with `+` or
`os.path.join`. You'll see `Path(...)` constantly from Phase 01 onward - for
locating a repo folder, checking whether a cache file already exists, or
saving output.

### 7.3 JSON

**JSON** is the text format almost every API and config file in this course
uses. It maps almost directly onto Python's `dict` and `list`.

```python
import json

data = {"name": "Claude", "skills": ["writing", "coding"]}

# Python object → JSON text
json_text = json.dumps(data, indent=2)

# JSON text → Python object
parsed = json.loads(json_text)

# Straight to/from a file, skipping the intermediate string
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("data.json", "r") as f:
    loaded = json.load(f)
```

Remember the naming: `json.dumps`/`json.loads` work with strings already in
memory ("dump **s**tring" / "load **s**tring"); `json.dump`/`json.load`
(no `s`) work directly with an already-open file. Every evaluation baseline,
every saved config, every cached result across this repo's phase projects is
read and written with one of these four functions.

---

## 8. Type Hints

A type hint is an annotation telling readers (and your editor) what type a
variable, parameter, or return value is *expected* to be:

```python
def double(x: int) -> int:
    return x * 2

age: int = 25
name: str | None = None       # "either a str, or None"
scores: list[float] = []
config: dict[str, str] = {}
```

**The most important thing to understand: type hints are not enforced at
runtime.** Python will happily run `double("ab")` even though `"ab"` isn't
an `int` - it just does whatever `x * 2` happens to mean for a string
(`"abab"`), instead of rejecting the call because of the hint. Nothing
about the type hint itself stops the wrong type from being passed in;
whether the call then crashes or silently does something unintended
depends entirely on what the function's code does with that value. Hints
exist for two
audiences: your editor (which can now warn you *before* running the code, by
static analysis) and other humans reading the function, who instantly know
what's expected without reading the whole implementation.

`str | None` is modern syntax (Python 3.10+, used throughout this repo)
meaning "a `str`, or `None`" - it replaces the older `Optional[str]` you may
see in older code or tutorials; both mean the same thing.

**From this repo:** every function and method from Phase 01 onward is
type-hinted, and Phase 02 builds directly on top of this idea - a Pydantic
`BaseModel` (which you'll meet there) is essentially "type hints that
*are* enforced," validating real data against the shape you declared.

---

## 9. Decorators

A decorator is a function that wraps another function, adding behavior
before/after/around it, without changing the wrapped function's own code.
You apply one by writing `@decorator_name` directly above a `def`.

### 9.1 Reading one first

```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

Read this as: "take the `health_check` function below, and hand it to
`app.get("/health")`, which wraps it with the logic to register it as a
web route." You don't need to know how `app.get` is implemented to
understand *what's happening*: the function underneath is being modified or
registered by whatever sits above it. That's true of every decorator you'll
meet in this course - `@retry(...)`, `@observe()`, `@tool`, `@dataclass`
from Section 5.3.

### 9.2 Writing a simple one

```python
import time
from functools import wraps

def timed(func):
    @wraps(func)                 # preserves func's name/docstring — good practice
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timed
def slow_function():
    time.sleep(1)
    return "done"

slow_function()   # prints "slow_function took 1.00s", then returns "done"
```

`@timed` is exactly equivalent to writing `slow_function = timed(slow_function)`
right after defining it - the decorator receives the original function,
returns a new one (`wrapper`) that does extra work around calling it, and
that replacement is what the name `slow_function` ends up pointing to. This
is the whole mechanism: a decorator is a function that takes a function and
returns a (usually wrapped) function.

You will not need to write many decorators yourself in this course - but
you'll read `@retry(...)` (Phase 01, automatic retries), `@tool` (Phase 05
and 06, registering agent tools), `@observe()` (Phase 08, tracing), and
FastAPI's `@app.get(...)`/`@app.post(...)` (Phase 08) constantly. Knowing
the mechanism above means none of them are mysterious, even before you could
write one from scratch.

---

## 10. Context Managers - the `with` statement

You've already used `with` for files (Section 7.1). The general pattern is
broader: a **context manager** guarantees some setup happens before a block
of code runs, and some cleanup happens after - *even if the code inside
raises an exception*.

```python
with open("notes.txt") as f:
    contents = f.read()
# f is guaranteed to be closed here, even if .read() had failed
```

Without `with`, you'd need a manual `try`/`finally` to guarantee the file
gets closed on every path, including error paths - `with` does that for you
in one line.

**From this repo**, the same pattern shows up for things other than files:

```python
# Streaming an HTTP response — the connection stays open only for the
# duration of the block, and is cleaned up automatically afterward
with httpx.stream("POST", url, json=payload) as response:
    for chunk in response.iter_text():
        print(chunk, end="")

# Limiting concurrency in async code (Section 12) — acquires a "slot"
# on entry, releases it on exit, automatically
async with semaphore:
    result = await call_api(prompt)
```

The specific resource changes (a file, an HTTP connection, a concurrency
slot) but the shape is always the same: `with thing as name:`, guaranteed
cleanup, no matter how the block inside exits.

---

## 11. Generators & `yield`

A **generator** is a function that produces a sequence of values one at a
time, pausing between each, instead of computing and returning a whole
collection up front.

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i     # pause here, hand back i, resume from this exact
                     # point the next time a value is requested
        i += 1

for number in count_up_to(3):
    print(number)
# → 1
# → 2
# → 3
```

The moment a function contains `yield` anywhere in its body, calling it
doesn't run the code immediately - it returns a generator object, and the
function's code only actually runs (up to the next `yield`) each time
something asks for the next value, e.g. via a `for` loop.

**Why this matters for this repo:** every streaming LLM response you'll
build works exactly this way - the model produces text one token at a time,
and a generator forwards each token to whoever's consuming it the instant
it arrives, instead of waiting for the entire response and sending it all
at once:

```python
def stream_reply(prompt: str):
    for chunk in some_client.stream(prompt):
        yield chunk.text   # forward each piece immediately as it arrives

for piece in stream_reply("Explain generators"):
    print(piece, end="", flush=True)
```

This is precisely what makes a chat UI feel like it's "typing" a response
live instead of freezing for several seconds and dumping the whole answer
at once. Section 12 shows the `async` version of the same idea, which is
what every phase project from Phase 01 onward actually uses.

---

## 12. Async & Await - A First Look

Network calls (calling an LLM API, fetching a web page, querying a
database) spend almost all their time *waiting*, not computing. Your CPU is
idle the entire time a request is in flight. `async`/`await` lets Python
start another task during that wait instead of sitting there doing nothing.

```python
import asyncio

async def fetch_one(name: str):
    print(f"Starting {name}")
    await asyncio.sleep(1)          # simulates waiting on a network call
    print(f"Finished {name}")
    return f"{name} result"

async def main():
    # Run three "requests" concurrently instead of one after another
    results = await asyncio.gather(
        fetch_one("A"),
        fetch_one("B"),
        fetch_one("C"),
    )
    print(results)

asyncio.run(main())
# All three "Starting" lines print almost immediately, then all three
# "Finished" lines print about 1 second later — not 3 seconds later.
```

A few pieces to recognize, not master yet:

- `async def` marks a function as a **coroutine** - a function that can be
  paused and resumed, instead of running start-to-finish uninterrupted.
- `await` is where the pausing happens: "this will take a while, let other
  coroutines run while I wait, resume me when the result is ready."
- `asyncio.gather(...)` runs several coroutines concurrently and waits for
  all of them to finish.
- `asyncio.run(main())` is the entry point that actually starts the event
  loop and runs a top-level coroutine - you'll see this at the bottom of
  every async script in this course.

**You are not expected to master async from this one section.** Phase 01,
Section 7 covers it in real depth, with the exact patterns (`asyncio.gather`
for parallel calls, `asyncio.Semaphore` for rate-limiting how many run at
once) this repo actually uses. The goal here is narrower: when you see
`async def`, `await`, or `async with` for the first time in Phase 01's code,
none of it should look like foreign syntax - just "a function that can
pause," and "the place it pauses."

---

## 13. Hands-On Walkthrough

Time to use several of these together. Create a file called
`fundamentals_demo.py` (or work through the companion script that ships
alongside this article) and build up this small program in pieces.

**Step 1 - a dataclass to hold a result:**

```python
from dataclasses import dataclass

@dataclass
class TicketResult:
    ticket_id: int
    priority: str
    resolved: bool = False
```

**Step 2 - a class that processes a batch of them, with error handling:**

```python
class TicketProcessor:
    def __init__(self):
        self.results: list[TicketResult] = []

    def process(self, ticket_id: int, urgency_score: float) -> TicketResult:
        try:
            priority = self._score_to_priority(urgency_score)
        except ValueError as e:
            print(f"Ticket {ticket_id}: {e}")
            priority = "unknown"
        result = TicketResult(ticket_id=ticket_id, priority=priority)
        self.results.append(result)
        return result

    def _score_to_priority(self, score: float) -> str:
        if not 0 <= score <= 1:
            raise ValueError(f"score {score} out of range 0-1")
        if score >= 0.7:
            return "high"
        elif score >= 0.3:
            return "medium"
        return "low"
```

**Step 3 - run it, then save the results as JSON:**

```python
import json
from pathlib import Path

processor = TicketProcessor()
for ticket_id, score in [(1, 0.9), (2, 0.4), (3, 1.5)]:   # 1.5 is deliberately invalid
    processor.process(ticket_id, score)

output = [r.__dict__ for r in processor.results]   # dataclass → plain dict
Path("results.json").write_text(json.dumps(output, indent=2))
print(f"Saved {len(output)} results to results.json")
```

Run it and open `results.json` afterward - you should see three entries,
with ticket 3's priority as `"unknown"` (its score was out of range, and
your `try`/`except` caught that instead of crashing the whole batch). This
one small program touches a dataclass, a class with a public and a private
method, `try`/`except`, a comprehension, and file/JSON I/O - five of this
guide's sections, working together exactly the way a real phase project
does.

---

## 14. Self-Check Before Moving On

You're ready for Part 4 when you can, without looking anything up:

- [ ] Write a function that takes a list of dicts and returns a
      filtered/transformed list (comprehension or loop, your choice)
- [ ] Write a class with an `__init__` and at least one method that mutates
      instance state
- [ ] Explain what `@dataclass` saves you from writing by hand
- [ ] Wrap a risky operation in `try`/`except` and handle a *specific*
      exception, not a bare `except:`
- [ ] Read a JSON file into a Python object and write a modified version
      back out
- [ ] Explain in one sentence what a type hint like `def f(x: int) -> str:`
      communicates, and why it isn't enforced at runtime
- [ ] Explain in one sentence what `@decorator` does to the function
      underneath it
- [ ] Explain what guarantee a `with` block gives you that manually calling
      `open()` doesn't
- [ ] Explain the difference between a normal function and one containing
      `yield`
- [ ] Read a small `async def` function with one `await` in it and explain,
      in your own words, what it's waiting for

If several of those feel shaky, that's a completely normal signal to
re-read the relevant section and rebuild the Section 13 walkthrough from
scratch without looking - not a sign anything is wrong with you. Everything
from Phase 01 onward assumes these are automatic.

**Want more practice reps?** The author's dedicated Python course,
**https://github.com/shikharkumar13/Python-Programming-Code**, has room to
practice each of these ideas at greater length with more exercises. This
page is scoped tightly to what this repo needs; that one is a fuller
course if you want it.

---

## 15. Key Takeaways

1. **This repo's Python is a small, repeated set of patterns**, not the
   whole language - a class with `__init__`/`self`, a `@dataclass` for
   results, `try`/`except` around anything networked, `with` for anything
   needing cleanup, and `async`/`await` around anything I/O-bound.

2. **Type hints describe intent; they don't enforce it.** `def f(x: int)`
   is a promise to readers and your editor, not a runtime check - Phase 02's
   Pydantic models are what actually *enforce* a shape at runtime.

3. **A decorator is a function that wraps a function.** `@retry`,
   `@app.get(...)`, `@observe()`, `@dataclass` all follow the identical
   mechanism from Section 9, even though what each one *does* differs.

4. **`with` and `async with` guarantee cleanup**, whether or not the code
   inside succeeds - that's the entire reason they exist, for files, HTTP
   connections, and concurrency limits alike.

5. **A generator (`yield`) produces values one at a time, on demand** -
   this is the exact mechanism behind every "the model is typing..."
   streaming response you'll build starting in Phase 01.

6. **`async`/`await` lets Python overlap waiting time across multiple
   network calls**, instead of handling them one at a time - Phase 01,
   Section 7 is where this gets its full treatment.

---

## 16. What's Next

Move on to **Part 4 - Command Line, APIs & the Web**. Everything there
builds directly on this page: the `dict`s and JSON from Section 2 and 7 are
exactly what an API response looks like, the `try`/`except` patterns from
Section 6 are exactly how Phase 01 handles a failed request, and the
classes from Section 5 are the shape every provider client in this course
takes.
