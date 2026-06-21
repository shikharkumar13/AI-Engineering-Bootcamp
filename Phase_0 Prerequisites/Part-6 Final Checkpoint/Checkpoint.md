# Phase 0, Part 6 - Capstone Checkpoint

> **Who this is for:** You've done Parts 1, 4, and 5 — terminal/Git/VS Code,
> APIs/JSON/.env, and the no-math AI primer. If you have **not** yet done Parts
> 2-3 (core Python fundamentals). This guide teaches just enough Python —
> functions, error handling, classes, file I/O, control flow to build one
> real project, explained inline as each piece is needed.  
> **What you'll have by the end:** A working command-line app you built
> yourself, pushed to GitHub, plus a final checklist before Phase 01.  
> **Time:** 2-3 hours.

---

## Table of Contents

1. [Before We Start — About the Parts 2-3 Gap](#1-before-we-start--about-the-parts-2-3-gap)
2. [What You're Building](#2-what-youre-building)
3. [Step 1 — Project Setup](#3-step-1--project-setup)
4. [Step 2 — Fetching Advice (Functions)](#4-step-2--fetching-advice-functions)
5. [Step 3 — Handling Errors Gracefully (try/except)](#5-step-3--handling-errors-gracefully-tryexcept)
6. [Step 4 — Saving Entries to a File (File I/O)](#6-step-4--saving-entries-to-a-file-file-io)
7. [Step 5 — Bundling It Together (Classes)](#7-step-5--bundling-it-together-classes)
8. [Step 6 — An Interactive Menu (Control Flow)](#8-step-6--an-interactive-menu-control-flow)
9. [The Complete Script](#9-the-complete-script)
10. [Running & Testing It](#10-running--testing-it)
11. [Pushing to GitHub](#11-pushing-to-github)
12. [The Final Phase 00 Readiness Checklist](#12-the-final-phase-00-readiness-checklist)
13. [Key Takeaways](#13-key-takeaways)
14. [What's Next](#14-whats-next)

---

## 1. Before We Start — About the Parts 2-3 Gap

Parts 2 and 3 normally is where you study Python properly: variables, data types, control
flow, functions, classes, error handling, file I/O, type hints — each with
room to breathe and practice. If you've skipped straight here, so this capstone
does something different: it builds **one real project**, and teaches each
new Python concept in a small **New concept** box, right at the moment the
project needs it — just enough to understand and use that piece correctly.

This is intentionally a tour, not a full course. By the end you'll have
working knowledge of functions, classes, error handling, file I/O, and
control flow loops — enough to read Phase 01 onward comfortably — but Parts
2 and 3 are still worth doing afterward for real depth and practice. Think
of this capstone as "just enough to be dangerous," with the proper
foundation available whenever you want it.

---

## 2. What You're Building

A **Daily Advice Journal** — a small command-line app that:

1. Fetches a random piece of advice from the same free API you used in
   Part 4 (`api.adviceslip.com`)
2. Saves it to a local file, with a timestamp
3. Lets you view everything you've saved so far
4. Runs as an interactive menu, looping until you choose to quit

This single project touches everything you might be missing from Parts 2-3 —
functions, error handling, classes, file I/O, loops — while reusing exactly
what you already know from Part 4 (calling an API, reading JSON).

```
┌─────────────────────────────────────────────┐
│            Daily Advice Journal              │
├─────────────────────────────────────────────┤
│  1. Get new advice and save it                │
│  2. View all saved advice                      │
│  3. Quit                                         │
└─────────────────────────────────────────────┘
```

---

## 3. Step 1 — Project Setup

This is pure Part 1 review — do it without looking, if you can.

```bash
# Navigate to your course folder
cd ai-engineer-course   # adjust the path if needed

# Create a folder for this capstone
mkdir phase00-capstone
cd phase00-capstone

# Create and activate a virtual environment
python -m venv venv          # Windows: python, Mac: python3
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac
# Confirm you see (venv) in your prompt

# Install the one package this project needs
pip install requests
```

Open this `phase00-capstone` folder in VS Code (**File > Open Folder**),
and create a new file called `advice_journal.py`. You'll build it up in
stages below — type each piece in yourself as you go.

---

## 4. Step 2 — Fetching Advice (Functions)

> **New concept: functions.** A function is a named, reusable block of
> code that does one specific job. You "call" it by name whenever you want
> that job done, instead of retyping the same code every time. Functions
> are defined with `def`, can accept inputs (called **parameters**, in
> parentheses), and can hand back a result with `return`.
>
> ```python
> def add_one(number):       # defines a function named add_one,
>     return number + 1       # taking one input called "number"
>
> result = add_one(5)         # "calling" it — result becomes 6
> ```

You already wrote code like this in Part 4 — now it's wrapped properly in a
function so it can be reused:

```python
import requests


def fetch_advice():
    """
    Calls the free Advice Slip API and returns a single piece of advice
    as plain text.
    """
    response = requests.get("https://api.adviceslip.com/advice", timeout=5)
    data = response.json()
    return data["slip"]["advice"]
```

The text in triple quotes right under the `def` line is called a
**docstring** — a description of what the function does, for humans
reading the code (you've seen these throughout every phase project in this
course).

Type this into `advice_journal.py` for now. We'll improve it in the next step.

---

## 5. Step 3 — Handling Errors Gracefully (try/except)

> **New concept: try/except.** Sometimes code fails — the internet might be
> down, the API might be temporarily broken. Without handling this, your
> entire program crashes the instant something goes wrong. `try`/`except`
> lets you say "attempt this risky thing, and if it fails, do something
> sensible instead of crashing."
>
> ```python
> try:
>     result = 10 / 0          # this would normally crash the program
> except ZeroDivisionError:
>     result = None             # instead, we handle it gracefully
>     print("Can't divide by zero!")
> ```

Update `fetch_advice()` to handle the API call failing — no internet, a
slow response, or the service being temporarily down:

```python
def fetch_advice():
    """
    Calls the free Advice Slip API and returns a single piece of advice
    as plain text. Returns None if anything goes wrong, so the calling
    code can decide what to do next instead of crashing.
    """
    try:
        response = requests.get("https://api.adviceslip.com/advice", timeout=5)
        response.raise_for_status()  # raises an error if status code is 4xx/5xx
        data = response.json()
        return data["slip"]["advice"]
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch advice right now: {e}")
        return None
```

`response.raise_for_status()` is a `requests` library shortcut: if the
status code (Part 4, Section 3.3) indicates an error, it deliberately
triggers an exception right there, which our `except` block then catches.

---

## 6. Step 4 — Saving Entries to a File (File I/O)

> **New concept: file I/O.** "I/O" means input/output — reading data in,
> writing data out. To work with a file in Python, you use `open()`,
> almost always paired with `with`, which automatically handles closing the
> file properly even if something goes wrong partway through.
>
> ```python
> with open("notes.txt", "w") as f:   # "w" = write (overwrites the file)
>     f.write("Hello, file!\n")
>
> with open("notes.txt", "r") as f:   # "r" = read
>     contents = f.read()
>     print(contents)
> ```
>
> The mode letter matters: `"w"` overwrites the whole file every time,
> `"a"` appends to the end without erasing what's already there, `"r"`
> just reads.

We'll save each advice entry as one line of JSON in a file — a common,
simple pattern called **JSON Lines** (one JSON object per line, instead of
one giant JSON file). Here's the saving function on its own, before we
wrap it in a class in the next step:

```python
import json
from datetime import datetime

def save_entry(filepath, advice_text):
    """Append a new advice entry to the journal file, with a timestamp."""
    entry = {
        "advice": advice_text,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(filepath, "a") as f:        # "a" = append, never erases old entries
        f.write(json.dumps(entry) + "\n")  # one JSON object, on its own line
```

`json.dumps()` converts a Python dictionary into a JSON-formatted string —
the exact reverse of `response.json()` from Part 4, which converts a JSON
string into a Python dictionary.

---

## 7. Step 5 — Bundling It Together (Classes)

> **New concept: classes.** A class bundles related data and functions
> together into one reusable package. Where a plain function does one job,
> a class represents a *thing* that has both information (called
> **attributes**) and abilities (called **methods** — just functions that
> belong to the class).
>
> ```python
> class Dog:
>     def __init__(self, name):   # __init__ runs automatically when you
>         self.name = name         # create a new Dog — it sets up the
>                                   # dog's starting attributes
>
>     def bark(self):              # a method — an ability every Dog has
>         print(f"{self.name} says woof!")
>
> my_dog = Dog("Rex")    # creates a new Dog, runs __init__ automatically
> my_dog.bark()           # → "Rex says woof!"
> ```
>
> `self` always refers to "this specific instance of the class" — when you
> call `my_dog.bark()`, inside that method, `self` *is* `my_dog`. Every
> class you've seen throughout this entire course (`RAGEngine`, `DocChat`,
> `ResearchAgent`) is built from exactly this pattern: `__init__` to set up,
> `self` to refer to the specific instance, and methods for its abilities.

Let's bundle saving *and* loading entries into one `AdviceJournal` class:

```python
class AdviceJournal:
    """Manages saving and loading advice entries to a local file."""

    def __init__(self, filepath="journal.jsonl"):
        self.filepath = filepath   # remembered for every method below to use

    def add_entry(self, advice_text):
        """Save a new advice entry, with the current date and time."""
        entry = {
            "advice": advice_text,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_entries(self):
        """Read every saved entry back from the file."""
        if not os.path.exists(self.filepath):
            return []   # no file yet means no entries yet — not an error

        entries = []
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def print_all_entries(self):
        """Display every saved entry in a friendly format."""
        entries = self.load_entries()
        if not entries:
            print("No entries saved yet.")
            return
        print(f"\nYou have {len(entries)} saved entries:\n")
        for i, entry in enumerate(entries, start=1):
            print(f"  {i}. [{entry['saved_at']}] {entry['advice']}")
```

Notice this is the exact same saving logic from Step 4, just moved inside
the class and using `self.filepath` instead of a separate `filepath`
argument every time — the class remembers it once, in `__init__`, so every
method afterward can just use `self.filepath`.

You'll also need `import os` at the top of your file for
`os.path.exists()`, which checks whether a file already exists before
trying to read it.

---

## 8. Step 6 — An Interactive Menu (Control Flow)

> **New concept: while loops, if/elif/else, and input().** A `while` loop
> repeats a block of code as long as some condition stays true.
> `if`/`elif`/`else` lets your program make decisions. `input()` pauses
> your program and waits for the person using it to type something.
>
> ```python
> while True:                      # loops forever, until something breaks out of it
>     answer = input("Continue? (y/n): ")
>     if answer == "y":
>         print("Continuing...")
>     elif answer == "n":
>         print("Stopping.")
>         break                     # break exits the loop immediately
>     else:
>         print("Please type y or n.")
> ```

Now the part that ties everything together — a menu loop:

```python
def main():
    journal = AdviceJournal()   # creates the journal — runs __init__ from Step 5

    print("=" * 50)
    print("  Daily Advice Journal")
    print("=" * 50)

    while True:
        print("\nWhat would you like to do?")
        print("  1. Get new advice and save it")
        print("  2. View all saved advice")
        print("  3. Quit")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            advice = fetch_advice()
            if advice:
                print(f"\nToday's advice: {advice}")
                journal.add_entry(advice)
                print("(saved to your journal)")
        elif choice == "2":
            journal.print_all_entries()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
```

That last line — `if __name__ == "__main__":` — you've seen in every
phase project's code. It means "only run `main()` if this file was run
directly (not imported by another file)." For now, just know it belongs at
the bottom of scripts meant to be run directly, exactly like this one.

---

## 9. The Complete Script

Here's everything from Steps 2-6, assembled in the order it should appear
in your file. Compare this against what you've typed — they should match.

```python
# advice_journal.py
import json
import os
from datetime import datetime
import requests


def fetch_advice():
    """
    Calls the free Advice Slip API and returns a single piece of advice
    as plain text. Returns None if anything goes wrong, so the calling
    code can decide what to do next instead of crashing.
    """
    try:
        response = requests.get("https://api.adviceslip.com/advice", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["slip"]["advice"]
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch advice right now: {e}")
        return None


class AdviceJournal:
    """Manages saving and loading advice entries to a local file."""

    def __init__(self, filepath="journal.jsonl"):
        self.filepath = filepath

    def add_entry(self, advice_text):
        """Save a new advice entry, with the current date and time."""
        entry = {
            "advice": advice_text,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_entries(self):
        """Read every saved entry back from the file."""
        if not os.path.exists(self.filepath):
            return []

        entries = []
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def print_all_entries(self):
        """Display every saved entry in a friendly format."""
        entries = self.load_entries()
        if not entries:
            print("No entries saved yet.")
            return
        print(f"\nYou have {len(entries)} saved entries:\n")
        for i, entry in enumerate(entries, start=1):
            print(f"  {i}. [{entry['saved_at']}] {entry['advice']}")


def main():
    journal = AdviceJournal()

    print("=" * 50)
    print("  Daily Advice Journal")
    print("=" * 50)

    while True:
        print("\nWhat would you like to do?")
        print("  1. Get new advice and save it")
        print("  2. View all saved advice")
        print("  3. Quit")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            advice = fetch_advice()
            if advice:
                print(f"\nToday's advice: {advice}")
                journal.add_entry(advice)
                print("(saved to your journal)")
        elif choice == "2":
            journal.print_all_entries()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
```

---

## 10. Running & Testing It

```bash
python advice_journal.py    # Windows
python3 advice_journal.py   # Mac
```

Try this exact sequence:

1. Choose `1` — you should see a real piece of advice printed, and
   "(saved to your journal)" underneath
2. Choose `1` again — a different piece of advice
3. Choose `2` — you should see both entries listed, each with a timestamp
4. Choose `3` — the program says goodbye and exits cleanly

Look in your project folder afterward — you'll see a new file called
`journal.jsonl` that didn't exist before. Open it in VS Code; you'll see
your entries as raw JSON, one per line, exactly as Step 4 described.

**If something goes wrong:** re-read the New Concept box for the section
involved, and check your code character-for-character against Section 9 —
a single mismatched indent or missing colon is the most common cause of
errors at this stage, and that's completely normal, not a sign anything is
wrong with you.

---

## 11. Pushing to GitHub

Pure Part 1, Section 7 review:

```bash
git init
git add .
git commit -m "Phase 00 capstone - Daily Advice Journal"
```

Create a new, empty repository on GitHub called `phase00-capstone`, then:

```bash
git remote add origin https://github.com/yourusername/phase00-capstone.git
git branch -M main
git push -u origin main
```

> **Note on `journal.jsonl`:** for this project, it's fine to push your
> journal file along with your code — it's just saved advice, not a
> secret. But it's worth asking yourself the question from Part 4 here:
> *is there anything in this folder that shouldn't be public?* For this
> project, no. Starting Phase 01, the answer will very often be yes (your
> `.env` file) — which is exactly why `.gitignore` exists.

Refresh your GitHub repository page — you should see `advice_journal.py`
and `journal.jsonl` sitting there. **This is your second real GitHub
repository**, and your first one with actual logic in it, not just a
one-line hello-world.

---

## 12. The Final Phase 00 Readiness Checklist

Go through this honestly. Anything unchecked is worth revisiting before
Phase 01 — there's no rush, and Phase 01 will be far smoother with a solid
foundation here.

**Dev environment (Part 1)**
```
[ ] I can open a terminal and navigate between folders without help
[ ] I can create and activate a virtual environment, and I know how to
    tell it's active
[ ] I can create a file in VS Code, run it, and read the output in the
    integrated terminal
[ ] I can git init, add, commit, and push a project to GitHub
```

**Python basics (Parts 2-3, or this capstone's condensed tour)**
```
[ ] I can read a function definition (def ...) and explain what it does
[ ] I understand what try/except is for, even if I couldn't write
    complex error handling from scratch yet
[ ] I can read a class with __init__ and self and explain roughly what's
    happening
[ ] I understand the difference between "w", "a", and "r" file modes
[ ] I can read a while loop and an if/elif/else block
```

**APIs & the web (Part 4)**
```
[ ] I can explain what an API call actually is, in my own words
[ ] I know why secrets go in .env files and never directly in code
[ ] I can read a JSON response and know it maps onto Python
    dictionaries and lists
[ ] I understand what a status code like 200, 401, or 404 means
```

**AI vocabulary (Part 5)**
```
[ ] I can explain, in one sentence each: token, parameter, training vs
    inference
[ ] I can explain why an LLM generates text one piece at a time
[ ] I have an intuitive feel for what an embedding is (points on a map,
    similar meaning = nearby points)
```

**If most of these are checked:** you're ready for Phase 01. Some
unfamiliarity is completely normal and will solidify as you go — this
checklist is a guide, not a gate.

**If several boxes in the Python basics section are unchecked:** strongly
consider doing Parts 2 and 3 properly before Phase 01. This capstone gave
you a working tour, but Phase 01's code will move faster than this guide
did, and a firmer foundation will make every phase from here on
noticeably less frustrating.

---

## 13. Key Takeaways

1. **You built one real project that touches every Python fundamental
   you were missing** — functions, error handling, classes, file I/O, and
   control flow — by building something, not by reading isolated examples.

2. **Every class you'll see in Phases 01-08** (`RAGEngine`, `DocChat`,
   `ResearchAgent`, and dozens more) follows the exact same shape as
   `AdviceJournal`: `__init__` to set up, `self` to refer to the current
   instance, methods for its abilities.

3. **This was a tour, not a replacement for Parts 2-3.** If anything in
   this guide felt rushed, that's the signal to go back and get the full
   depth before moving forward.

4. **The readiness checklist is honest, not a formality** — use it for
   real. Phase 01 assumes everything in it.

---

## 14. What's Next

If your checklist came back mostly solid: **you're done with Phase 0.**
Move straight into **Phase 01 - Universal LLM Client**, where everything
from this entire pre-requisite series finally gets put to direct use —
terminal and venv skills from Part 1, the HTTP/JSON/API-key concepts from
Part 4 (you'll recognize `Authorization: Bearer` immediately), the
token/parameter vocabulary from Part 5, and the functions/classes/error-
handling fundamentals from this capstone.
