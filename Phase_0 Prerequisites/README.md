# Phase 0 — Prerequisites

A hand-held bridge for anyone starting this roadmap with **no** programming
background. If you already know Python and have shipped code before, skip
straight to `Phase_1 LLM APIs/` — nothing here is required reading for you.

Work through the parts in order:

| Part | Folder | What it covers |
|---|---|---|
| 1 | [`Part-1 Dev Environment/`](Part-1%20Dev%20Environment/dev_environment_and_tools.md) | Terminal basics, installing Python, virtual environments, VS Code, your first Git/GitHub push |
| 2 & 3 | [`Part-2 & 3 Python Fundamentals/`](Part-2%20%26%203%20Python%20Fundamentals/Python_fundamentals.md) | In-repo Python course scoped to exactly what this repo needs: variables/control flow/functions/comprehensions, then classes/dataclasses/error handling/file I/O/type hints/decorators/context managers/generators/async, each tied to real code from later phases, plus a runnable companion script (`fundamentals_demo.py`) |
| 4 | [`Part-4 Command Line and APIs/`](Part-4%20Command%20Line%20and%20APIs/Command_line_APIs_the_Web.md) | HTTP, JSON, what an API actually is, API keys and `.env` files |
| 5 | [`Part-5 No-Math AI Basics/`](Part-5%20No-Math%20AI%20Basics/AI_Primer.md) | A no-math mental model for ML, neural networks, LLMs, tokens, and embeddings |
| 6 | [`Part-6 Final Checkpoint/`](Part-6%20Final%20Checkpoint/Checkpoint.md) | Capstone project (a command-line advice journal) + a readiness checklist before Phase 01 |

Each part folder is self-contained: its own article (`.md`) plus, where
relevant, a demo/exercise script you can run directly (e.g. `check_setup.py`,
`api_demo.py`, `Embedding_example.py`, `advice_journal.py`). None of these
scripts need an external `requirements.txt` — each article tells you the one
or two packages to `pip install` as they come up.

Once you've finished Part 6's readiness checklist, move on to
`Phase_1 LLM APIs/`.
