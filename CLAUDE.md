# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A self-contained, nine-phase curriculum ("AI Engineer Roadmap") for learning to build,
evaluate, and deploy LLM-powered products, plus a small set of Practice Projects that
chain multiple phases together. It is **not** a single application — it is nine
independent, runnable mini-projects (Phase 0 through Phase 8), four cross-phase practice
projects, and a two-part bonus system-design series, each in its own top-level directory
with its own dependencies, `.env`, and entry point. There is no root-level build, lint, or
test command that spans the whole repo; every command below is run from inside a specific
project directory.

Each phase pairs a theory article (`*.md`, the "why") with a working project (`*.py`, the
"how") — the article isn't just doc scaffolding, it's the source of truth for the design
decisions baked into that phase's code.

## Repository layout

Top-level directories are named `Phase_N <Title>` **with spaces** (e.g.
`Phase_4 RAG & Vector Databases`), matching the root `README.md`'s "Repository Structure"
section exactly. Every phase (0 through 8) now follows the same filenames:
`requirements.txt`, `.env.example`, `demo.py` (or `app.py`/`main.py` where a phase's own
README says so), a `README.md`, and a topic-named article (e.g. `rag_vector_databases.md`)
— there is no `ARTICLE.md` anywhere, the topic-named `.md` file *is* the article. This
convention is uniform across all 9 phases now, including Phase 3, which previously used a
`phase03_`-prefixed filename scheme; if you see a `phaseNN_`-prefixed filename referenced
anywhere, that reference is stale and should be corrected to the plain name.

Three additional top-level directories sit alongside the phases:
- `Bonus AI System Design/` — `framework_and_tradeoffs.md` and `architecture_patterns.md`,
  a two-part system-design-interview series referenced from the root README.
- `Project_1 Smart Inbox Triage/`, `Project_2 Research Copilot/`,
  `Project_3 Autonomous Content Desk/`, `Project_4 Recipe & Meal Planner/` — see
  "Practice Projects" below.

## Running a phase project

```bash
cd "Phase_N <Title>"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in the keys listed inside it
python demo.py            # or app.py (Phase 4 Streamlit UI) / main.py (Phase 8 FastAPI)
```

Every phase's own `README.md` is the authoritative quick-start for that phase — it names
the exact entry point and any phase-specific deviation:
- **Phase 4** and **Phase 8** also expose a Streamlit UI (`streamlit run app.py` /
  `streamlit run streamlit_app.py`).
- **Phase 7** (Fine-tuning) requires a CUDA GPU (Colab or equivalent) for
  `unsloth`/`bitsandbytes`/`torch`; its `requirements.txt` is not installable on a
  CPU-only machine, and its other `.py` files should not be imported from a non-GPU
  script for the same reason (see Project 3 below for how to work around this).
- **Phase 8** is the only phase with a Docker path: `docker-compose up --build` runs a
  FastAPI backend (`main.py`) + Streamlit frontend (`streamlit_app.py`) together, wired to
  Langfuse observability (`observability.py`) and a RAGAS evaluation gate.

## Practice Projects — cross-phase, not standalone

`Project_1 Smart Inbox Triage/`, `Project_2 Research Copilot/`,
`Project_3 Autonomous Content Desk/`, and `Project_4 Recipe & Meal Planner/` each combine
2-3 adjacent phases into one project. Critically, **they import the relevant phase's
actual code directly** rather than duplicating logic — each project's core module
(`triage.py`, `copilot_graph.py`, `crew.py`, `meal_planner.py`) does this at the top of
the file:

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_N <Title>"))
from some_module import SomeClass   # imported straight from that phase's folder
```

This means:
- **Renaming or restructuring a file inside a phase folder can silently break a Practice
  Project's import.** Before renaming anything inside `Phase_1`, `Phase_2`, `Phase_3`,
  `Phase_4`, `Phase_5`, or `Phase_6`, grep the `Project_*` folders for imports of that
  phase's modules.
- **Project 3 deliberately does NOT import anything from Phase 7.** Phase 7's `train.py`
  and `inference_api.py` depend on CUDA-only packages (`unsloth`, `bitsandbytes`) that
  won't import on a non-GPU machine. Instead, `Project_3/crew.py` calls Phase 7's fine-tuned
  model over plain HTTP (`FINETUNED_MODEL_URL`), the same way Phase 7's own
  `demo_client.py` does, and degrades gracefully if that URL is unset or unreachable. Keep
  this pattern if you touch that code — do not add a direct import of Phase 7 modules.
- Run `python -m py_compile <file>.py` after editing any Practice Project module as a
  cheap sanity check; a full run additionally needs the target phase(s)' dependencies
  installed and real API keys, since these projects make live LLM calls with no mocking.

## Testing

There is no repo-wide test suite. Two evaluation gates exist, both run from inside their
own project directory:

```bash
# Phase 8 — RAGAS regression gate over the RAG service
cd "Phase_8 Production & Deployment" && pytest evaluation.py -v

# Project 3 — LLM-as-judge content-quality regression gate
cd "Project_3 Autonomous Content Desk" && pytest evaluation.py -v
```

Both follow the same pattern: run a suite, compare against a locally stored baseline
(`eval_baseline.json`), and fail if any metric regresses by more than a threshold (0.05).
First run on a fresh checkout establishes the baseline rather than failing.

## Architecture notes worth knowing before editing a phase

- **Phase 1** (`llm_client.py`) is a universal client wrapping OpenAI, Anthropic, and
  Gemini behind one interface (streaming, retries via `tenacity`, async, cost tracking).
  `LLMClient.__init__` constructs all three provider clients unconditionally, so all three
  API keys must be present in `.env` even if only one provider is actually used downstream
  (Project 1 inherits this behavior). Its `google-generativeai` dependency is EOL upstream
  (the vendor has deprecated it in favor of `google-genai`) — importing it prints a
  `FutureWarning`; this is a pre-existing upstream deprecation, not a bug in this repo.
- **Phase 2** (`extractor.py` + `models.py`) builds structured extraction on `instructor` +
  Pydantic models per document type (email, job posting, article, meeting notes, receipt).
  Project 1 imports `DataExtractor` and the shared `Priority`/`Sentiment`/`ActionItem`
  types from here directly. Project 4 also imports `DataExtractor`, calling its generic
  `extract(text, output_model)` entry point against its own locally-defined Pydantic
  models rather than the email/job-posting/etc. types defined in this phase's `models.py`.
- **Phase 3** (`doc_chat.py`) is a LangChain LCEL chain with `RunnableWithMessageHistory`
  for multi-turn memory over loaded documents. Project 2 reuses its *memory pattern*
  (recent turns woven into the next query) rather than importing `DocChat` itself, since
  `DocChat` stuffs the whole document into context instead of retrieving from it.
- **Phase 4** (`rag_engine.py` + `app.py`) implements hybrid retrieval (ChromaDB dense +
  BM25 sparse) with cross-encoder re-ranking and HyDE, exposed through Streamlit. Project 2
  imports `RAGEngine` directly and uses each retrieved chunk's `.score` to decide whether to
  trust local retrieval or fall back to live web search. Project 4 also imports `RAGEngine`
  directly, for cited Q&A over a personal recipe collection.
- **Phase 5** (`research_agent.py` + `tools.py`) is a LangGraph `StateGraph` ReAct agent
  with a SQLite checkpointer and human-in-the-loop support. Project 2 imports
  `ResearchAgent` directly as its low-confidence fallback.
- **Phase 6** (`agents.py`, `tasks.py`, `tools.py`, `content_factory.py`) is a CrewAI
  multi-agent pipeline (research → write → edit → publish) fanning out to 3 platforms in
  parallel. Project 3 imports `ContentFactory` directly and wraps it with a FastAPI layer.
- **Phase 7** (`dataset_prep.py`, `train.py`, `evaluate.py`, `inference_api.py`) is a
  QLoRA fine-tuning pipeline (Unsloth/PEFT/TRL) plus an LLM-as-judge evaluation against the
  base model, served via a small FastAPI (`inference_api.py`). GPU-only; see "Practice
  Projects" above for why Project 3 talks to it over HTTP instead of importing it.
- **Phase 8** (`main.py`, `rag_service.py`, `observability.py`, `evaluation.py`,
  `streamlit_app.py`) is the production shell: FastAPI backend with API-key auth
  (`X-API-Key`) and `slowapi` rate limiting sits in front of `RAGService` (adapted from
  Phase 4's RAG engine), traced through Langfuse, gated by the RAGAS suite above. Project
  3's `main.py`/`evaluation.py` follow this same shell pattern applied to Phase 6's crew
  instead of Phase 4's RAG service.
- **Phase 0** is the zero-to-hero on-ramp (dev environment, Python fundamentals, CLI/APIs,
  a no-math AI primer, and a capstone checkpoint) — start here only if picking up
  programming/AI concepts from scratch; otherwise start at Phase 1. Parts 2 & 3 (Python
  fundamentals) are an in-repo article plus a runnable, stdlib-only companion script
  (`fundamentals_demo.py`) — see `Phase_0 Prerequisites/Part-2 & 3 Python Fundamentals/Python_fundamentals.md`.
  It covers variables/control flow/functions/comprehensions, classes, dataclasses, error
  handling, file I/O, type hints, decorators, context managers, generators, and a first
  look at async/await, each grounded in a real pattern from later phases (e.g. `LLMResponse`
  as a `@dataclass`, `RAGEngine` as the canonical "class wraps setup, methods reuse it"
  shape). It no longer points to an external Python course.
