# Project 2 — Research Copilot

**Exercises:** Phase 3 (LangChain Orchestration) + Phase 4 (RAG & Vector Databases) + Phase 5 (AI Agents & LangGraph)

## What this project does

Multi-turn chat over your own documents that knows when to stop trusting
its own index:

1. **Retrieval:** every question is answered against your indexed
   documents using Phase 4's `RAGEngine` (hybrid BM25 + vector search,
   cross-encoder re-ranked).
2. **Memory:** follow-up questions ("what about X instead?") are
   understood in context, using Phase 3's memory pattern: recent turns are
   woven into the next query rather than every question being answered in
   isolation.
3. **Fallback to live research:** if the top re-ranked retrieval score
   comes back below a confidence threshold, the question is handed to
   Phase 5's `ResearchAgent` to search the live web instead of letting the
   RAG chain generate an answer from weak or irrelevant local context.

## Why this pairing

Phase 3 teaches memory, Phase 4 teaches retrieval, Phase 5 teaches knowing
when to reach for a different tool, each in isolation. A real "chat with
your docs" assistant needs all three at once: it has to remember what you
already asked, ground answers in your actual documents, and, critically,
recognize when a question falls outside what those documents cover instead
of confidently hallucinating an answer anyway.

## Project structure
```
Project_2 Research Copilot/
├── copilot_graph.py   ← ResearchCopilot class (RAG + memory + agent fallback)
├── demo.py             ← indexes a sample doc, shows an in-scope answer,
│                          a memory-dependent follow-up, and a fallback
├── requirements.txt
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and TAVILY_API_KEY
python demo.py
```

Note: on first run, `RAGEngine.__init__` (Phase 4) downloads the local
embedding and re-ranking models (~100-200MB combined); this only happens
once.

## API reference

```python
from copilot_graph import ResearchCopilot

copilot = ResearchCopilot()
copilot.load_document("handbook.pdf")     # Phase 4's RAGEngine.index()

turn = copilot.chat("What's the refund policy?")
print(turn.source)   # "local docs (hybrid+rerank, top score 0.81)"
print(turn.answer)

turn = copilot.chat("What about international orders?")  # remembers turn 1

turn = copilot.chat("What's today's weather in Tokyo?")   # not in your docs
print(turn.source)   # "web (Phase 5 research agent — local docs had no confident match)"

copilot.show_history()
```

## What this exercises from each phase

| From | What's reused |
|---|---|
| Phase 4 | `RAGEngine.index()`, `RAGEngine.ask()`, hybrid retrieval + re-ranking, per-chunk relevance scores used for the confidence check |
| Phase 5 | `ResearchAgent.research()` as the low-confidence fallback, its own per-session `thread_id` memory |
| Phase 3 | The conversational-memory *pattern* (per-session turn history woven into the next query) — applied on top of `RAGEngine`, which is single-shot on its own |

## A known limitation worth knowing

The confidence check uses `RetrievedChunk.score` from a single `RAGEngine.ask()`
call rather than a second, independent retrieval pass. This keeps the
common case (question is answerable locally) to one retrieval instead of
two, at the cost of still paying for one LLM generation on the low-confidence
path before falling back. For documents where near-miss retrieval on
off-topic questions still generates plausible-sounding text, this is worth
being aware of: a stricter design would check `retrieve()` confidence
*before* generating anything.

## Adapting this further

Swap `SAMPLE_DOC` in `demo.py` for your own documents, and tune
`confidence_threshold` in `ResearchCopilot.__init__` against your own
corpus. It's a re-ranker score, and what counts as "confident" depends
heavily on how well your documents match the questions you expect.
