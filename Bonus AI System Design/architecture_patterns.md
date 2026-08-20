# Bonus, Part 2 — AI System Design: Architecture Pattern Library

> **Prerequisites:** Part 1 (framework & trade-offs) and Phases 01–08.
> **What you'll learn:** The RAG architecture spectrum from naive to
> agentic, the common agent/multi-agent topologies, how caching actually
> works for LLM systems, where safety/guardrail layers sit, what a real
> monitoring stack looks like, and a combined reference architecture tying
> all of it together — with a pointer to exactly which phase of this repo
> already builds each piece.
> **Time:** 1-2 hours.

---

## Table of Contents

1. [The RAG Architecture Spectrum](#1-the-rag-architecture-spectrum)
2. [Agent & Multi-Agent Patterns](#2-agent--multi-agent-patterns)
3. [Caching Architecture](#3-caching-architecture)
4. [Safety & Guardrail Layers](#4-safety--guardrail-layers)
5. [The Monitoring Stack](#5-the-monitoring-stack)
6. [A Combined Reference Architecture](#6-a-combined-reference-architecture)
7. [Key Takeaways](#7-key-takeaways)

---

## 1. The RAG Architecture Spectrum

RAG isn't one architecture — it's a spectrum from "one retrieval call
stapled to one generation call" to "an agent that decides for itself when,
how, and how many times to retrieve." Knowing where a given design sits on
this spectrum, and why, is a large fraction of any RAG-flavored system
design answer.

### 1.1 Naive RAG

```
query → embed → vector search (top-k) → stuff chunks into prompt → LLM → answer
```

One retrieval pass, no re-ranking, no query understanding, whatever the
top-k similarity search returns is what the model sees. This is where
everyone starts (Phase 04's Demo 2), and it's a legitimate production
architecture when the corpus is small, well-structured, and questions map
closely to how the docs are written. It breaks down on: vague questions,
questions needing information from multiple chunks, and near-duplicate
chunks crowding out the actually-relevant one.

### 1.2 Advanced RAG

Everything Phase 04 actually builds:

```
query ──► query rewrite/expansion (optional)
            │
            ├──► BM25 keyword search   ──┐
            │                            ├──► RRF merge (top ~20)
            └──► vector semantic search ──┘
                            │
                  cross-encoder re-rank
                            │
                     top 5 → LLM
                            │
              answer + [source N] citations
```

Additions over naive RAG, each solving a specific failure mode:
- **Hybrid retrieval (BM25 + vector, merged with Reciprocal Rank Fusion):**
  vector search misses exact keyword/code/ID matches; BM25 misses semantic
  paraphrases. Combining them covers both failure modes.
- **Cross-encoder re-ranking:** the initial retrieval is optimized for
  recall (cast a wide net, top-20), a heavier cross-encoder model then
  re-scores those 20 for precision before only the top ~5 reach the LLM.
- **HyDE (Hypothetical Document Embeddings):** for abstract/vague queries,
  generate a hypothetical answer first and embed *that* to search — a
  hypothetical answer is often closer in embedding space to the real answer
  chunk than the bare question is.
- **Parent-child chunking:** index small chunks (precise retrieval) but
  return their larger parent chunk to the LLM (enough surrounding context to
  actually answer from).

This is the right default for most production RAG systems: it solves the
common failure modes without introducing an agent's unpredictability or cost.

### 1.3 Agentic RAG

```
query → agent decides:
          "Do I have enough to answer, or do I need to retrieve (again)?"
             │                                    │
         retrieve (specific sub-query)        answer directly
             │                                    │
        agent re-evaluates ◄──────────────────────┘
             │
        (loop, bounded by a step/tool-call limit)
             │
        final answer
```

The retrieval step becomes a *tool* the agent chooses to call, potentially
multiple times with different sub-queries, interleaved with reasoning. This
is Phase 05's ReAct pattern applied specifically to retrieval instead of
(or in addition to) web search. Use this when questions genuinely require
multi-hop reasoning ("compare X's Q1 numbers to Y's Q1 numbers, which are in
two different documents") that a single retrieval pass can't satisfy — the
cost is materially higher latency and token spend, per Part 1's
latency-vs-quality trade-off, so don't reach for it by default.

### 1.4 Choosing a point on the spectrum

| Signal | Points toward |
|---|---|
| Questions map closely to how docs are phrased, small corpus | Naive RAG |
| General-purpose Q&A over a real document set, tight latency SLA | Advanced RAG |
| Multi-hop questions, comparisons across documents, "research this" tasks | Agentic RAG |
| Sub-second latency requirement | Naive or Advanced, never Agentic |

---

## 2. Agent & Multi-Agent Patterns

### 2.1 Single ReAct agent

```
        ┌─────────┐
     ┌─►│  agent  │──► "call a tool" or "conclude"?
     │  └────┬────┘
     │       │
     │   ┌───┴────┐
     │  tool     done
     │   │         │
     │   ▼         ▼
     │ ┌──────┐   answer
     └─┤ tools│
       └──────┘
```

One model, one loop, a set of tools, a step limit to prevent infinite loops
(Phase 05). Right for tasks with a single, coherent "persona" and no need to
separate concerns — a research agent, a coding assistant, a support bot.

### 2.2 Planner–executor

```
     query → planner (decomposes into subtasks)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    executor 1  executor 2  executor 3   (can run in parallel if independent)
        │          │          │
        └──────────┴──────────┘
                   │
              synthesizer → final answer
```

A planning step explicitly decomposes the task before execution, instead of
deciding one step at a time like ReAct. Better for tasks with clear
sub-structure known upfront (multi-part research questions, "gather these
4 pieces of information then combine them") — worse for tasks where the next
step genuinely depends on what the previous step returned.

### 2.3 Role-specialized multi-agent crew

```
Researcher → Writer → Editor → ┬─► Formatter A ┐
                                ├─► Formatter B ├─ (parallel, independent outputs)
                                └─► Formatter C ┘
```

This is Phase 06 exactly: agents with distinct roles, goals, and system
prompts, wired with explicit context-passing between sequential stages, and
independent stages fanned out in parallel. The advantage over a single
mega-prompt: each stage is independently testable, independently
promptable, and you can swap the model per stage (a cheap/fast model for
formatting, an expensive one for the writing stage that actually needs
quality).

### 2.4 Supervisor / orchestrator pattern

```
                   ┌──────────────┐
   query ────────► │  supervisor   │
                   └───────┬──────┘
                           │  routes to the right specialist
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        agent: billing agent: technical agent: general
```

A router/supervisor agent (or even a simple classifier) decides *which*
specialist handles a given request, rather than every request flowing
through the same fixed pipeline. Common in support/assistant systems with
genuinely different domains that shouldn't share a system prompt.

### 2.5 Choosing a pattern

| Signal | Points toward |
|---|---|
| One coherent task, tools decided step-by-step | Single ReAct agent |
| Task has known sub-structure upfront | Planner–executor |
| Distinct roles with a natural pipeline (research→write→edit) | Role-specialized crew |
| Requests fall into genuinely different categories needing different handling | Supervisor/orchestrator |

The failure mode to watch for in interviews: reaching for multi-agent
because it sounds sophisticated, when a single well-prompted agent (or no
agent at all — see Part 1 Section 5's worked example) would be faster,
cheaper, and easier to debug. Every extra agent hop is another latency and
cost line-item per Part 1's trade-off table.

---

## 3. Caching Architecture

LLM calls are the most expensive, slowest part of almost any AI system —
caching them is often the single highest-leverage optimization available.

```
                    ┌─────────────────┐
  query ──────────► │  exact-match     │── hit ──► return cached response
                    │  cache (hash of   │
                    │  normalized query)│
                    └────────┬─────────┘
                             │ miss
                             ▼
                    ┌─────────────────┐
                    │  semantic cache   │── hit (similarity > threshold) ──► return
                    │  (embed query,    │        cached response for nearest match
                    │  search cache)    │
                    └────────┬─────────┘
                             │ miss
                             ▼
                        full pipeline
                     (retrieve + generate)
                             │
                             ▼
                   write result to both caches
```

- **Exact-match caching:** hash the normalized query (and any parameters
  that affect the answer — model, retrieved-doc version). Cheap, simple,
  catches literal repeats ("what's your refund policy" asked 200 times/day).
- **Semantic caching:** embed the incoming query and search a cache of
  past query embeddings for a near-duplicate above a similarity threshold.
  Catches paraphrases ("what's your return policy" vs. "how do refunds
  work") that exact-match misses — at the cost of an embedding call and a
  small risk of returning a *slightly* wrong cached answer if the threshold
  is too loose.
- **Invalidation:** the hard part. If the underlying docs change (Phase 04's
  index gets re-built) or the prompt/model changes, cached answers can go
  stale or become inconsistent with a new system version. Common approach:
  version-tag cache entries with a doc-index version and a prompt-template
  version, and invalidate on either changing — not just a blind TTL.

This is exactly the layer Phase 08's production shell is the natural home
for, sitting between the FastAPI endpoint and the RAG/agent pipeline.

---

## 4. Safety & Guardrail Layers

Guardrails sit at two points: before the model sees the input, and before
the user sees the output.

```
  user input
      │
      ▼
┌─────────────────┐
│ INPUT guardrails  │  prompt-injection detection, PII scrubbing/flagging,
│                    │  off-topic/abuse classification, rate limiting
└────────┬───────────┘
         │
         ▼
    core pipeline (retrieval / agent / generation)
         │
         ▼
┌─────────────────┐
│ OUTPUT guardrails │  hallucination/faithfulness check (did the answer stay
│                    │  grounded in retrieved sources?), toxicity/safety
│                    │  classification, schema validation (Phase 02's
│                    │  Pydantic models — malformed structured output should
│                    │  never reach the caller), confidence threshold →
│                    │  escalate to human instead of guessing
└────────┬───────────┘
         │
         ▼
    response to user (or "let me connect you to a human")
```

- **Input side:** prompt-injection is the AI-specific threat classic system
  design doesn't have to think about — a malicious document *in the
  retrieved context* can contain instructions aimed at the model, not the
  user. Treat retrieved content as untrusted data, not as instructions, in
  how you structure the prompt.
- **Output side:** for RAG specifically, a faithfulness check (does the
  claim actually appear in the cited source, roughly what Phase 08's RAGAS
  `faithfulness` metric measures offline) can run online too, as a
  lightweight second LLM call or NLI model, gating whether the answer ships
  as-is or triggers a "low confidence" fallback.
- **Human-in-the-loop as a guardrail, not an afterthought:** Phase 05's
  `require_fetch_approval` pattern generalizes — any action with real-world
  consequences (a refund, an email sent, a ticket closed) above some
  risk/value threshold should require explicit approval, not just be logged
  after the fact.

---

## 5. The Monitoring Stack

"It works in the demo" and "it's still working in production, three prompt
changes later" are different claims, and the gap between them is this stack —
exactly what Phase 08 wires together.

```
┌───────────────────────────────────────────────────────────┐
│                        Every request                        │
└──────────────────────────┬────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │  tracing  │  │   cost    │  │   feedback    │
        │(Langfuse: │  │ tracking  │  │ (thumbs up/   │
        │ full req/ │  │(tokens ×  │  │  down, tied   │
        │ resp trace│  │  price)   │  │  to trace_id) │
        └──────────┘  └──────────┘  └──────────────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
              ┌───────────────────────────┐
              │   offline evaluation gate   │
              │ (RAGAS/DeepEval, run on a   │
              │  golden set on every deploy;│
              │  regression = block deploy) │
              └──────────────┬─────────────┘
                             ▼
                    alerting / dashboards
             (latency p50/p95, error rate, cost/day,
              feedback rate, eval-metric trend over time)
```

- **Tracing:** every request's full path — inputs, retrieved chunks,
  intermediate agent steps, final output, latency per stage — logged with a
  `trace_id`. Phase 08's `observability.py` does this via Langfuse's
  `@observe()`. Without this, debugging a bad answer means guessing.
- **Cost tracking:** token counts × current pricing, per request and rolled
  up daily — Phase 01's `client.stats.report()` at the individual-call level,
  aggregated at the service level in production.
- **Feedback loop:** user-facing thumbs up/down tied back to the trace_id
  (Phase 08's `/feedback` endpoint) — this is how a production eval set
  actually grows over time, rather than staying frozen at whatever the team
  thought to test before launch.
- **Offline evaluation as a CI gate:** Phase 08's `evaluation.py` — run
  RAGAS/DeepEval metrics against a golden set on every meaningful change
  (prompt, model, retrieval config) and fail the deploy if a metric regresses
  past a threshold. This is the AI-system equivalent of a test suite, and
  it's the answer to "how do you know a prompt change didn't make things
  worse" in an interview.
- **Drift detection:** watching the eval metrics and the online feedback
  rate *trend over time*, not just at launch — model providers change
  underlying models without always announcing it, and retrieved-corpus
  composition drifts as documents are added.

---

## 6. A Combined Reference Architecture

Putting Sections 1-5 together — this is the shape a strong final answer to
almost any "design a production AI system" prompt converges toward, and it
maps directly onto phases you've already built:

```
                              ┌──────────────┐
                              │   client UI   │  (Phase 08: streamlit_app.py,
                              └──────┬───────┘   or any HTTP client)
                                     │ HTTPS
                                     ▼
                    ┌────────────────────────────────┐
                    │   API layer (Phase 08: main.py)  │
                    │ auth · rate limiting · streaming  │
                    └───────────────┬────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                 ┌────────────────┐   ┌──────────────────┐
                 │ INPUT guardrails │   │  semantic + exact  │
                 │ (Part 2 §4)      │   │  cache (Part 2 §3) │──── hit ──► response
                 └────────┬─────────┘   └──────────────────┘
                          │ miss
                          ▼
        ┌──────────────────────────────────────────┐
        │      core pipeline — pick a spectrum point:  │
        │  naive / advanced / agentic RAG (Part 2 §1)   │
        │  single agent / crew / supervisor (Part 2 §2) │
        │  (Phases 03-06 build every piece of this)      │
        └───────────────────┬────────────────────────┘
                             ▼
                 ┌────────────────────┐
                 │  OUTPUT guardrails   │
                 │  (Part 2 §4)         │
                 └──────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │  tracing  │  │   cost    │  │   feedback    │   (Part 2 §5 —
        │(Langfuse) │  │ tracking  │  │  → eval set   │    Phase 08's
        └──────────┘  └──────────┘  └──────────────┘    observability.py)
                             │
                             ▼
                 response streamed to client
```

Layered offline, running continuously: the **evaluation gate**
(`evaluation.py`) blocking bad deploys, and **alerting/dashboards** watching
the metrics this whole stack produces trend over time.

In an interview, you don't need to draw every box — but being able to name
each layer, why it exists, and which trade-off from Part 1 justifies
including (or deliberately omitting) it is what separates "I've read about
RAG" from "I've built this."

---

## 7. Key Takeaways

- RAG is a spectrum (naive → advanced → agentic), not one architecture —
  advanced RAG (hybrid retrieval + re-ranking) is the right default for most
  production systems; reach for agentic RAG only when questions genuinely
  need multi-hop reasoning.
- Multi-agent patterns exist on a similar spectrum from a single ReAct loop
  to a full supervisor-routed crew — more agents means more latency and cost,
  so match the pattern to actual task structure, not to sound impressive.
- Caching is often the single highest-leverage cost/latency optimization in
  an LLM system, and its hard part is invalidation, not the cache lookup itself.
- Guardrails belong on both sides of the core pipeline — input (untrusted
  retrieved content, prompt injection) and output (faithfulness, schema
  validation, confidence-based escalation).
- A monitoring stack (tracing, cost, feedback, offline eval gate) is what
  turns "it worked in the demo" into "we'd know if it stopped working" — and
  it's exactly what Phase 08 builds, because it's the one thing every
  production AI system needs regardless of what the core pipeline does.
