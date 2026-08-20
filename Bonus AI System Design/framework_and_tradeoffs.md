# Bonus, Part 1 — AI System Design: Framework & Trade-off Vocabulary

> **Prerequisites:** Phases 01–08 complete, or at least skimmed. This isn't a
> ninth phase with a project to build — it's the vocabulary and structure you
> need to talk about everything you've already built, in the specific format
> an AI Engineer interview loop expects.
> **What you'll learn:** A repeatable 7-step framework for any "design an AI
> system" interview question; the 8 trade-off dimensions that come up in
> nearly every one of those questions; and the back-of-envelope latency/cost
> math to back your answers with numbers instead of vibes.
> **Time:** 1-2 hours to read, then practice out loud against Part 2's
> architecture patterns and real prompts.

---

## Table of Contents

1. [Why System Design Interviews Look Different for AI](#1-why-system-design-interviews-look-different-for-ai)
2. [The 7-Step Framework](#2-the-7-step-framework)
3. [The 8 Core Trade-off Dimensions](#3-the-8-core-trade-off-dimensions)
4. [Back-of-Envelope Math](#4-back-of-envelope-math)
5. [A Worked Example](#5-a-worked-example)
6. [Key Takeaways](#6-key-takeaways)
7. [Practice Prompts](#7-practice-prompts)

---

## 1. Why System Design Interviews Look Different for AI

A classic system design interview ("design Twitter", "design a URL shortener")
is mostly about **data at scale**: sharding, replication, consistency,
caching. Those skills still matter here, but an AI Engineer design round adds
a second axis that classic system design doesn't have: **the model itself is
a black box with a cost, a latency, and an error rate that vary with every
input.**

That changes what a strong answer sounds like. You're no longer just
reasoning about "how do we handle 10K requests/sec" — you're also reasoning
about:

- What happens when the model is *confidently wrong* (not just slow or down)
- How you'd know if quality silently degraded after a prompt or model change
- Where a human needs to be in the loop, and where they explicitly shouldn't be
- Why the "best" architecture (most retrieval steps, biggest model, most
  agent hops) is very often the wrong answer, because cost and latency compound

Every phase in this repo built one piece of the puzzle — a client (Phase 01),
structured output (Phase 02), orchestration (Phase 03), retrieval (Phase 04),
agents (Phase 05), multi-agent pipelines (Phase 06), a fine-tune (Phase 07),
and a production shell (Phase 08). This page is about combining those pieces
under interview conditions: 30-45 minutes, a vague prompt, and an interviewer
who wants to see *how you think*, not a finished architecture diagram.

---

## 2. The 7-Step Framework

Use this shape for almost any "design an AI-powered X" prompt. Say the step
out loud as you move through it — interviewers are grading your process, not
just your final diagram.

```
1. Clarify            →  what does this system actually need to do?
2. Define success      →  how would we know it's working?
3. Sketch the naive design  →  simplest thing that could plausibly work
4. Name the trade-offs      →  where does the naive design break, and why?
5. Design for production    →  the real architecture, built from Part 2's patterns
6. Do the math               →  latency budget, cost per request, throughput
7. Discuss failure & iteration →  what breaks first, how do you find out, what do you do next
```

### 2.1 Clarify

Never start drawing boxes on hearing "design a customer support AI." Ask (or
state your assumptions if the interviewer wants you to drive):

- **Functional scope:** What can it actually answer? Does it need to take
  actions (refunds, ticket creation) or only answer questions?
- **Non-functional targets:** What's the latency SLA (interactive chat ≈
  <2-3s to first token; async report generation ≈ minutes is fine)? Expected
  QPS? Is this a B2C product with unpredictable traffic or an internal tool
  with known usage?
- **Data:** Where does the knowledge live? How fresh does it need to be?
  Structured (a database) or unstructured (docs, PDFs, tickets)?
- **Constraints:** Cost ceiling per request? Regulatory/PII constraints? Must
  it run on-prem, or is a hosted API fine?

This single step is the biggest differentiator between junior and senior
answers. A junior candidate starts drawing a RAG pipeline immediately. A
senior candidate spends 3-5 minutes here and often discovers the "AI" part
isn't even the hard part of the problem.

### 2.2 Define success

Before designing, state how you'd know the system works — this previews
Phase 08's evaluation gate and Part 2's monitoring stack:

- **Offline:** a labeled eval set, RAGAS-style metrics (faithfulness,
  relevancy) if it's RAG, task-completion rate if it's agentic
- **Online:** thumbs up/down feedback rate, escalation-to-human rate, task
  success rate, latency percentiles
- **Guardrail metrics:** hallucination rate, refusal rate, PII leak rate —
  metrics you watch even if they're not the primary success metric

### 2.3 Sketch the naive design

Draw the simplest version first, even though you know it's insufficient —
this is your baseline and your fallback if the real design goes over time:

```
User query → single LLM call (with the whole knowledge base stuffed
             into context, if it fits) → response
```

or, if retrieval is obviously needed:

```
User query → embed → vector search (top-k) → stuff into prompt → LLM → response
```

This is "naive RAG" from Part 2, Section 2. Say explicitly why it's a
starting point, not the answer: no re-ranking, no query understanding, no
caching, no safety layer, no observability.

### 2.4 Name the trade-offs

Walk through Section 3 below and call out which 2-3 dimensions matter most
*for this specific prompt*. A customer-support bot cares enormously about
hallucination-vs-coverage and latency-vs-quality; an internal research tool
cares much more about freshness-vs-cost and build-vs-buy. Naming which
trade-offs matter *and why for this use case* is worth more than listing all
eight generically.

### 2.5 Design for production

Now layer in the real architecture, drawing from Part 2's patterns:
retrieval strategy (hybrid + re-ranking, or agentic retrieval if multi-hop
reasoning is needed), a caching layer, input/output guardrails, and an
observability layer. Reference this repo's own phases as evidence you've
actually built these pieces: "the retrieval layer here is what Phase 04
builds — hybrid BM25 + vector search merged with RRF, then cross-encoder
re-ranked."

### 2.6 Do the math

See Section 4. Even rough numbers ("at 50 req/s and ~800ms p50 per request,
we need roughly 40 concurrent LLM calls in flight, which is why we'd want a
connection pool and a queue, not synchronous fan-out") show you can reason
quantitatively, not just architecturally.

### 2.7 Discuss failure & iteration

Every design has a weakest link. Name it: "the vector index is a single
point of staleness — if the source docs change and we don't re-index, we
confidently serve wrong answers with no error signal." Then say how you'd
detect it (freshness metric, scheduled re-index job, canary queries) and how
you'd iterate post-launch (feedback loop → eval set growth → prompt/model
regression testing, which is exactly what Phase 08's `evaluation.py` gate does).

---

## 3. The 8 Core Trade-off Dimensions

These recur across almost every AI system design question. You don't need to
mention all eight in every answer — pick the 2-3 that are load-bearing for
the specific prompt, per step 2.4 above.

| # | Dimension | The tension | Where you've already seen it |
|---|---|---|---|
| 1 | **Latency vs. quality** | More retrieval steps, bigger models, more agent hops = better answers but slower ones | Phase 04's re-ranking step; Phase 05's multi-step ReAct loop |
| 2 | **Cost vs. quality** | GPT-4-class quality costs 10-20x a smaller model per token | Phase 01's cost tracking; Phase 07's fine-tune-a-small-model alternative |
| 3 | **Freshness vs. latency/cost** | Real-time re-indexing or re-computation is expensive; stale caches are cheap but wrong | Phase 04's index vs. Phase 08's caching layer |
| 4 | **Accuracy vs. coverage** | Strict guardrails (refuse if not confident) cut hallucination but also cut how many questions get answered at all | Phase 08's `/chat` endpoint returning sources vs. refusing |
| 5 | **Determinism vs. creativity** | Low temperature = consistent, repeatable, sometimes robotic; high temperature = varied, sometimes wrong | Phase 02's structured extraction (near-zero temp) vs. Phase 06's content writer agent (higher temp) |
| 6 | **Build vs. buy** | Self-hosted vector DB/model vs. managed service — control and cost-at-scale vs. speed to ship | Phase 04's local ChromaDB vs. a managed vector DB in production |
| 7 | **Monolithic call vs. pipeline/multi-agent** | One big prompt doing everything is simple and fast but hard to debug and improve piece-by-piece; a pipeline of specialized steps is slower and more expensive but each piece is testable | Phase 06's research→write→edit→publish crew vs. a single "write me content" prompt |
| 8 | **Synchronous vs. streaming/async UX** | Users perceive streamed tokens as much faster than the same total latency delivered all at once; async (job + webhook/poll) fits long-running agent tasks better than a blocking request | Phase 01's streaming support; Phase 08's `/chat/stream` endpoint |

A useful interview habit: for whichever dimension you pick, state the two
extremes and where you're deliberately landing, and why. "We could go fully
agentic with unbounded tool calls for maximum answer quality, but for a
support bot with a 3-second latency SLA, we cap it at 2 tool calls and fall
back to 'let me connect you to a human' — trading some coverage for a
latency guarantee we can actually hit."

---

## 4. Back-of-Envelope Math

Interviewers aren't asking for precision — they're checking you can turn
"is this fast/cheap enough" into an actual calculation instead of a guess.
Two calculations come up constantly: **latency budget** and **cost per
request**.

### 4.1 Latency budget

Break the end-to-end request into its components and give each a rough
number. For a RAG chat endpoint with a ~2.5s p50 target:

```
Network round-trip (client ↔ server)         ~50ms
Auth / rate-limit check                      ~5ms
Query embedding (local model)                ~20ms
Vector search (top-20)                       ~30ms
Cross-encoder re-rank (20 → 5)               ~80ms
Prompt assembly                              ~5ms
LLM generation (streamed, time-to-first-token)  ~600-900ms
                                              ─────────────
                                    Total to first token:  ~0.8-1.1s
```

The LLM call dominates — this is true in almost every AI system, which is
why streaming (perceived latency) and prompt/output length (actual latency)
are the highest-leverage things to optimize, not the retrieval pipeline
around it.

### 4.2 Cost per request

Rough formula:

```
cost_per_request ≈ (input_tokens × input_price_per_token)
                  + (output_tokens × output_price_per_token)
                  + embedding_cost (often negligible if run locally)
```

Worked example at illustrative pricing (check current provider pricing —
these move fast, that's the point of doing the math live rather than
memorizing a number):

```
Input:  ~1,500 tokens (system prompt + 5 retrieved chunks + question)
Output: ~300 tokens
Price:  $2.50 / 1M input tokens, $10 / 1M output tokens  (illustrative)

Cost = (1,500 × $2.50/1,000,000) + (300 × $10/1,000,000)
     = $0.00375 + $0.003
     = ~$0.0068 per request
```

At 100K requests/day, that's ~$680/day — a number worth saying out loud,
because it's the kind of figure that changes an architecture decision (e.g.
"that's why we cache repeated questions instead of re-generating," which is
exactly Part 2 Section 3's caching layer).

### 4.3 Throughput and concurrency

```
required_concurrent_requests ≈ requests_per_second × avg_latency_seconds
```

At 50 req/s and 900ms average latency: ~45 requests in flight at any moment.
This is the number that tells you whether you need connection pooling, a
request queue, and rate limiting on the upstream LLM provider (Phase 08's
`slowapi` rate limiter exists for exactly this reason) — versus whether
synchronous, one-at-a-time calls are fine for an internal tool with 2 req/min.

---

## 5. A Worked Example

**Prompt:** "Design an AI system that lets internal support agents ask
natural-language questions over our product documentation and get an
answer with the SLA of a live chat (sub-3-second first response)."

Walking the framework briefly:

1. **Clarify:** Internal tool → predictable, moderate traffic (not
   viral-spike shaped). Docs update a few times a week, not per-second. No
   PII in the docs themselves, but support agents might paste customer
   questions containing PII into the query.
2. **Success:** Offline eval set of 50-100 real support questions with
   known-good answers, scored with RAGAS faithfulness + answer relevancy.
   Online: thumbs up/down per answer, escalation rate to "I don't know."
3. **Naive design:** Single OpenAI call with the whole doc set stuffed into
   context — rejected immediately, docs exceed context window and this has
   no source citations (support agents need to verify answers).
4. **Trade-offs that matter here:** accuracy vs. coverage (a wrong answer to
   a support agent becomes a wrong answer to a customer — bias toward
   refusing over guessing), and freshness vs. cost (docs change a few times
   a week, so nightly re-indexing is more than sufficient — no need for
   real-time indexing complexity).
5. **Production design:** Phase 04's hybrid RAG (BM25 + vector + re-rank) +
   an input guardrail that strips/flags PII before it's embedded or logged +
   a confidence threshold below which the system says "I'm not confident,
   escalate to a human" rather than guessing.
6. **Math:** ~1s p50 latency budget is achievable per Section 4.1 without an
   agent loop — a single retrieve-then-generate pass is enough, which also
   keeps cost near the $0.007/request figure above.
7. **Failure & iteration:** weakest link is stale docs after a release;
   mitigate with a nightly re-index job plus a "docs last updated" timestamp
   surfaced in the UI so agents know how fresh the answer is. Feed
   thumbs-down answers back into the eval set weekly.

Notice this never needed an agent, multi-agent crew, or fine-tune — a strong
answer is often "here's the simplest architecture that meets the actual
requirements," not "here's every pattern I know."

---

## 6. Key Takeaways

- The framework's first two steps (clarify, define success) are where most
  candidates lose points by skipping straight to architecture — spend real
  time there.
- Trade-offs are only useful named *in context*: which 2-3 actually bind for
  this prompt, and where you're deliberately landing on each.
- Numbers beat adjectives. "That should be fast enough" is weaker than a
  30-second back-of-envelope latency or cost calculation, even a rough one.
- The best answer is usually the simplest architecture that meets the stated
  requirements — not the most sophisticated one you know how to draw.

---

## 7. Practice Prompts

Run the 7-step framework out loud (or written, timed to 30 minutes) against
each of these. Part 2 gives you the architecture vocabulary to fill in step 5:

1. Design an AI code-review assistant that comments on pull requests.
2. Design a system that answers questions about a company's internal
   financial reports, where wrong numbers are unacceptable.
3. Design a multi-language customer support system serving 5 languages with
   a single shared knowledge base.
4. Design an AI system that triages and routes 10,000 inbound support
   tickets per day to the right team, with a human able to override any
   routing decision.
5. Design a system where an AI agent is allowed to actually issue refunds
   up to $50 autonomously, and must escalate above that.
