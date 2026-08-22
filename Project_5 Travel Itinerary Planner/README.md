# Project 5 — Travel Itinerary Planner

**Exercises:** Phase 2 (Prompt Engineering) + Phase 5 (AI Agents & LangGraph)

## What this project does

Give it a destination and a number of days, and it builds a real,
structured day-by-day itinerary from live web research:

1. **Researches the destination**, via Phase 5's `ResearchAgent` (a ReAct
   agent with live web search): attractions, food, and practical
   logistics, in two separate research passes.
2. **Structures the findings** into a day-by-day itinerary, via Phase 2's
   `DataExtractor`, organizing scattered research notes into
   `day_number`-ordered activities with categories and sources.
3. **Answers one-off follow-up questions** about the destination directly
   through the agent, without going through extraction.

This is the only Practice Project so far that touches agents, and it's a
deliberate step up in difficulty from the others: live web search means
real, sometimes messy, real-world text to structure, not a fixed local
file.

## Why this pairing

An agent alone can research a destination and hand back a wall of prose;
it's good at open-ended search and synthesis, not at producing a clean,
consistently-shaped `day_number: 1..N` structure. Structured extraction
alone has nothing to structure without something doing the research
first. Phase 5 does the searching and reading, Phase 2 turns whatever it
finds into something an itinerary app could actually render.

## Project structure

```
Project_5 Travel Itinerary Planner/
├── README.md
├── PLAN.md                  ← the implementation plan this project was built from
├── itinerary_planner.py      ← TravelPlanner: composes ResearchAgent + DataExtractor
├── demo.py
├── requirements.txt
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and TAVILY_API_KEY

python demo.py
```

`TAVILY_API_KEY` isn't optional here the way some keys are in other
Practice Projects: without it, `web_search` returns a literal error
string to the agent instead of raising, so the itinerary will still get
built, but from the model's own general knowledge instead of live
research, defeating the point of the project. Get a free-tier key at
[tavily.com](https://tavily.com).

## API reference

```python
from itinerary_planner import TravelPlanner

planner = TravelPlanner()

# Two research passes (attractions/food, then practical logistics),
# then structured into a day-by-day Itinerary
itinerary = planner.plan_trip("Kyoto, Japan", num_days=3, interests=["history", "food"])

for day in itinerary.days:
    print(f"Day {day.day_number}: {day.theme}")
    for activity in day.activities:
        print(f"  [{activity.category}] {activity.name}: {activity.description}")

# A second call for the same destination is safe and starts fresh —
# see "Known limitations" for why this needed a deliberate fix
itinerary_2 = planner.plan_trip("Kyoto, Japan", num_days=2, interests=["nature"])

# One-off follow-up question, no extraction step
answer = planner.ask_about_destination("Kyoto, Japan", "What's the best way to get around without a car?")
```

Every field on `Itinerary` is real, LLM-researched content from the two
`.research()` calls, so exact attraction names, hours, and prices will
vary run to run and should always be double-checked before an actual
trip. The output *shape* is fixed, though:

```
Itinerary(
    destination="Kyoto, Japan",
    num_days=3,
    days=[
        DayPlan(day_number=1, theme="...", activities=[
            Activity(name="...", description="...", category="sightseeing", estimated_time="..."),
            ...
        ]),
        DayPlan(day_number=2, ...),
        DayPlan(day_number=3, ...),
    ],
    practical_tips=["...", "..."],
    sources=["https://...", "..."],
)
```

## What this exercises from each phase

| From | What's reused |
|---|---|
| Phase 5 | `ResearchAgent.research()`, called twice per `plan_trip()` for two different research goals, plus once more for `ask_about_destination()` — live web search, ReAct tool-calling, per-call notes capture |
| Phase 2 | `DataExtractor.extract()`, the generic entry point, turning two research passes' worth of prose and notes into one structured `Itinerary` |

## Known limitations worth knowing

- **`plan_trip()` supports 1-5 day trips.** Each call does exactly two
  `.research()` passes regardless of `num_days`, bounded by Phase 5's own
  `MAX_AGENT_STEPS=12` per call. Longer trips would get noticeably
  shallower coverage per day rather than more research, so the cap raises
  a clear error instead of silently under-delivering.
- **A research pass can come back with an empty result if the agent's step
  budget runs out mid-search.** `TravelPlanner` falls back to the raw
  research notes when this happens, so `plan_trip()` shouldn't outright
  fail, but a pass that hits this edge case will produce thinner findings
  for the extractor to work with.
- **Every `.research()` call needs a unique `thread_id`.** Phase 5's agent
  persists conversation state per `thread_id` for the life of the process
  (`MemorySaver`), and *appends* to a thread's history on reuse rather than
  starting fresh. `TravelPlanner` keeps an internal call counter so every
  call, across both `plan_trip()` and `ask_about_destination()`, gets a
  fresh thread automatically. This was a real bug caught while designing
  this project, not just a theoretical concern: a naive
  `thread_id=f"{destination}-{num_days}"` scheme would silently leak one
  call's research into the next call for the same destination.
- **Live research quality depends entirely on what's on the web that day**
  and what `TAVILY_API_KEY`'s search returns. There's no fact-checking step
  beyond what the ReAct agent itself does; treat the output as a starting
  point, not a booked itinerary.

## Adapting this further

- Add a real fact-check pass: re-query `web_search` for any activity the
  extractor produced, to catch hallucinated names or hours.
- Cache `plan_trip()` results per destination so repeat requests don't
  re-run live research (and re-spend the API budget) for the same trip.
- Wrap `TravelPlanner` in a small FastAPI layer, the way Project 3 wraps
  Phase 6's crew, to turn this into a callable service instead of a script.
