# Project 4 — Recipe & Meal Planner

## Context

The AI Engineering Bootcamp repo already has three Practice Projects that chain
adjacent phases into one runnable project by importing that phase's real code
directly (never duplicating it), per `CLAUDE.md`'s documented pattern. The user
picked "Recipe & Meal Planner" as the next one to build: an intermediate,
deliberately-not-too-hard project for new learners, combining **Phase 2**
(structured extraction via `instructor`) and **Phase 4** (RAG via `RAGEngine`).

Concept: ingest a small personal collection of recipes, ask natural-language
questions about them with cited answers (Phase 4), match recipes against a
list of ingredients the user already has using their *extracted* structured
ingredient lists rather than semantic search (Phase 2), and generate a merged,
deduplicated shopping list across several chosen recipes (Phase 2 again, used
a second time for a different purpose — cleanup/merge, not just ingestion).

User confirmed: **script only**, no Streamlit UI — matches Project 1 & 2's
shape (a core module + `demo.py`), not Project 3's FastAPI/Docker shape. No
`evaluation.py` either (only Phase 8 and Project 3 have eval gates; this
project's scope doesn't call for one).

## Verified API surface (from reading the actual code, not memory)

```python
# Phase 4 rag_engine.py
class RAGEngine:
    def __init__(self, collection_name="rag_collection", persist_dir="./chroma_db", llm_model=LLM_MODEL): ...
    def index(self, source: str, use_parent_child: bool = True, reset: bool = False) -> dict: ...
    def ask(self, question: str, k: int = 5, strategy: Literal[...] = "hybrid", use_hyde: bool = False) -> RAGResponse: ...
    def reset(self) -> None: ...
# index() auto-detects source type: http(s):// -> WebBaseLoader, .pdf -> PyPDFLoader,
# .csv -> CSVLoader, else -> TextLoader. Single file path or URL, no directory walking.
# Upserts by id, so re-indexing the same fixed path is idempotent (no dupes).

# Phase 2 extractor.py
class DataExtractor:
    def __init__(self, model: str = "gpt-4o-mini"): ...
    def extract(self, text: str, output_model: Type[T], extra_instructions: str = "",
                examples: list[dict] | None = None, use_cot: bool = False,
                temperature: float = 0) -> T: ...
# extract() is the generic, reusable entry point (instructor validates the LLM's
# output straight into any Pydantic model you pass) — this is how a NEW
# extraction type gets added, with zero changes to Phase 2's own files.
```

## Design decisions (and why)

- **No separate `models.py`.** Project 1 inlines its own `TicketExtraction`
  model directly in `triage.py` under a `# ── This project's own schema ──`
  banner; four small models here is the same order of magnitude. Inline all
  of them into `meal_planner.py` to match that precedent rather than
  introducing a file split no other Practice Project uses.
- **Recipe sources are plain `.txt`/`.md` files only, not PDFs.** `RAGEngine`
  already handles PDFs for indexing, but the extraction step needs raw text
  separately, and a PDF-text-extraction path just for that adds complexity
  without teaching anything new. Document this as a deliberate scope choice
  in the README, and guard `add_recipe()` against URLs (`Path(source)` would
  otherwise fail with a confusing error).
- **`persist_dir` uses a `tempfile`-based path** (matching Project 2's
  precedent), not a repo-relative `./chroma_db`, so repeated dev runs don't
  pollute the repo or accumulate stale state.
- **`RecipeMealPlanner.reset()`** wraps `self.rag.reset()` *and* clears
  `self.recipes = {}` (the dict has no analog to Chroma's upsert). `demo.py`
  calls this once before indexing, rather than threading `reset=True`
  through every `add_recipe()` call (which would wipe prior recipes).
- **`add_recipe()` indexes first, then extracts**, and does not catch the
  extraction call's exceptions. If extraction fails, the recipe is
  answerable via `ask()` but invisible to `find_recipes_for_ingredients()`/
  `build_shopping_list()` — a real, documented limitation, not a bug to
  paper over with a try/except (matches this project's "not too hard" scope
  and Project 2's precedent of naming its own limitation in the README).
- **Ingredient matching is pure Python, deterministic, no LLM call** —
  bidirectional case-insensitive substring containment plus simple plural
  stripping, *not* semantic search. Good enough for "chicken" matching
  "chicken breast, diced," documented as not catching true synonyms
  ("scallion" vs "green onion").
- **`Ingredient.quantity` and `ShoppingListItem.combined_quantity` are free
  text (`str`)**, not numeric — recipe quantities ("2 cups," "3 cloves,"
  "salt to taste") aren't cleanly numeric, matching how Phase 2's own
  `Receipt`/`LineItem` precedent handles messy real-world fields via prompt
  instructions rather than a brittle type.

## Files to create

```
Project_4 Recipe & Meal Planner/
├── README.md
├── meal_planner.py          ← RecipeMealPlanner + inlined Pydantic models
│                                (Ingredient, RecipeExtraction, ShoppingListItem,
│                                ShoppingList) + RecipeMatch dataclass
├── demo.py                  ← single main(), mirrors Project 1/2's shape
├── requirements.txt         ← union of Phase 2 + Phase 4's requirements, minus streamlit
├── .env.example             ← OPENAI_API_KEY only
└── sample_recipes/
    ├── chicken_stir_fry.txt
    ├── spaghetti_bolognese.txt
    ├── veggie_tacos.txt
    ├── tomato_soup.txt
    └── pancakes.txt
```

Plus edits to `/Users/kumarshikhar/AI-Engineering-Bootcamp/README.md` (Repository
Structure tree + Practice Projects table + intro sentence, "these three" →
"these four") and `/Users/kumarshikhar/AI-Engineering-Bootcamp/CLAUDE.md`
(Project_1/2/3 list, Practice Projects intro, and a clause added to the
existing Phase 2/Phase 4 architecture-note bullets).

## `meal_planner.py` — method-by-method

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_4 RAG & Vector Databases"))
sys.path.insert(0, str(_REPO_ROOT / "Phase_2 Prompt Engineering"))
from rag_engine import RAGEngine          # Phase 4
from extractor import DataExtractor        # Phase 2

# ── This project's own schema ──────────────────────────────────────────
class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name as written, e.g. 'chicken breast, diced'")
    quantity: str | None = Field(None, description="Quantity + unit as written, e.g. '2 cups'")

class RecipeExtraction(BaseModel):
    title: str
    servings: int | None = None
    ingredients: list[Ingredient]
    instructions_summary: str = Field(description="A 1-3 sentence summary of how to cook it")
    cuisine: str | None = None
    total_time_minutes: int | None = None

class ShoppingListItem(BaseModel):
    name: str
    combined_quantity: str = Field(description="Merged quantity across all selected recipes, human-readable")

class ShoppingList(BaseModel):
    items: list[ShoppingListItem]

@dataclass
class RecipeMatch:
    source: str
    title: str
    have_count: int
    total_count: int
    match_ratio: float
    missing: list[str]

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

def _ingredient_matches(have: str, ingredient_name: str) -> bool:
    h, n = _normalize(have), _normalize(ingredient_name)
    if not h or not n:
        return False
    if h in n or n in h:
        return True
    return h.rstrip("s") == n.rstrip("s")

class RecipeMealPlanner:
    def __init__(self, collection_name="recipe_planner", persist_dir=None):
        persist_dir = persist_dir or os.path.join(tempfile.gettempdir(), "recipe_planner_db")
        self.rag = RAGEngine(collection_name=collection_name, persist_dir=persist_dir)
        self.extractor = DataExtractor()
        self.recipes: dict[str, RecipeExtraction] = {}

    # ── Indexing ──
    def add_recipe(self, source: str) -> RecipeExtraction:
        if source.startswith(("http://", "https://")):
            raise ValueError("add_recipe() only accepts local .txt/.md files, not URLs.")
        self.rag.index(source)
        text = Path(source).read_text(encoding="utf-8")
        recipe = self.extractor.extract(text, RecipeExtraction)
        self.recipes[source] = recipe
        return recipe

    def reset(self) -> None:
        self.rag.reset()
        self.recipes = {}

    # ── Query ──
    def ask(self, question: str):
        return self.rag.ask(question)

    # ── Planning ──
    def find_recipes_for_ingredients(self, have: list[str], min_match: float = 0.4) -> list[RecipeMatch]:
        matches = []
        for source, recipe in self.recipes.items():
            total = len(recipe.ingredients)
            if total == 0:
                continue
            missing = [ing.name for ing in recipe.ingredients
                       if not any(_ingredient_matches(h, ing.name) for h in have)]
            have_count = total - len(missing)
            ratio = have_count / total
            if ratio >= min_match:
                matches.append(RecipeMatch(source, recipe.title, have_count, total, ratio, missing))
        return sorted(matches, key=lambda m: m.match_ratio, reverse=True)

    def build_shopping_list(self, recipe_sources: list[str]) -> ShoppingList:
        lines = []
        for source in recipe_sources:
            recipe = self.recipes.get(source)
            if recipe is None:
                raise KeyError(f"No recipe indexed for '{source}'. Call add_recipe() first.")
            lines.append(f"# {recipe.title} (serves {recipe.servings})")
            for ing in recipe.ingredients:
                qty = f"{ing.quantity} " if ing.quantity else ""
                lines.append(f"- {qty}{ing.name}")
        blob = "\n".join(lines)
        return self.extractor.extract(
            blob, ShoppingList,
            extra_instructions=(
                "This is a combined ingredient list from multiple recipes, grouped by "
                "recipe under '#' headers. Merge duplicate or near-duplicate ingredients "
                "into one line, summing quantities in natural language when units match "
                "and listing both amounts together when they don't."
            ),
        )
```

## Build order

1. `sample_recipes/*.txt` — 5 real recipes with deliberate ingredient overlap
   (onion, garlic, salt) so matching/merging has something real to do.
2. `meal_planner.py` — sys.path wiring, inline schema, then
   `__init__` → `add_recipe()` → `reset()` → `ask()` →
   `find_recipes_for_ingredients()` → `build_shopping_list()`, each under a
   `# ── ... ──` comment banner. Run `python -m py_compile meal_planner.py`
   as soon as it parses (the standard cheap sanity check `CLAUDE.md` calls
   for on every Practice Project module).
3. `demo.py` — single `main()`: construct `RecipeMealPlanner`, `reset()`,
   loop `add_recipe()` over the 5 samples with progress prints, one `ask()`
   call, one `find_recipes_for_ingredients()` call with a deliberately
   partial ingredient list, one `build_shopping_list()` over 2-3 recipes,
   print each result labeled. Run it live (real API key) before writing the
   README so the README's example output reflects what actually comes back.
4. `requirements.txt` — union of Phase 2 + Phase 4's `requirements.txt`,
   dedup shared entries, drop `streamlit`.
5. `.env.example` — `OPENAI_API_KEY` only, with a one-line comment.
6. `README.md` — same heading order as the other three projects (`# Project
   4 — Recipe & Meal Planner` / `**Exercises:**` / `## What this project
   does` / `## Why this pairing` / `## Project structure` / `## Quick start`
   / `## API reference` / `## What this exercises from each phase` / `##
   Adapting this further`), with a named-limitation callout for the naive
   ingredient matcher and `add_recipe()`'s partial-failure behavior, and an
   explicit note that `sample_recipes/` is a deliberate deviation from
   Project 1/2's inline-string demo data.
7. Root `README.md` — insert `Project_4 Recipe & Meal Planner/` into the
   Repository Structure tree, add its row to the Practice Projects table,
   "these three" → "these four."
8. `CLAUDE.md` — add Project 4 to the Practice Projects list/intro, append a
   clause to the existing Phase 2 and Phase 4 architecture-note bullets.

## Verification

- `python -m py_compile meal_planner.py` and `demo.py` as each is written.
- Run `python demo.py` end-to-end from a fresh venv (`pip install -r
  requirements.txt`, real `.env` with `OPENAI_API_KEY`) and confirm: all 5
  recipes index without error, the `ask()` answer cites a real source, the
  ingredient-match results look sensible (correct recipes ranked higher,
  `missing` lists are accurate), and the shopping list actually merges
  overlapping ingredients across the selected recipes rather than just
  concatenating them.
- Re-run `demo.py` a second time against the same `persist_dir` to confirm
  `reset()` genuinely clears prior state (no duplicate/stale chunks, no
  leftover entries in `self.recipes`).
