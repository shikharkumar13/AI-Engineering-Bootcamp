# Project 4 — Recipe & Meal Planner

**Exercises:** Phase 2 (Prompt Engineering) + Phase 4 (RAG & Vector Databases)

## What this project does

A small tool for a personal recipe collection, with three capabilities:

1. **Ask questions about your recipes**, with cited answers, via Phase 4's
   `RAGEngine`: "How do I make the spaghetti bolognese?"
2. **Find what you can cook tonight** from ingredients you already have, by
   matching your pantry list against each recipe's *structured* ingredient
   list (Phase 2's `DataExtractor`), not by semantic search.
3. **Build a shopping list** for a few chosen recipes, merging duplicate or
   overlapping ingredients across them into one clean list, using Phase 2's
   extractor a second time, for cleanup rather than initial ingestion.

## Why this pairing

RAG alone can tell you *about* a recipe, but it can't reliably answer
"what can I make with chicken and rice" as an exact-match question, because
retrieval is semantic, not structured. Structured extraction alone gives you
clean ingredient lists, but no natural-language Q&A over free text.
Combining them gives each capability to the piece of the pipeline that's
actually good at it: Phase 4 for open-ended questions with citations, Phase
2 for anything that needs an exact, queryable shape.

## Project structure

```
Project_4 Recipe & Meal Planner/
├── README.md
├── PLAN.md               ← the implementation plan this project was built from
├── meal_planner.py        ← RecipeMealPlanner: composes RAGEngine + DataExtractor
├── demo.py
├── requirements.txt
├── .env.example
└── sample_recipes/        ← 5 real recipe .txt files, shipped for demo.py to use
    ├── chicken_stir_fry.txt
    ├── spaghetti_bolognese.txt
    ├── veggie_tacos.txt
    ├── tomato_soup.txt
    └── pancakes.txt
```

`sample_recipes/` is a deliberate deviation from Project 1 and Project 2,
which inline their demo data as Python string constants inside `demo.py`.
Here, seeing an actual recipe `.txt` file matters: it's what good extraction
input looks like, and "multiple recipes merge into one shopping list" only
feels real with more than one file to point at.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY

python demo.py
```

## API reference

```python
from meal_planner import RecipeMealPlanner

planner = RecipeMealPlanner()
planner.reset()  # clears both the vector store and the extracted-recipe registry

# Index a recipe — indexes it for RAG and extracts its structured ingredient list
recipe = planner.add_recipe("sample_recipes/chicken_stir_fry.txt")

# Ask a cited question over everything indexed so far
response = planner.ask("How do I make the spaghetti bolognese?")
print(response.answer, [s.source for s in response.sources])

# Rank indexed recipes by how many ingredients you already have
matches = planner.find_recipes_for_ingredients(["onion", "garlic", "salt"])

# Merge ingredient lists across a few chosen recipes into one shopping list
shopping_list = planner.build_shopping_list([
    "sample_recipes/spaghetti_bolognese.txt",
    "sample_recipes/tomato_soup.txt",
])
```

`find_recipes_for_ingredients(["onion", "garlic", "salt"])` is pure Python
(no LLM call), so its output is exact and reproducible against the shipped
`sample_recipes/` files:

```
- Tomato Soup: 3/8 ingredients on hand (38%), missing: ['crushed tomatoes', 'vegetable broth', 'olive oil', 'heavy cream', 'crusty bread']
- Spaghetti Bolognese: 3/9 ingredients on hand (33%), missing: ['ground beef', 'crushed tomatoes', 'olive oil', 'dried oregano', 'spaghetti', 'parmesan']
```

`build_shopping_list()` does call an LLM to merge the two recipes' lists, so
exact wording will vary run to run, but a correct merge of these two
specific files' raw ingredients looks like this (4 cans crushed tomatoes
from 2+2, 6 cloves garlic from 3+3, and so on):

```
Shopping list:
    - 4 cans crushed tomatoes
    - 2 onions, diced
    - 6 cloves garlic, minced
    - 4 tbsp olive oil
    - 2 tsp salt
    - 1 lb ground beef
    - 1 tsp dried oregano
    - 12 oz spaghetti
    - grated parmesan, for serving
    - 2 cups vegetable broth
    - 1/2 cup heavy cream
    - crusty bread, for serving
```

## What this exercises from each phase

| From | What's reused |
|---|---|
| Phase 4 | `RAGEngine.index()` / `.ask()` / `.reset()` in full — hybrid retrieval, chunking, cross-encoder re-ranking, cited answers |
| Phase 2 | `DataExtractor.extract()`, the generic entry point, called twice for two different purposes: once per recipe to get a structured `RecipeExtraction`, and once over a combined ingredient blob to get a merged `ShoppingList` |

## Known limitations worth knowing

- **Ingredient matching is pure Python, not semantic.** `find_recipes_for_ingredients()`
  uses case-insensitive substring containment plus simple plural stripping
  (`"chicken"` matches `"chicken breast, diced"`), not embeddings or an LLM
  call. It will not catch true synonyms like `"scallion"` vs `"green onion"`.
  This is intentional: exact, deterministic, free matching over the
  structured data, rather than another LLM call for something Python can do
  directly.
- **`add_recipe()` can leave a recipe half-indexed.** It indexes for RAG
  first, then extracts structured data. If the extraction call fails (rate
  limit, validation error), the recipe is answerable via `ask()` but
  invisible to `find_recipes_for_ingredients()`/`build_shopping_list()`
  until `add_recipe()` is called again successfully. It's not caught and
  silently retried, on purpose, so a failure is loud rather than hidden.
- **Recipe sources are local `.txt`/`.md` files only, not PDFs or URLs.**
  `RAGEngine` itself can index PDFs and URLs, but the structured-extraction
  step needs raw text separately, and adding PDF-text-extraction just for
  that would add complexity without teaching anything new for this project.

## Adapting this further

- Swap the naive ingredient matcher for embedding similarity between
  pantry items and recipe ingredients, to catch synonyms the substring
  match misses.
- Add unit-aware quantity math (a real "2 cups + 500 mL" converter) instead
  of leaving quantity merging to the LLM's judgment.
- Wrap `RecipeMealPlanner` in a small FastAPI layer, the way Project 3 wraps
  Phase 6's crew, to turn this into a callable service instead of a script.
