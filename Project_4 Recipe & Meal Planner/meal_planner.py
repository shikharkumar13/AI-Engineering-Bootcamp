"""
meal_planner.py — RecipeMealPlanner

Combines Phase 4's RAGEngine (RAG over a personal recipe collection) and
Phase 2's DataExtractor (structured extraction) into one project: ask
cited questions about your recipes, find which ones you can cook with
ingredients you already have, and generate a merged shopping list.

Imports both phases' code directly rather than duplicating it.
"""

import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

# ── Wire up sibling phase folders so we can import their code directly ────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_4 RAG & Vector Databases"))
sys.path.insert(0, str(_REPO_ROOT / "Phase_2 Prompt Engineering"))

from rag_engine import RAGEngine   # Phase 4
from extractor import DataExtractor  # Phase 2


# ── This project's own schema ──────────────────────────────────────────
class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name as written, e.g. 'chicken breast, diced'")
    quantity: str | None = Field(None, description="Quantity and unit as written, e.g. '2 cups'")


class RecipeExtraction(BaseModel):
    title: str = Field(description="The recipe's name")
    servings: int | None = Field(None, description="How many people this recipe serves")
    ingredients: list[Ingredient]
    instructions_summary: str = Field(description="A 1-3 sentence summary of how to cook it")
    cuisine: str | None = Field(None, description="Cuisine style, e.g. 'Italian', if stated or obvious")
    total_time_minutes: int | None = Field(None, description="Total time in minutes, if stated")


class ShoppingListItem(BaseModel):
    name: str = Field(description="Ingredient name, deduplicated and merged where possible")
    combined_quantity: str = Field(
        description="Merged quantity across all selected recipes, as a human-readable string"
    )


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
    """Bidirectional, case-insensitive substring match with simple plural
    stripping — deliberately not semantic search. Catches 'chicken' inside
    'chicken breast, diced', but not true synonyms like 'scallion' vs
    'green onion' (a documented limitation, see README)."""
    h, n = _normalize(have), _normalize(ingredient_name)
    if not h or not n:
        return False
    if h in n or n in h:
        return True
    return h.rstrip("s") == n.rstrip("s")


class RecipeMealPlanner:
    def __init__(self, collection_name: str = "recipe_planner", persist_dir: str | None = None):
        persist_dir = persist_dir or os.path.join(tempfile.gettempdir(), "recipe_planner_db")
        self.rag = RAGEngine(collection_name=collection_name, persist_dir=persist_dir)
        self.extractor = DataExtractor()
        self.recipes: dict[str, RecipeExtraction] = {}

    # ── Indexing ──────────────────────────────────────────────────────
    def add_recipe(self, source: str) -> RecipeExtraction:
        """Indexes a recipe for RAG and extracts its structured ingredient
        list. Only local .txt/.md files are supported — a URL would index
        fine via RAGEngine but has no local text to extract structured
        data from, so it's rejected up front with a clear error instead of
        a confusing one from Path()."""
        if source.startswith(("http://", "https://")):
            raise ValueError("add_recipe() only accepts local .txt/.md files, not URLs.")

        self.rag.index(source)
        text = Path(source).read_text(encoding="utf-8")
        recipe = self.extractor.extract(text, RecipeExtraction)
        self.recipes[source] = recipe
        return recipe

    def reset(self) -> None:
        """Wipes both the vector store and the extracted-recipe registry.
        Call this once before re-indexing, rather than passing
        reset=True into individual add_recipe() calls, which would wipe
        prior recipes instead of just clearing stale state."""
        self.rag.reset()
        self.recipes = {}

    # ── Query ─────────────────────────────────────────────────────────
    def ask(self, question: str):
        """Thin passthrough to RAGEngine.ask() — cited, retrieval-grounded
        answers over every recipe indexed so far."""
        return self.rag.ask(question)

    # ── Planning ──────────────────────────────────────────────────────
    def find_recipes_for_ingredients(
        self, have: list[str], min_match: float = 0.4
    ) -> list[RecipeMatch]:
        """Ranks indexed recipes by how many of their ingredients are
        covered by `have`, using exact structured data rather than
        semantic search over the raw text."""
        matches = []
        for source, recipe in self.recipes.items():
            total = len(recipe.ingredients)
            if total == 0:
                continue
            missing = [
                ing.name
                for ing in recipe.ingredients
                if not any(_ingredient_matches(h, ing.name) for h in have)
            ]
            have_count = total - len(missing)
            ratio = have_count / total
            if ratio >= min_match:
                matches.append(
                    RecipeMatch(source, recipe.title, have_count, total, ratio, missing)
                )
        return sorted(matches, key=lambda m: m.match_ratio, reverse=True)

    def build_shopping_list(self, recipe_sources: list[str]) -> ShoppingList:
        """Combines the selected recipes' ingredient lists into one text
        blob, grouped by recipe, then reuses DataExtractor.extract() a
        second time — not for initial ingestion this time, but to merge
        and deduplicate quantities across recipes."""
        lines = []
        for source in recipe_sources:
            recipe = self.recipes.get(source)
            if recipe is None:
                raise KeyError(f"No recipe indexed for '{source}'. Call add_recipe() first.")
            serves = f" (serves {recipe.servings})" if recipe.servings else ""
            lines.append(f"# {recipe.title}{serves}")
            for ing in recipe.ingredients:
                qty = f"{ing.quantity} " if ing.quantity else ""
                lines.append(f"- {qty}{ing.name}")
        blob = "\n".join(lines)

        return self.extractor.extract(
            blob,
            ShoppingList,
            extra_instructions=(
                "This is a combined ingredient list from multiple recipes, grouped by "
                "recipe under '#' headers. Merge duplicate or near-duplicate ingredients "
                "into one line, summing quantities in natural language when units match "
                "and listing both amounts together when they don't."
            ),
        )
