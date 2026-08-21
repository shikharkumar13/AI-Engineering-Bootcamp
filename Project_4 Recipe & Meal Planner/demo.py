"""
demo.py — Recipe & Meal Planner walkthrough

Indexes the sample recipes in sample_recipes/, asks a question about them,
finds which recipes can be made with a partial pantry list, and builds a
merged shopping list across a couple of chosen recipes.

Run: python demo.py
"""

from pathlib import Path

from meal_planner import RecipeMealPlanner

SAMPLE_RECIPES_DIR = Path(__file__).parent / "sample_recipes"


def main():
    planner = RecipeMealPlanner()
    planner.reset()

    print("=" * 60)
    print("1. Indexing sample recipes")
    print("=" * 60)
    recipe_files = sorted(SAMPLE_RECIPES_DIR.glob("*.txt"))
    for path in recipe_files:
        recipe = planner.add_recipe(str(path))
        print(f"  + {recipe.title} ({len(recipe.ingredients)} ingredients, serves {recipe.servings})")

    print()
    print("=" * 60)
    print("2. Ask a question about the indexed recipes")
    print("=" * 60)
    question = "How do I make the spaghetti bolognese?"
    response = planner.ask(question)
    print(f"  Q: {question}")
    print(f"  A: {response.answer}")
    print(f"  Sources: {[s.source for s in response.sources]}")

    print()
    print("=" * 60)
    print("3. What can I cook with what I already have?")
    print("=" * 60)
    pantry = ["onion", "garlic", "salt", "black beans", "corn tortillas", "avocado"]
    print(f"  Pantry: {pantry}")
    matches = planner.find_recipes_for_ingredients(pantry, min_match=0.3)
    for m in matches:
        print(
            f"  - {m.title}: {m.have_count}/{m.total_count} ingredients on hand "
            f"({m.match_ratio:.0%}), missing: {m.missing}"
        )

    print()
    print("=" * 60)
    print("4. Build a shopping list for this week's picks")
    print("=" * 60)
    this_weeks_picks = [
        str(SAMPLE_RECIPES_DIR / "spaghetti_bolognese.txt"),
        str(SAMPLE_RECIPES_DIR / "tomato_soup.txt"),
    ]
    shopping_list = planner.build_shopping_list(this_weeks_picks)
    print("  Shopping list:")
    for item in shopping_list.items:
        print(f"    - {item.combined_quantity} {item.name}")


if __name__ == "__main__":
    main()
