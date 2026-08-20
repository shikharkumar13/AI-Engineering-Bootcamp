# Embedding_example.py
#
# A toy demonstration of "similar meaning = nearby points" — using
# hand-picked 2D coordinates instead of a real trained model, so you can
# see the core idea behind embeddings with zero downloads and zero
# dependencies. See Phase 00 Part 5, Section 8-9 for the full explanation.
#
# Run it:
#   python Embedding_example.py     (Windows)
#   python3 Embedding_example.py    (Mac)

# Each word gets a made-up (size, wildness) coordinate, in the same spirit
# as the diagram in the guide's Section 8.2. A REAL embedding model learns
# coordinates like these automatically from text, across thousands of
# dimensions — here, we're just hand-placing a few points to see the
# principle at work with numbers simple enough to sanity-check yourself.
words = {
    "cat":      (2, 3),
    "dog":      (3, 3),
    "wolf":     (3, 8),
    "mouse":    (1, 2),
    "elephant": (7, 7),
    "car":      (6, 0),
    "truck":    (8, 0),
    "bicycle":  (4, 0),
}


def distance(point_a, point_b):
    """
    How far apart two points are. Smaller = more similar.
    (This is just the Pythagorean theorem — the distance between two
    points on a grid. No calculus, no training math, just geometry.)
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def nearest_words(target_word, all_words, top_n=3):
    """Find the words whose points are closest to the target word's point."""
    target_point = all_words[target_word]

    distances = []
    for word, point in all_words.items():
        if word == target_word:
            continue
        distances.append((word, distance(target_point, point)))

    distances.sort(key=lambda pair: pair[1])  # closest first
    return distances[:top_n]


def print_full_grid(all_words):
    """Bonus: print every pairwise distance, so you can see the full picture."""
    print("\nFull distance table (smaller = more similar):")
    word_list = list(all_words.keys())
    header = "          " + "".join(f"{w[:6]:>8s}" for w in word_list)
    print(header)
    for w1 in word_list:
        row = f"{w1[:9]:10s}"
        for w2 in word_list:
            d = distance(all_words[w1], all_words[w2])
            row += f"{d:8.1f}"
        print(row)


if __name__ == "__main__":
    print("=" * 60)
    print("  Nearest neighbors (the embedding intuition in action)")
    print("=" * 60)

    for query in ["cat", "elephant", "car"]:
        print(f"\nWords most similar to '{query}':")
        for word, dist in nearest_words(query, words):
            print(f"  {word:10s}  (distance: {dist:.2f})")

    print_full_grid(words)

    print("\n" + "=" * 60)
    print("  Notice: animals cluster near animals, vehicles cluster near")
    print("  vehicles — nothing in the code says 'these are categories,'")
    print("  it emerges purely from the coordinates. That's the entire")
    print("  idea behind a real embedding, just scaled to thousands of")
    print("  dimensions learned from real text instead of 2 made up by hand.")
    print("=" * 60)
