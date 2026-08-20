"""
demo.py — Phase 04: RAG & Vector Databases

Run: python demo.py  (no Streamlit needed — plain terminal output)

Demos:
  1. Embeddings — create, compare, visualize similarity
  2. ChromaDB   — index, search, filter
  3. Retrieval strategies side-by-side comparison
  4. Cross-encoder re-ranking
  5. HyDE retrieval
  6. Full RAG pipeline with citations
"""

import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DIVIDER = "═" * 62

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# Sample document for all demos
SAMPLE_DOC = """
The Transformer Architecture: Technical Deep-Dive
==================================================

Attention Mechanism
-------------------
The transformer attention mechanism computes three projections from the input:
Query (Q), Key (K), and Value (V) vectors. The attention score between two
positions is: softmax(QK^T / sqrt(d_k)) * V.

The scaling factor sqrt(d_k) prevents the dot products from becoming too large
in high-dimensional spaces, which would push softmax into regions with near-zero
gradients and slow down learning.

Multi-head attention runs h=8 parallel attention operations with different learned
weight matrices. Each "head" can attend to information from different representation
subspaces. The outputs are concatenated and projected back to the model dimension.

Positional Encoding
-------------------
Self-attention is permutation-invariant — it treats the input as a set, not a
sequence. Without positional information, the model cannot distinguish "cat sat
on mat" from "mat on sat cat". Positional encodings are added to the input
embeddings to inject sequence order.

The original paper used sinusoidal functions: PE(pos, 2i) = sin(pos/10000^(2i/d_model)).
Modern models use learned positional embeddings (GPT) or rotary positional encodings
(RoPE, used in LLaMA and Mistral).

Training Scale
--------------
GPT-3 was trained on 45TB of web text (filtered Common Crawl, Wikipedia, books).
It has 175 billion parameters and was trained on ~10,000 NVIDIA V100 GPUs for
several weeks. The training cost was estimated at $4-12 million USD.

The key finding from scaling laws (Kaplan et al., 2020): model performance follows
predictable power laws as a function of model size, data size, and compute budget.
Chinchilla (Hoffmann et al., 2022) showed that most LLMs were undertrained relative
to their parameter count — the optimal is roughly 20 tokens per parameter.

Inference Optimization
----------------------
Key-Value (KV) caching stores the K and V projections from the attention mechanism
for all previous tokens. Without it, each new token would require recomputing
attention over the entire context. With KV caching, only the new token's projections
need to be computed. This reduces inference from O(n^2) to O(n) per step.

Flash Attention (Dao et al., 2022) reorganizes the attention computation to reduce
memory bandwidth usage and enable training on much longer sequences. It is now
standard in all major frameworks.

Limitations
-----------
The quadratic attention complexity (O(n^2) in sequence length) means that doubling
the context length quadruples the attention computation. Sparse attention, linear
attention, and state-space models (Mamba) are active research areas addressing this.

Hallucination remains the fundamental unsolved problem: models generate fluent,
confident text that is factually incorrect. RAG (Retrieval-Augmented Generation)
mitigates but does not eliminate hallucination by grounding generation in retrieved
facts.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Write sample doc to file
# ─────────────────────────────────────────────────────────────────────────────

def setup() -> str:
    path = "/tmp/transformer_doc.txt"
    Path(path).write_text(SAMPLE_DOC)
    print(f"  ✓ Sample document written to {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def demo_1_embeddings():
    section("DEMO 1 — Embeddings: Similarity in Vector Space")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"\n  Model: BAAI/bge-small-en-v1.5")
    print(f"  Dimensions: {model.get_embedding_dimension()}")

    sentences = [
        "The transformer uses self-attention to process sequences.",
        "Multi-head attention allows the model to focus on different parts.",
        "The boiling point of water is 100 degrees Celsius.",
        "Stock prices fell sharply following the interest rate announcement.",
        "KV caching reduces inference complexity from O(n^2) to O(n).",
    ]

    embeddings = model.encode(sentences, normalize_embeddings=True)

    print(f"\n  Pairwise cosine similarities:")
    print(f"  {'':45} Sim")
    print(f"  {'─'*50}")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            bar = "█" * int(sim * 20)
            print(f"  [{i}]↔[{j}] {bar:<20} {sim:.3f}  "
                  f"'{sentences[i][:25]}...' vs '{sentences[j][:25]}...'")

    print(f"\n  Observation: sentences [0],[1],[4] (all about transformers) "
          f"have high similarity.\n"
          f"  Sentences [2],[3] (water/stocks) are far from the transformer cluster.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def demo_2_chromadb():
    section("DEMO 2 — ChromaDB: Vector Store Basics")

    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # In-memory client for this demo
    client     = chromadb.Client()
    collection = client.create_collection("demo", metadata={"hnsw:space": "cosine"})

    # Add documents with metadata
    texts = [
        "The transformer uses multi-head self-attention.",
        "Positional encodings are added to input embeddings.",
        "KV caching reduces inference complexity.",
        "Flash Attention reduces memory bandwidth usage.",
        "Hallucination is a fundamental limitation of LLMs.",
        "Stock markets rose on positive earnings data.",
    ]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids        = [f"doc_{i}" for i in range(len(texts))]
    metadatas  = [
        {"source": "transformer_doc.txt", "topic": "attention",   "page": 1},
        {"source": "transformer_doc.txt", "topic": "position",    "page": 2},
        {"source": "transformer_doc.txt", "topic": "inference",   "page": 3},
        {"source": "transformer_doc.txt", "topic": "attention",   "page": 3},
        {"source": "transformer_doc.txt", "topic": "limitations", "page": 4},
        {"source": "finance_news.txt",    "topic": "stocks",      "page": 1},
    ]

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"\n  Added {collection.count()} documents to ChromaDB")

    # Query 1: plain similarity search
    query_vec = model.encode(["How does attention work?"], normalize_embeddings=True).tolist()

    print(f"\n  Query: 'How does attention work?'")
    print(f"  Top 3 by cosine similarity:")
    results = collection.query(
        query_embeddings=query_vec,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = 1.0 - dist
        print(f"    [{sim:.3f}] '{doc[:60]}' (topic: {meta['topic']})")

    # Query 2: metadata filtering
    print(f"\n  Query with metadata filter (topic='attention' only):")
    filtered = collection.query(
        query_embeddings=query_vec,
        n_results=3,
        where={"topic": "attention"},
        include=["documents", "distances"],
    )
    for doc, dist in zip(filtered["documents"][0], filtered["distances"][0]):
        print(f"    [{1-dist:.3f}] '{doc[:60]}'")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Retrieval Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_3_retrieval_comparison(doc_path: str):
    section("DEMO 3 — Retrieval Strategies Side-by-Side")

    from rag_engine import RAGEngine

    print("\n  Indexing sample document...")
    engine = RAGEngine(collection_name="demo3", persist_dir="/tmp/demo3_chroma")
    engine.index(doc_path, reset=True, use_parent_child=False)

    query     = "What is KV caching and why does it matter?"
    strategies = ["similarity", "mmr", "bm25", "hybrid"]

    print(f"\n  Query: '{query}'")
    print(f"\n  {'Strategy':<12} {'Score':<8} {'Chunk preview (first 80 chars)'}")
    print(f"  {'─'*70}")

    for strat in strategies:
        results = engine.retrieve(query, k=3, strategy=strat, rerank=False)
        print(f"\n  {strat.upper()}")
        for r in results:
            print(f"  {'':12} [{r.score:6.3f}]  {r.text[:75]}...")

    # Compare with re-ranking
    print(f"\n  HYBRID + CROSS-ENCODER RE-RANKING (top 3 after re-ranking top 20):")
    reranked = engine.retrieve(query, k=3, strategy="hybrid", rerank=True)
    for r in reranked:
        print(f"    [{r.score:6.2f}] {r.text[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: HyDE
# ─────────────────────────────────────────────────────────────────────────────

def demo_4_hyde(doc_path: str):
    section("DEMO 4 — HyDE: Hypothetical Document Embeddings")

    from rag_engine import RAGEngine

    engine = RAGEngine(collection_name="demo4", persist_dir="/tmp/demo4_chroma")
    engine.index(doc_path, reset=True, use_parent_child=False)

    query = "What scaling laws say about training efficiency?"

    print(f"\n  Query: '{query}'")

    print(f"\n  Standard hybrid retrieval (top 2):")
    standard = engine.retrieve(query, k=2, strategy="hybrid")
    for r in standard:
        print(f"    [{r.score:.2f}] {r.text[:100]}...")

    print(f"\n  HyDE retrieval (top 2):")
    hyde = engine.retrieve_hyde(query, k=2)
    for r in hyde:
        print(f"    [{r.score:.2f}] {r.text[:100]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Full RAG with Citations
# ─────────────────────────────────────────────────────────────────────────────

def demo_5_full_rag(doc_path: str):
    section("DEMO 5 — Full RAG Pipeline with Source Citations")

    from rag_engine import RAGEngine

    engine = RAGEngine(collection_name="demo5", persist_dir="/tmp/demo5_chroma")
    engine.index(doc_path, reset=True)

    questions = [
        "What is the scaling factor in attention and why is it used?",
        "How much did it cost to train GPT-3?",
        "What is Flash Attention?",
    ]

    for q in questions:
        response = engine.ask(q, k=4, strategy="hybrid")
        response.pretty_print()
        time.sleep(0.5)  # avoid rate limits


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Parent-Child Chunking Impact
# ─────────────────────────────────────────────────────────────────────────────

def demo_6_parent_child(doc_path: str):
    section("DEMO 6 — Parent-Child Chunking: Precision vs Context")

    from rag_engine import RAGEngine, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE

    print(f"\n  Child chunks:  ~{CHILD_CHUNK_SIZE} chars (indexed in vector DB)")
    print(f"  Parent chunks: ~{PARENT_CHUNK_SIZE} chars (returned to LLM)")

    # Without parent-child
    engine_flat = RAGEngine(collection_name="flat",   persist_dir="/tmp/flat_chroma")
    engine_flat.index(doc_path, reset=True, use_parent_child=False)

    # With parent-child
    engine_pc   = RAGEngine(collection_name="pc",     persist_dir="/tmp/pc_chroma")
    engine_pc.index(doc_path, reset=True, use_parent_child=True)

    query = "What are the limitations of transformer models?"

    print(f"\n  Query: '{query}'")

    flat_chunks = engine_flat.retrieve(query, k=2, strategy="hybrid")
    pc_chunks   = engine_pc.retrieve(query,   k=2, strategy="hybrid")

    print(f"\n  WITHOUT parent-child (flat chunks ~{PARENT_CHUNK_SIZE} chars each):")
    for r in flat_chunks:
        print(f"    Chunk length: {len(r.text)} chars")
        print(f"    Text: {r.text[:150]}...")

    print(f"\n  WITH parent-child (child indexed ~{CHILD_CHUNK_SIZE} chars, "
          f"parent returned ~{PARENT_CHUNK_SIZE} chars):")
    for r in pc_chunks:
        print(f"    Returned chunk length: {len(r.text)} chars (richer context)")
        print(f"    Text: {r.text[:150]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  Phase 04 — RAG & Vector Databases: All Demos")
    print(DIVIDER)

    doc_path = setup()

    demo_1_embeddings()
    demo_2_chromadb()
    demo_3_retrieval_comparison(doc_path)
    demo_4_hyde(doc_path)
    demo_5_full_rag(doc_path)
    demo_6_parent_child(doc_path)

    print(f"\n{DIVIDER}")
    print("  ✅ Phase 04 complete.")
    print("  You built: hybrid RAG with BM25 + vector + re-ranking + HyDE.")
    print("  To launch the Streamlit UI: streamlit run app.py")
    print(DIVIDER)
