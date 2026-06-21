# Phase 04 — RAG Document Q&A App

## What this project does
A production-grade RAG pipeline that can answer questions over any PDF or text file with:
- **Hybrid retrieval** — BM25 keyword search + vector semantic search, merged with RRF
- **Cross-encoder re-ranking** — precise relevance scoring after initial retrieval
- **HyDE** — generate a hypothetical answer first, search with that embedding
- **Parent-child chunking** — index small chunks, return large chunks for context
- **Source citations** — every claim references its source chunk
- **Streamlit UI** — upload, chat, see sources in real-time

## Project structure
```
phase04_project/
├── rag_engine.py   ← Core RAG engine (the library)
├── app.py          ← Streamlit web UI
├── phase04_demo.py         ← Terminal demos (no UI needed)
└── phase04_requirements.txt
```

## Quick start

### 1. Install
```bash
pip install -r phase04_requirements.txt
```
Note: `sentence-transformers` downloads the embedding model (~130MB) on first run.

### 2. Configure .env
```bash
OPENAI_API_KEY=sk-...  # for LLM generation only; embeddings run locally
```

### 3. Run terminal demos
```bash
python phase04_demo.py
```

### 4. Run Streamlit UI
```bash
streamlit run app.py
```
Then open http://localhost:8501, upload a PDF, and start chatting.

## API reference

```python
from rag_engine import RAGEngine

engine = RAGEngine(
    collection_name="my_project",
    persist_dir="./chroma_db",
)

# Index a document (PDF, TXT, URL, CSV)
stats = engine.index("research_paper.pdf")
# → {"chunks_indexed": 47, "total_in_store": 47, "parent_child": True}

# Ask a question (full RAG)
response = engine.ask(
    "What is the main finding?",
    k=5,
    strategy="hybrid",   # "hybrid", "similarity", "mmr", "bm25"
    use_hyde=False,
)
response.pretty_print()  # prints answer + sources

# Just retrieve (no generation)
chunks = engine.retrieve("attention mechanism", k=5, strategy="hybrid")
for chunk in chunks:
    print(f"[{chunk.score:.2f}] {chunk.text[:100]}")

# HyDE retrieval
chunks = engine.retrieve_hyde("What year was the model released?", k=5)

# Stats
print(engine.stats())
```

## The retrieval pipeline

```
User question
     │
     ├──► BM25 keyword search ──────────┐
     │                                  │
     └──► Vector embedding search ──────┤
                                        │
                              RRF merge (top 20)
                                        │
                              Cross-encoder re-rank
                                        │
                              Top 5 chunks → LLM
                                        │
                              Answer + [Source N] citations
```

## Concepts demonstrated

| Demo | Concept |
|------|---------|
| Demo 1 | Embedding similarity, cosine distance, semantic clustering |
| Demo 2 | ChromaDB — add, query, metadata filter |
| Demo 3 | 4 retrieval strategies compared on the same query |
| Demo 4 | HyDE vs standard retrieval for abstract queries |
| Demo 5 | Full RAG with cited answers |
| Demo 6 | Parent-child: small chunks in DB, large chunks to LLM |
