# Phase 04 — RAG & Vector Databases

> **Prerequisites:** Phases 01–03 complete. You can call LLM APIs, engineer prompts,
> and build LangChain pipelines.  
> **What you'll learn:** Embeddings, vector databases (FAISS, ChromaDB, Qdrant),
> chunking strategies, four retrieval methods, and three advanced RAG techniques.  
> **Project:** A RAG Document Q&A app with hybrid retrieval, cross-encoder re-ranking,
> source citations, and a Streamlit UI.

---

## Table of Contents

1. [The Big Picture — The RAG Paradigm](#1-the-big-picture--the-rag-paradigm)
2. [Embeddings — Dense Vector Representations](#2-embeddings--dense-vector-representations)
3. [Vector Databases](#3-vector-databases)
4. [Chunking Strategies](#4-chunking-strategies)
5. [Retrieval Strategies](#5-retrieval-strategies)
6. [Advanced RAG Techniques](#6-advanced-rag-techniques)
7. [The Full RAG Pipeline with LCEL](#7-the-full-rag-pipeline-with-lcel)
8. [Key Takeaways](#8-key-takeaways)
9. [Practice Exercises](#9-practice-exercises)

---

## 1. The Big Picture — The RAG Paradigm

### 1.1 The problem with context stuffing

In Phase 03, `DocChat` worked by loading the entire document into the system prompt.
This is called **context stuffing**. It works, but it has hard limits:

- A 300-page PDF has roughly 150,000 tokens — exceeding most context windows
- Even with a 1M-token window (Gemini), sending 150,000 tokens costs money on every
  single question, even if the answer is in a single paragraph
- Research shows models struggle with "lost in the middle" — relevant information
  buried in a long context is often ignored ("Lost in the Middle", Liu et al., 2023)
- You cannot dynamically update context-stuffed knowledge without rebuilding the prompt

RAG solves all four problems simultaneously.

---

### 1.2 The RAG paradigm

**RAG (Retrieval-Augmented Generation)** splits document Q&A into two phases:

```
INDEX TIME (done once, offline)
────────────────────────────────────────────────────────
Document → chunk → embed each chunk → store in vector DB

QUERY TIME (done per question, online)
────────────────────────────────────────────────────────
Question → embed → search vector DB → retrieve top-K chunks
→ stuff only those chunks into prompt → generate answer
```

Instead of sending the entire 300-page PDF every time, you only send the 3-5 paragraphs
most relevant to the user's question. This is fast, cheap, and more accurate.

**The key insight:** Questions and their answers are semantically similar. If the user
asks "What is the boiling point of water?", the chunk containing "Water boils at 100°C
at sea level" will be nearby in embedding space — even though the phrasing is different.

---

### 1.3 Components of a RAG system

```
                  ┌─────────────────┐
   Document ────► │  Chunker        │ ──────────────────────────────┐
                  └─────────────────┘                               │
                                                                     ▼
                  ┌─────────────────┐                  ┌─────────────────────┐
                  │  Embedding      │ ────────────────► │  Vector Database    │
                  │  Model          │                   │  (ChromaDB / FAISS  │
                  └─────────────────┘                   │   / Qdrant)         │
                         ▲                              └──────────┬──────────┘
                         │                                         │
   Question ─────────────┘                               top-K chunks
                                                                   │
                  ┌─────────────────┐                              │
                  │  Prompt with    │ ◄────────────────────────────┘
                  │  context chunks │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  LLM            │ ──► Answer with citations
                  └─────────────────┘
```

---

## 2. Embeddings — Dense Vector Representations

### 2.1 What is an embedding?

An embedding is a fixed-size vector of floating-point numbers that represents the
**meaning** of a piece of text. Texts with similar meaning produce vectors that are
close together in the high-dimensional space.

**From your ML background:** You already know embeddings. In a classification neural
network, the layer before the softmax output is the embedding (representation) layer —
it projects the input into a learned feature space. For text embedding models, the entire
network is trained to produce vectors where semantic similarity maps to geometric
proximity. The most common training objective is contrastive learning: push embeddings
of semantically similar sentences together, push dissimilar ones apart.

```python
# An embedding is just a list of numbers
text = "The transformer uses self-attention to process sequences."
embedding = [0.024, -0.102, 0.891, ..., 0.334]  # 1536 numbers for OpenAI
                                                  # 384 numbers for BGE-small
                                                  # 768 numbers for BERT-base
```

The numbers themselves are meaningless individually. What matters is the **distance**
between vectors: similar texts produce vectors that are close; unrelated texts produce
vectors that are far apart.

---

### 2.2 Why cosine similarity

Two similarity metrics are used in practice:

**Cosine similarity:** Measures the angle between two vectors, ignoring magnitude.
```
cos(θ) = (A · B) / (|A| × |B|)    range: -1 to 1
```
- `1.0` = identical direction (most similar)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite direction (antonyms)

**Dot product:** No normalization — larger vectors get higher scores.

Cosine similarity is preferred for text because the magnitude of an embedding does not
carry semantic information. A long text and a short text about the same topic should be
considered similar — cosine handles this correctly by ignoring magnitude.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Example
vec1 = [0.5, 0.3, 0.8]  # "neural networks learn from data"
vec2 = [0.4, 0.4, 0.7]  # "deep learning models train on examples"
vec3 = [0.9, -0.1, -0.5] # "the stock market rose 3% today"

print(cosine_similarity(vec1, vec2))  # → ~0.98  (same topic)
print(cosine_similarity(vec1, vec3))  # → ~0.12  (different topic)
```

---

### 2.3 OpenAI Embedding API

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's embedding API.
    Always pass texts as a list — this batches them in one API call.
    """
    response = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float",   # "float" or "base64" (base64 is smaller over the wire)
    )
    return [item.embedding for item in response.data]


# Available models (as of late 2024)
# text-embedding-3-small: 1536 dimensions, $0.02/1M tokens — best cost/quality for RAG
# text-embedding-3-large: 3072 dimensions, $0.13/1M tokens — highest quality
# text-embedding-ada-002: 1536 dimensions, $0.10/1M tokens — legacy, use 3-small instead


# Embed single text
single = embed_openai(["What is attention in transformers?"])[0]
print(f"Dimensions: {len(single)}")          # → 1536
print(f"Type: {type(single[0])}")            # → float
print(f"Range: {min(single):.3f} to {max(single):.3f}")  # → approx -0.4 to 0.4


# Embed a batch efficiently — always prefer batching over single calls
texts = [
    "Transformers use self-attention to process sequences in parallel.",
    "RNNs process sequences one token at a time.",
    "Convolutional networks are primarily used for image tasks.",
    "The stock market closed higher on Tuesday.",
]
embeddings = embed_openai(texts)

# Compare pairwise similarity
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"[{i}] vs [{j}]: {sim:.3f}")
# → [0] vs [1]: 0.87  (both about sequence models — high similarity)
# → [0] vs [2]: 0.74  (both about neural networks — medium)
# → [0] vs [3]: 0.42  (unrelated topic — low similarity)
```

---

### 2.4 Sentence Transformers (open source, local)

OpenAI embeddings cost money and require an internet connection. `sentence-transformers`
runs locally for free.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# BGE (Beijing Academy of AI) models — best open-source embedding models
# BAAI/bge-small-en-v1.5  → 384 dims, 33M params, fast, good quality
# BAAI/bge-base-en-v1.5   → 768 dims, 110M params, better quality
# BAAI/bge-large-en-v1.5  → 1024 dims, 335M params, best open-source quality

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# Downloads ~130MB on first run, then cached locally

# Embed text — returns numpy array
single = model.encode("What is attention in transformers?")
print(f"Dimensions: {single.shape}")     # → (384,)
print(f"Type: {type(single)}")           # → numpy.ndarray

# Embed batch (much faster than one at a time)
texts = ["text one", "text two", "text three"]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
print(f"Batch shape: {embeddings.shape}")  # → (3, 384)

# BGE models need a query prefix for retrieval tasks
# Document chunks: embed as-is
# Queries: prefix with "Represent this sentence for searching relevant passages: "
query   = "Represent this sentence for searching relevant passages: What is attention?"
doc     = "The attention mechanism allows the model to focus on relevant parts..."

query_emb = model.encode(query)
doc_emb   = model.encode(doc)
sim       = cosine_similarity(query_emb, doc_emb)
print(f"Similarity: {sim:.3f}")
```

---

### 2.5 Bi-encoder vs Cross-encoder — a critical distinction

This distinction matters for understanding re-ranking later:

**Bi-encoder** (all embedding models above):
```
Query ──► Encoder ──► query_vector
                                    ──► cosine_similarity ──► score
Doc   ──► Encoder ──► doc_vector
```
Each text is encoded independently. Similarity is computed between the resulting
vectors. This is fast because document vectors can be pre-computed and stored.
The downside: the query and document never "see" each other during encoding, so
nuanced interactions are missed.

**Cross-encoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`):
```
[Query + Document] ──► Encoder ──► relevance_score
```
The query and document are concatenated and passed through the model together.
The model explicitly attends to interactions between query terms and document terms.
This is much more accurate but too slow for searching millions of documents (you would
have to run it on every document). It is used as a **re-ranker**: first get top-50 with
a bi-encoder, then re-rank those 50 with a cross-encoder.

```python
from sentence_transformers import CrossEncoder

# Cross-encoder: takes (query, document) pairs, returns scores
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is the boiling point of water?"
candidates = [
    "Water boils at 100°C (212°F) at sea level.",
    "The melting point of ice is 0°C.",
    "Mercury has a boiling point of 356.7°C.",
    "Boiling is the process of converting liquid to vapor.",
]

pairs  = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)  # higher score = more relevant

for doc, score in sorted(zip(candidates, scores), key=lambda x: -x[1]):
    print(f"  {score:.2f}  {doc[:60]}")
# → 9.21  Water boils at 100°C (212°F) at sea level.      ← correct
# → 3.14  Boiling is the process of converting liquid...
# → 1.87  The melting point of ice is 0°C.
# → 0.33  Mercury has a boiling point of 356.7°C.
```

---

### 2.6 Embedding model comparison

| Model | Dims | Size | Speed | Quality | Cost |
|---|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | API | Fast | ★★★★☆ | $0.02/1M tokens |
| `text-embedding-3-large` | 3072 | API | Fast | ★★★★★ | $0.13/1M tokens |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Very fast | ★★★☆☆ | Free |
| `BAAI/bge-base-en-v1.5` | 768 | 430MB | Fast | ★★★★☆ | Free |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Medium | ★★★★★ | Free |

**When to use which:**
- Prototyping: `bge-small` (free, fast, good enough)
- Development: `text-embedding-3-small` (consistent quality, easy)
- Production: benchmark on your data — often `bge-large` matches `text-embedding-3-large`

---

## 3. Vector Databases

### 3.1 Why a specialized database?

A regular SQL database can store vectors, but finding the nearest neighbor requires
comparing a query vector against every stored vector — O(n) complexity. With 1 million
chunks, that means 1 million cosine similarity calculations per query.

Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms to find the
most similar vectors in O(log n) or better, trading a tiny bit of recall for massive
speed gains. The most common algorithm is HNSW (Hierarchical Navigable Small World),
a graph-based index where nodes are vectors and edges connect similar ones.

---

### 3.2 FAISS — in-memory, from Meta/Facebook

FAISS (Facebook AI Similarity Search) is a C++ library with Python bindings. It is not
a database — it has no persistence built-in — but it is the fastest option for
pure vector search.

```python
import faiss
import numpy as np

# 1. Create an index
dimension = 384  # must match your embedding model's output dimension

# IndexFlatL2: exact search using L2 (Euclidean) distance — slowest but 100% accurate
index_exact = faiss.IndexFlatL2(dimension)

# IndexFlatIP: exact search using inner product (dot product) — use with normalized vectors
index_ip = faiss.IndexFlatIP(dimension)

# IndexIVFFlat: approximate search — partitions space into nlist clusters
# train() needed before add() — learns the cluster structure from data
quantizer = faiss.IndexFlatL2(dimension)
index_approx = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)  # 100 clusters
# Tradeoff: nlist higher → faster search but must also set nprobe higher for accuracy

# 2. Add vectors
# faiss requires numpy float32 arrays
# shape: (number_of_vectors, dimension)
texts = ["text one", "text two", "text three", "text four", "text five"]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vectors = model.encode(texts).astype("float32")   # FAISS needs float32

if hasattr(index_exact, 'is_trained') and not index_exact.is_trained:
    index_exact.train(vectors)  # needed for IVF, not for Flat
index_exact.add(vectors)
print(f"Index size: {index_exact.ntotal}")  # → 5

# 3. Search
query = "What is text one about?"
query_vec = model.encode([query]).astype("float32")  # shape: (1, 384)

k = 3  # return top 3
distances, indices = index_exact.search(query_vec, k)
# distances: array of L2 distances, shape (1, k)
# indices:   array of stored vector indices, shape (1, k)

print(f"Top {k} results:")
for dist, idx in zip(distances[0], indices[0]):
    print(f"  [{idx}] '{texts[idx][:50]}' (L2 distance: {dist:.3f})")

# 4. Save and load (FAISS index — not the text)
faiss.write_index(index_exact, "my_index.faiss")
loaded_index = faiss.read_index("my_index.faiss")
# Note: you must save the texts separately (FAISS only stores vectors, not metadata)
```

**When to use FAISS:** Batch processing, research, or when you need maximum speed and
you are managing persistence yourself. FAISS is the engine inside many higher-level
libraries including LangChain's FAISS vector store.

---

### 3.3 ChromaDB — simple, persistent, developer-friendly

ChromaDB is a full vector database with a simple Python API, built-in persistence,
and metadata filtering. It handles the embedding model, storage, and retrieval in one
package. The best choice for development and small-to-medium production workloads.

```python
import chromadb
from chromadb.utils import embedding_functions

# Persistent client — saves to disk automatically
client = chromadb.PersistentClient(path="./chroma_db")
# In-memory client (data lost on restart):
# client = chromadb.Client()

# Create or load a collection
# A collection is like a table — it holds vectors, documents, and metadata
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"},   # use cosine similarity (default is L2)
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    # ChromaDB embeds automatically if you provide an embedding_function
    # Or pass pre-computed embeddings manually (see below)
)

# Add documents — ChromaDB embeds them automatically
collection.add(
    documents=["text one about ML", "text two about finance", "text three about cooking"],
    ids=["doc-1", "doc-2", "doc-3"],
    metadatas=[
        {"source": "ml_paper.pdf", "page": 1},
        {"source": "report.pdf",   "page": 5},
        {"source": "cookbook.pdf", "page": 12},
    ]
)

print(f"Collection size: {collection.count()}")  # → 3

# Query — ChromaDB embeds the query automatically
results = collection.query(
    query_texts=["machine learning algorithms"],
    n_results=2,
    include=["documents", "metadatas", "distances"],
)

print("Query results:")
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  [{dist:.3f}] '{doc[:50]}' from {meta['source']} p.{meta['page']}")

# With metadata filtering — only search within a specific source
results_filtered = collection.query(
    query_texts=["algorithms"],
    n_results=5,
    where={"source": "ml_paper.pdf"},    # metadata filter
)

# Update, delete
collection.update(ids=["doc-1"], documents=["updated text about ML"])
collection.delete(ids=["doc-3"])

# Add pre-computed embeddings (when you compute them yourself)
my_embeddings = model.encode(["new text"]).tolist()
collection.add(
    documents=["new text"],
    embeddings=my_embeddings,   # skip auto-embedding
    ids=["doc-4"],
)
```

---

### 3.4 Qdrant — production-grade vector database

Qdrant is a production-ready vector database written in Rust. It offers: payload
filtering, vector quantization (smaller memory footprint), distributed mode (scale
horizontally), and a managed cloud service.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
import numpy as np

# In-memory (for testing)
client = QdrantClient(":memory:")

# Local persistent storage
# client = QdrantClient(path="./qdrant_storage")

# Cloud (production)
# client = QdrantClient(url="https://xyz.qdrant.io", api_key="your_key")

# Create a collection
client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(
        size=384,             # must match embedding dimensions
        distance=Distance.COSINE,
    ),
)

# Index documents — PointStruct is Qdrant's document object
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts     = ["ML text", "Finance text", "Cooking text"]
vectors   = embed_model.encode(texts).tolist()

points = [
    PointStruct(
        id=i,
        vector=vectors[i],
        payload={              # metadata — can filter on any payload field
            "text": texts[i],
            "source": f"doc_{i}.pdf",
            "page": i + 1,
            "category": ["ml", "finance", "cooking"][i],
        }
    )
    for i in range(len(texts))
]

client.upsert(collection_name="my_docs", points=points)

# Search
query_vec = embed_model.encode(["machine learning"]).tolist()[0]

results = client.search(
    collection_name="my_docs",
    query_vector=query_vec,
    limit=2,
    with_payload=True,
    query_filter=Filter(           # optional: filter by metadata
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="ml")
            )
        ]
    )
)

for r in results:
    print(f"  [{r.score:.3f}] {r.payload['text']} (page {r.payload['page']})")

# Collection info
info = client.get_collection("my_docs")
print(f"Vectors: {info.vectors_count}")
```

---

### 3.5 FAISS vs ChromaDB vs Qdrant

| Feature | FAISS | ChromaDB | Qdrant |
|---|---|---|---|
| Persistence | Manual (save/load index) | Built-in | Built-in |
| Metadata storage | No (vectors only) | Yes | Yes (payload) |
| Metadata filtering | No | Yes | Yes (advanced) |
| Setup complexity | Low (Python only) | Low | Low–Medium |
| Production-ready | Not standalone | Small–medium scale | Full production |
| Cloud service | No | No | Yes (Qdrant Cloud) |
| Best for | Speed benchmarks, research | Development, small apps | Production |

**For learning and this project:** ChromaDB. Simple, persistent, no server needed.  
**For production at scale:** Qdrant (or Pinecone if you want fully managed).

---

## 4. Chunking Strategies

### 4.1 Why chunking strategy matters

The chunk is the atomic unit of retrieval. If a chunk is too large:
- You retrieve more text than needed → LLM has to sift through noise
- Similarity is diluted across many topics in one chunk

If a chunk is too small:
- Retrieved chunks lack context → LLM cannot answer fully
- A sentence alone often lacks the surrounding context that makes it meaningful

The goal is chunks that are **topically coherent** — each chunk is about one thing.

---

### 4.2 Fixed-size splitting (baseline)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split by character count — the Phase 03 default
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # max characters per chunk
    chunk_overlap=200,  # overlap between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # try these in order
)
```

**Pros:** Simple, predictable chunk sizes.  
**Cons:** Splits mid-sentence or mid-paragraph if boundaries don't align.

---

### 4.3 Token-based splitting

Character count ≠ token count. For precise context window management, split by tokens:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,     # 256 TOKENS (not characters) per chunk
    chunk_overlap=32,
    length_function=lambda text: len(enc.encode(text)),  # token counting
)
```

---

### 4.4 Semantic splitting — split on meaning, not size

Semantic splitting embeds each sentence, then splits where the embedding similarity
drops — i.e., where the topic changes. This produces chunks that are guaranteed to be
about one topic, regardless of how long they are.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # split where similarity drops below Nth percentile
    breakpoint_threshold_amount=95,           # split at the bottom 5% of similarities
)

# The splitter embeds every sentence, compares adjacent ones,
# and inserts chunk boundaries where topics shift
chunks = splitter.split_text(long_document)
```

**Pros:** Chunks align with topic boundaries rather than arbitrary character counts.  
**Cons:** Requires embedding every sentence (slow and costly for large docs).  
**When to use:** Long structured documents (textbooks, reports) where chapters and
sections do not have consistent character lengths.

---

### 4.5 Parent-child chunking (preview of Section 6)

Index small child chunks for precise retrieval. When a child is retrieved, return its
larger parent chunk to give the LLM more context.

```python
# Child chunks: small (200 chars) — retrieved precisely
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Parent chunks: large (1000 chars) — returned to the LLM
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

parents  = parent_splitter.split_documents(docs)
children = []

for i, parent in enumerate(parents):
    child_chunks = child_splitter.split_documents([parent])
    for child in child_chunks:
        child.metadata["parent_id"] = i   # link each child to its parent
    children.extend(child_chunks)

# Index children in vector DB (for precise retrieval)
# Map parent_id → parent text (for context retrieval)
parent_map = {i: p.page_content for i, p in enumerate(parents)}
```

---

## 5. Retrieval Strategies

### 5.1 Similarity search (dense retrieval)

The baseline: embed the query, find the top-k most similar vectors in the database.

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load an existing ChromaDB collection
vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Similarity search — returns list of Document objects
results = vector_store.similarity_search(
    query="What is the boiling point of water?",
    k=5,
)
for doc in results:
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content: {doc.page_content[:100]}\n")

# With scores — returns list of (Document, score) tuples
# Score is cosine similarity: 1.0 = identical, 0.0 = unrelated
results_with_scores = vector_store.similarity_search_with_score(
    query="boiling point of water",
    k=5,
)
for doc, score in results_with_scores:
    print(f"  Score: {score:.3f}  {doc.page_content[:80]}")
```

**Limitation of similarity search:** If three of your top-5 chunks all cover the same
paragraph from page 3, you get redundant information. The answer might actually be in
a unique chunk that ranked 6th.

---

### 5.2 MMR — Maximal Marginal Relevance

MMR trades a little relevance for diversity. It iteratively picks the next chunk that
is: (a) most relevant to the query, and (b) least similar to already-selected chunks.

```python
# MMR retrieval — same interface as similarity_search
results_mmr = vector_store.max_marginal_relevance_search(
    query="What is attention in transformers?",
    k=5,           # number of results to return
    fetch_k=20,    # candidate pool to fetch before applying MMR
    lambda_mult=0.5,   # 0 = maximize diversity, 1 = maximize relevance
                       # 0.5 = balanced (default)
)
```

**When to use MMR:** When your document has many similar chunks on the same topic and
you want to avoid returning redundant information. For most production RAG systems, MMR
is a better default than plain similarity search.

---

### 5.3 BM25 — sparse keyword retrieval

BM25 is a classic information retrieval algorithm (an evolution of TF-IDF) that scores
documents based on keyword overlap with the query, not semantic similarity.

```
BM25(query, doc) = Σ IDF(term) × (TF(term, doc) × (k₁ + 1)) / (TF(term, doc) + k₁ × (1 - b + b × |doc|/avgdl))
```

Where:
- IDF: inverse document frequency (rare terms score higher)
- TF: term frequency in the document
- k₁, b: tuning parameters (k₁=1.5, b=0.75 by default)
- avgdl: average document length in the corpus

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Corpus: list of tokenized documents
corpus_texts = [doc.page_content for doc in all_chunks]
tokenized_corpus = [text.lower().split() for text in corpus_texts]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "attention mechanism transformer architecture"
tokenized_query = query.lower().split()

# Get BM25 scores for all documents
scores = bm25.get_scores(tokenized_query)

# Top-k indices
top_k = 5
top_indices = np.argsort(scores)[::-1][:top_k]

for idx in top_indices:
    print(f"  Score: {scores[idx]:.3f}  {corpus_texts[idx][:80]}")
```

**When BM25 beats vector search:**
- Exact product codes: "SKU-XR-4521" — vector search may not find exact matches
- Person names: "Geoff Hinton" — might not be in the embedding model's vocabulary cleanly
- Technical abbreviations: "GPT-4o", "RLHF", "LoRA"
- Legal references: "Section 42(b)(ii)" — exact text matching matters

**When vector search beats BM25:**
- Paraphrased questions: "error during training" matches "loss not converging"
- Conceptual questions: "how does the model learn?" matches technical explanations
- Any case where the user's phrasing differs from the document's phrasing

---

### 5.4 Hybrid retrieval with Reciprocal Rank Fusion

Hybrid retrieval combines BM25 and vector search rankings using RRF (Reciprocal Rank
Fusion). RRF is simple, parameter-free, and works remarkably well:

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    *rankings: list[int],   # each ranking is a list of chunk indices, most relevant first
    k: int = 60,            # RRF constant — higher k reduces the impact of high rankings
) -> list[int]:
    """
    Merge multiple rankings into one using Reciprocal Rank Fusion.
    
    RRF score for a document d = Σ 1/(k + rank(d)) across all rankings
    Documents appearing at the top of multiple rankings score highest.
    
    k=60 is the standard value from the original RRF paper.
    """
    scores: dict[int, float] = defaultdict(float)
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def hybrid_retrieve(
    query: str,
    vector_store: Chroma,
    bm25_index: BM25Okapi,
    corpus_texts: list[str],
    k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 + vector search merged with RRF.
    
    1. Get top-candidate_k from vector search
    2. Get top-candidate_k from BM25
    3. Merge rankings with RRF
    4. Return top-k merged results
    """
    # Vector search ranking
    vector_results = vector_store.similarity_search(query, k=candidate_k)
    # Map from chunk text to index in corpus (for cross-referencing BM25)
    text_to_idx = {text: i for i, text in enumerate(corpus_texts)}
    vector_ranking = [
        text_to_idx.get(doc.page_content, -1)
        for doc in vector_results
        if doc.page_content in text_to_idx
    ]

    # BM25 ranking
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    bm25_ranking = list(np.argsort(scores)[::-1][:candidate_k])

    # Merge with RRF
    fused = reciprocal_rank_fusion(vector_ranking, bm25_ranking)[:k]

    return [{"text": corpus_texts[i], "score": i} for i in fused]
```

---

## 6. Advanced RAG Techniques

### 6.1 Cross-encoder re-ranking

After getting top-20 candidates with hybrid retrieval, re-rank them with a cross-encoder.
The cross-encoder reads each (query, chunk) pair together and produces a precise
relevance score — far more accurate than embedding similarity.

```python
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # small but effective
    # Other options:
    # "cross-encoder/ms-marco-MiniLM-L-12-v2"  → larger, slower, more accurate
    # "BAAI/bge-reranker-large"                 → best quality
)

def rerank_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Re-rank a list of retrieved chunks using a cross-encoder.
    Returns the top_k chunks sorted by relevance score.
    """
    pairs  = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Sort by score descending, take top_k
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# Full pipeline: get 20 candidates, re-rank to top 5
candidates = hybrid_retrieve(query, vector_store, bm25_index, corpus_texts, k=20)
candidate_texts = [c["text"] for c in candidates]

top_chunks = rerank_chunks(query, candidate_texts, top_k=5)

for chunk_text, score in top_chunks:
    print(f"  [{score:.2f}] {chunk_text[:80]}")
```

**Why the two-stage pipeline?**

| Stage | Method | Speed | Accuracy | Scales to |
|---|---|---|---|---|
| Retrieval | Bi-encoder + BM25 | Fast (ms) | Good | Millions of docs |
| Re-ranking | Cross-encoder | Medium (0.5–2s) | Excellent | Top 20–50 docs |

You cannot use a cross-encoder for initial retrieval (too slow for millions of docs).
You cannot skip re-ranking if you want production-level accuracy. The two stages are
complementary.

---

### 6.2 HyDE — Hypothetical Document Embeddings

**The problem:** Questions and document chunks are phrased differently.

- Query: "What year was the transformer model introduced?"
- Document: "Vaswani et al. published 'Attention Is All You Need' in 2017, introducing..."

The query embedding lives in "question space"; the chunk embedding lives in "answer space".
They might not be close even though they are semantically related.

**HyDE's solution:** Generate a hypothetical answer to the question first, then embed
*that* instead of the raw question. Hypothetical answers are phrased like document chunks.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def hyde_embed(query: str, embedding_model) -> list[float]:
    """
    1. Generate a hypothetical passage that would answer the query
    2. Embed that passage instead of the raw query
    """
    # Step 1: generate hypothetical answer
    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a short, factual passage (2-3 sentences) that directly "
                   "answers the following question. Write as if from a textbook or "
                   "technical document. Do not mention that this is hypothetical."),
        ("human", "{question}")
    ])
    chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_passage = chain.invoke({"question": query})
    
    print(f"  Query:               '{query}'")
    print(f"  Hypothetical answer: '{hypothetical_passage[:100]}...'")
    
    # Step 2: embed the hypothetical passage (not the question)
    return embedding_model.encode(hypothetical_passage).tolist()


# Usage in retrieval
query   = "What year was the transformer model introduced?"
hyde_vec = hyde_embed(query, embed_model)

# Search with the HyDE vector instead of the query vector
results = vector_store.similarity_search_by_vector(hyde_vec, k=5)
```

**When HyDE helps:**
- Highly technical queries where phrasing differs significantly from document text
- Short queries (1 question) against long, dense documents

**When HyDE hurts:**
- Factual queries where the LLM might hallucinate a wrong hypothetical answer
- When the hypothetical answer is far from the actual document content

---

### 6.3 Parent-child retrieval

**The problem:** Small chunks are precise for retrieval but lack context for answering.
Large chunks have context but are imprecise for retrieval (diluted similarity).

**Solution:** Index small child chunks for retrieval, return large parent chunks to the LLM.

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Split into parents (large, contextual) and children (small, precise)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=300,  chunk_overlap=30)

def build_parent_child_index(
    docs: list[Document],
    vector_store: Chroma,
    embedding_model,
) -> dict[str, str]:
    """
    Build parent-child index.
    Returns: parent_map — dict mapping child_id to parent text
    """
    parent_map = {}
    all_children = []
    
    parents = parent_splitter.split_documents(docs)
    
    for parent_id, parent in enumerate(parents):
        children = child_splitter.split_documents([parent])
        
        for child_idx, child in enumerate(children):
            child_id = f"child_{parent_id}_{child_idx}"
            child.metadata["child_id"]  = child_id
            child.metadata["parent_id"] = parent_id
            
            parent_map[child_id] = parent.page_content  # ← the key mapping
            all_children.append(child)
    
    # Index only the children in the vector store
    vector_store.add_documents(all_children)
    
    print(f"Indexed {len(parents)} parents → {len(all_children)} children")
    return parent_map


def parent_child_retrieve(
    query: str,
    vector_store: Chroma,
    parent_map: dict[str, str],
    k: int = 5,
) -> list[str]:
    """
    Retrieve child chunks (precise), return their parent chunks (contextual).
    Deduplicates parents — multiple children from the same parent return it only once.
    """
    children = vector_store.similarity_search(query, k=k * 2)  # get more children
    
    seen_parent_ids = set()
    parent_texts = []
    
    for child in children:
        parent_id = child.metadata.get("parent_id")
        
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            child_id = child.metadata.get("child_id")
            parent_text = parent_map.get(child_id, child.page_content)
            parent_texts.append(parent_text)
        
        if len(parent_texts) >= k:
            break
    
    return parent_texts
```

---

## 7. The Full RAG Pipeline with LCEL

### 7.1 Putting retrieval into an LCEL chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAG prompt — instructs the model to cite sources
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about a document.

Use ONLY the provided context to answer the question.
For each claim, cite the source using [Source N].
If the answer is not in the context, say: "I cannot find this in the document."

Context:
{context}"""),
    ("human", "{question}"),
])


def format_context(chunks: list) -> str:
    """Format retrieved chunks with source labels for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        parts.append(
            f"[Source {i}] (from {source}, page {page})\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def retrieve(query: str, vector_store: Chroma, k: int = 5) -> list:
    return vector_store.similarity_search(query, k=k)


# Build the LCEL RAG chain
def build_rag_chain(vector_store: Chroma) -> any:
    retriever = RunnableLambda(
        lambda q: retrieve(q, vector_store)
    )
    
    chain = (
        RunnablePassthrough.assign(
            retrieved=lambda x: retriever.invoke(x["question"])
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["retrieved"])
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain


# Use the chain
chain  = build_rag_chain(vector_store)
answer = chain.invoke({"question": "What is the role of Query, Key, and Value in attention?"})
print(answer)
# → "In the attention mechanism, Query, Key, and Value vectors serve distinct roles [Source 1].
#    The Query represents what the current token is looking for [Source 1], while
#    the Key represents what each token offers to others [Source 2]..."
```

---

### 7.2 Source extraction for citations

```python
from dataclasses import dataclass

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str
    num_chunks: int

def rag_with_citations(
    question: str,
    vector_store: Chroma,
    k: int = 5
) -> RAGResponse:
    """Full RAG with structured source citations."""
    
    # Retrieve
    chunks = vector_store.similarity_search(question, k=k)
    
    # Format context with numbered source labels
    context = format_context(chunks)
    
    # Generate
    answer = (rag_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })
    
    # Extract source metadata
    sources = [
        {
            "label":   f"Source {i+1}",
            "source":  c.metadata.get("source", "unknown"),
            "page":    c.metadata.get("page", "?"),
            "excerpt": c.page_content[:200] + "...",
        }
        for i, c in enumerate(chunks)
    ]
    
    return RAGResponse(
        answer=answer,
        sources=sources,
        query=question,
        num_chunks=len(chunks),
    )
```

---

## 8. Key Takeaways

1. **RAG = index offline, retrieve online.** Never send the full document every time.
   Embed once, search fast, augment the prompt with only what is relevant.

2. **Embeddings map meaning to geometry.** Similar texts → nearby vectors. Cosine
   similarity is the standard metric. Bi-encoders are fast (pre-compute docs); cross-
   encoders are accurate (joint query-doc encoding). Use both in a two-stage pipeline.

3. **ChromaDB for development, Qdrant for production.** ChromaDB is the simplest
   persistent vector database — zero setup, great Python API. Qdrant adds filtering,
   quantization, and scale.

4. **Hybrid retrieval (BM25 + vector) beats either alone.** Vector search misses exact
   matches. BM25 misses semantic paraphrasing. RRF merges the two rankings without any
   tunable parameters.

5. **Re-ranking is the highest-ROI improvement.** Get top-20 with bi-encoder + BM25,
   re-rank with cross-encoder, return top-5 to the LLM. Dramatically improves answer
   quality with only 0.5–2 seconds of extra latency.

6. **HyDE for hard queries, parent-child for long documents.** HyDE bridges the
   question-answer phrasing gap. Parent-child balances retrieval precision (small
   children) with answer context (large parents).

7. **Format context with source labels.** Number your retrieved chunks (`[Source 1]`,
   `[Source 2]`), instruct the model to cite them, and display sources alongside
   the answer. This makes your RAG system trustworthy and debuggable.

---

## 9. Practice Exercises

### Exercise 1 — Embedding Explorer (Easy)
Embed 20 sentences from a Wikipedia article on any topic. Compute the pairwise cosine
similarity matrix. Find the pair with highest similarity and the pair with lowest
similarity. Visualize the matrix as a heatmap using matplotlib.

### Exercise 2 — Retrieval Strategy Benchmark (Medium)
Take a document and 10 questions with known ground-truth answers. Run similarity search,
MMR, and BM25 independently. For each strategy, mark each question as correctly answered
(the relevant chunk was in the top-5) or not. Compare recall@5 across strategies.

### Exercise 3 — Persistent Multi-Document RAG (Medium)
Extend the project's `RAGEngine` to support indexing multiple documents into the same
ChromaDB collection. Add a `source` metadata field to every chunk. When answering, show
which document each source chunk came from. Allow the user to ask "only search the
first document" using ChromaDB metadata filters.

### Exercise 4 — Full Evaluation with RAGAS (Hard)
Install `ragas`. Create a test set of 15 (question, ground-truth-answer) pairs over your
indexed document. Run your RAG pipeline on all 15 questions. Use RAGAS to compute:
`faithfulness` (answer supported by context), `answer_relevancy` (answer relevant to
question), and `context_recall` (ground truth covered by retrieved chunks). Report the
scores and identify the weakest component.

---

*Next: Phase 05 — AI Agents & LangGraph*  
*You will build autonomous systems that plan, decide which tools to use, and loop
until a task is complete — not just answer one question.*
