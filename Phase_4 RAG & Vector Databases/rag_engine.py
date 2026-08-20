"""
rag_engine.py — Production-grade RAG engine

Demonstrates all Phase 04 concepts:
  - ChromaDB as the vector store
  - sentence-transformers for local embeddings (or OpenAI)
  - BM25 + vector hybrid retrieval
  - Reciprocal Rank Fusion (RRF)
  - Cross-encoder re-ranking
  - HyDE (Hypothetical Document Embeddings)
  - Parent-child chunking
  - LangChain LCEL for generation
  - Source citations in answers
"""

import os
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal

from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document loaders
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader,
)

# Vector store & embeddings
import chromadb
from chromadb.utils import embedding_functions as chroma_ef

# Sparse retrieval
from rank_bm25 import BM25Okapi

# Sentence transformers (bi-encoder + cross-encoder)
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────────────

EMBED_MODEL_NAME    = "BAAI/bge-small-en-v1.5"   # local, free
RERANK_MODEL_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL           = "gpt-4o-mini"
CHILD_CHUNK_SIZE    = 300
CHILD_CHUNK_OVERLAP = 30
PARENT_CHUNK_SIZE   = 1200
PARENT_CHUNK_OVERLAP = 120
DEFAULT_K           = 5
CANDIDATE_K         = 20   # how many candidates to fetch before re-ranking


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with source metadata and relevance score."""
    text:    str
    source:  str
    page:    int | None
    chunk_id: str
    score:   float = 0.0

@dataclass
class RAGResponse:
    """Full response from the RAG engine."""
    answer:     str
    sources:    list[RetrievedChunk]
    query:      str
    strategy:   str
    num_chunks: int
    metadata:   dict = field(default_factory=dict)

    def pretty_print(self):
        print(f"\n{'─'*60}")
        print(f"Q: {self.query}")
        print(f"{'─'*60}")
        print(f"A: {self.answer}")
        print(f"\nSources ({self.num_chunks} chunks retrieved via {self.strategy}):")
        for i, src in enumerate(self.sources, 1):
            page = f", page {src.page}" if src.page else ""
            print(f"  [Source {i}] {src.source}{page}")
            print(f"  {src.text[:120]}...")
        print()


# ── Embedding wrapper ──────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps sentence-transformers bi-encoder for local free embeddings.
    Swap for OpenAI by changing the embed() method.
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        print(f"  Loading embedding model: {model_name} ...")
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim        = self.model.get_embedding_dimension()
        print(f"  ✓ Embedding model ready (dim={self.dim})")

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Embed a list of texts.
        BGE models recommend a prefix for queries (not for documents).
        """
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}"
                     for t in texts]
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,   # for cosine similarity via dot product
        )

    def embed_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.embed([text], is_query=is_query)[0]


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Production-grade RAG engine with:
    - ChromaDB vector store (persistent)
    - BM25 sparse retrieval
    - Hybrid retrieval via Reciprocal Rank Fusion
    - Cross-encoder re-ranking
    - HyDE (Hypothetical Document Embeddings)
    - Parent-child chunking
    - Source-cited answers via LCEL

    Usage:
        engine = RAGEngine(collection_name="my_doc")
        engine.index("paper.pdf")
        response = engine.ask("What is the main contribution?")
        response.pretty_print()
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_dir: str = "./chroma_db",
        llm_model: str = LLM_MODEL,
    ):
        self.collection_name = collection_name
        self.persist_dir     = persist_dir

        # Embedding model (bi-encoder for retrieval)
        self.embedder = EmbeddingModel(EMBED_MODEL_NAME)

        # Cross-encoder for re-ranking
        print(f"  Loading re-ranker: {RERANK_MODEL_NAME} ...")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)
        print(f"  ✓ Re-ranker ready")

        # ChromaDB vector store (persistent to disk)
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection    = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (rebuilt on each index() call from stored texts)
        self._bm25: BM25Okapi | None = None
        self._corpus_texts: list[str] = []
        self._corpus_ids:   list[str] = []

        # Parent-child map: child_id → parent text
        self._parent_map: dict[str, str] = {}

        # LLM chain
        self._llm   = ChatOpenAI(model=llm_model, temperature=0)
        self._chain = self._build_rag_chain()

        print(f"  ✓ RAG engine ready — collection '{collection_name}' "
              f"({self._collection.count()} existing chunks)")

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(
        self,
        source: str,
        use_parent_child: bool = True,
        reset: bool = False,
    ) -> dict:
        """
        Load, chunk, embed, and index a document.

        Args:
            source:           Path to file or URL
            use_parent_child: If True, index small child chunks but store parents
            reset:            Clear the collection before indexing

        Returns:
            dict with indexing stats
        """
        if reset:
            self.reset()

        print(f"\n  Indexing: {source}")

        # 1. Load
        docs = self._load(source)
        if not docs:
            raise ValueError(f"No content loaded from: {source}")
        print(f"  ✓ Loaded {len(docs)} document segment(s)")

        # 2. Chunk
        if use_parent_child:
            child_chunks, parent_map = self._parent_child_split(docs, source)
            self._parent_map.update(parent_map)
            chunks_to_index = child_chunks
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=PARENT_CHUNK_SIZE,
                chunk_overlap=PARENT_CHUNK_OVERLAP,
            )
            chunks_to_index = splitter.split_documents(docs)
            for i, c in enumerate(chunks_to_index):
                c.metadata.setdefault("source", source)
                c.metadata["chunk_id"] = f"{source}_chunk_{i}"

        print(f"  ✓ Split into {len(chunks_to_index)} chunks")

        # 3. Embed
        texts = [c.page_content for c in chunks_to_index]
        print(f"  Embedding {len(texts)} chunks ...")
        embeddings = self.embedder.embed(texts, is_query=False)
        print(f"  ✓ Embeddings ready")

        # 4. Store in ChromaDB
        ids       = [c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks_to_index)]
        metadatas = [c.metadata for c in chunks_to_index]

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        # 5. Build / rebuild BM25 index from ALL chunks in the collection
        self._rebuild_bm25()

        stats = {
            "source":         source,
            "doc_segments":   len(docs),
            "chunks_indexed": len(chunks_to_index),
            "total_in_store": self._collection.count(),
            "parent_child":   use_parent_child,
        }
        print(f"  ✓ Indexed: {stats}")
        return stats

    def reset(self):
        """Clear all indexed data from the collection."""
        self._chroma_client.delete_collection(self.collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25         = None
        self._corpus_texts = []
        self._corpus_ids   = []
        self._parent_map   = {}
        print(f"  ✓ Collection '{self.collection_name}' reset")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        strategy: Literal["similarity", "mmr", "bm25", "hybrid"] = "hybrid",
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query:    The user's question
            k:        Number of chunks to return
            strategy: Retrieval strategy
            rerank:   Apply cross-encoder re-ranking after retrieval

        Returns:
            List of RetrievedChunk ordered by relevance
        """
        if self._collection.count() == 0:
            raise RuntimeError("No documents indexed. Call .index(source) first.")

        if strategy == "similarity":
            candidates = self._vector_retrieve(query, k=CANDIDATE_K if rerank else k)
        elif strategy == "mmr":
            candidates = self._mmr_retrieve(query, k=CANDIDATE_K if rerank else k)
        elif strategy == "bm25":
            candidates = self._bm25_retrieve(query, k=CANDIDATE_K if rerank else k)
        elif strategy == "hybrid":
            candidates = self._hybrid_retrieve(query, k=CANDIDATE_K)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply cross-encoder re-ranking
        if rerank and len(candidates) > k:
            candidates = self._rerank(query, candidates, top_k=k)
        else:
            candidates = candidates[:k]

        # Resolve parent chunks if using parent-child
        if self._parent_map:
            for chunk in candidates:
                parent_text = self._parent_map.get(chunk.chunk_id)
                if parent_text:
                    chunk.text = parent_text  # return richer parent context

        return candidates

    def retrieve_hyde(self, query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
        """
        HyDE retrieval: generate hypothetical answer, embed it, search.
        """
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a 2-3 sentence factual passage that directly answers "
                       "the question. Write as if from a textbook. Be specific."),
            ("human", "{question}"),
        ])
        chain = hyde_prompt | self._llm | StrOutputParser()
        hypothetical = chain.invoke({"question": query})
        print(f"  HyDE hypothetical: '{hypothetical[:100]}...'")

        hyp_vec = self.embedder.embed_one(hypothetical, is_query=False)
        raw = self._collection.query(
            query_embeddings=[hyp_vec.tolist()],
            n_results=CANDIDATE_K,
            include=["documents", "metadatas"],
        )
        candidates = self._parse_chroma_results(raw)
        return self._rerank(query, candidates, top_k=k)

    # ── Generation ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        k: int = DEFAULT_K,
        strategy: Literal["similarity", "mmr", "bm25", "hybrid"] = "hybrid",
        use_hyde: bool = False,
    ) -> RAGResponse:
        """
        Full RAG: retrieve relevant chunks then generate a cited answer.

        Args:
            question:  The user's question
            k:         Number of source chunks to use
            strategy:  Retrieval strategy
            use_hyde:  Use HyDE for retrieval instead of direct query embedding

        Returns:
            RAGResponse with answer and source citations
        """
        # Retrieve
        if use_hyde:
            chunks = self.retrieve_hyde(question, k=k)
            strat_label = f"hyde+{strategy}"
        else:
            chunks = self.retrieve(question, k=k, strategy=strategy)
            strat_label = f"{strategy}+rerank"

        # Format context with numbered source labels
        context = self._format_context(chunks)

        # Generate with citations
        answer = self._chain.invoke({
            "context":  context,
            "question": question,
        })

        return RAGResponse(
            answer=answer,
            sources=chunks,
            query=question,
            strategy=strat_label,
            num_chunks=len(chunks),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load(self, source: str) -> list[Document]:
        """Auto-detect source type and load."""
        if source.startswith(("http://", "https://")):
            return WebBaseLoader(source).load()
        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(source).load()
        if ext == ".csv":
            return CSVLoader(source).load()
        return TextLoader(source, encoding="utf-8").load()

    def _parent_child_split(
        self,
        docs: list[Document],
        source: str,
    ) -> tuple[list[Document], dict[str, str]]:
        """Split into parent chunks and smaller child chunks for indexing."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP,
        )

        parent_map: dict[str, str] = {}
        children: list[Document]   = []

        parents = parent_splitter.split_documents(docs)

        for p_idx, parent in enumerate(parents):
            p_chunks = child_splitter.split_documents([parent])
            for c_idx, child in enumerate(p_chunks):
                child_id = f"{source}_p{p_idx}_c{c_idx}"
                child.metadata.update({
                    "chunk_id":  child_id,
                    "parent_id": f"{source}_p{p_idx}",
                    "source":    source,
                    "page":      parent.metadata.get("page"),
                })
                parent_map[child_id] = parent.page_content
                children.append(child)

        return children, parent_map

    def _rebuild_bm25(self):
        """Rebuild BM25 index from all chunks currently in ChromaDB."""
        result = self._collection.get(include=["documents"])
        self._corpus_texts = result["documents"]
        self._corpus_ids   = result["ids"]
        tokenized = [t.lower().split() for t in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)

    def _vector_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Dense vector similarity search."""
        q_vec = self.embedder.embed_one(query, is_query=True)
        raw   = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_chroma_results(raw)

    def _mmr_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        MMR (Maximal Marginal Relevance) — balance relevance with diversity.
        Fetches 3x candidates, applies MMR to select k diverse chunks.
        """
        lambda_mult = 0.5
        candidates  = self._vector_retrieve(query, k=k * 3)
        if len(candidates) <= k:
            return candidates

        q_vec = self.embedder.embed_one(query, is_query=True)

        selected: list[RetrievedChunk] = []
        remaining = list(candidates)

        # Embed all candidate texts for inter-chunk similarity
        cand_texts = [c.text for c in remaining]
        cand_vecs  = self.embedder.embed(cand_texts, is_query=False)

        while len(selected) < k and remaining:
            if not selected:
                # First selection: most relevant to query
                best_idx = 0
            else:
                sel_vecs = self.embedder.embed([s.text for s in selected])

                scores = []
                for i, _ in enumerate(remaining):
                    relevance  = float(np.dot(q_vec, cand_vecs[i]))
                    redundancy = max(
                        float(np.dot(cand_vecs[i], sv)) for sv in sel_vecs
                    )
                    mmr_score  = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                    scores.append(mmr_score)

                best_idx = int(np.argmax(scores))

            selected.append(remaining.pop(best_idx))
            cand_vecs = np.delete(cand_vecs, best_idx, axis=0)

        return selected

    def _bm25_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Sparse BM25 keyword retrieval."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Index documents first.")

        scores    = self._bm25.get_scores(query.lower().split())
        top_idxs  = np.argsort(scores)[::-1][:k]

        return [
            RetrievedChunk(
                text=self._corpus_texts[i],
                source=self._get_metadata(self._corpus_ids[i]).get("source", "?"),
                page=self._get_metadata(self._corpus_ids[i]).get("page"),
                chunk_id=self._corpus_ids[i],
                score=float(scores[i]),
            )
            for i in top_idxs if scores[i] > 0
        ]

    def _hybrid_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Hybrid retrieval: BM25 + vector search merged with RRF."""
        vec_results = self._vector_retrieve(query, k=CANDIDATE_K)
        bm25_results = self._bm25_retrieve(query, k=CANDIDATE_K)

        # Build id → chunk mapping
        all_chunks: dict[str, RetrievedChunk] = {}
        for c in vec_results + bm25_results:
            all_chunks[c.chunk_id] = c

        # RRF fusion
        rrf_k = 60
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, c in enumerate(vec_results, 1):
            rrf_scores[c.chunk_id] += 1.0 / (rrf_k + rank)
        for rank, c in enumerate(bm25_results, 1):
            rrf_scores[c.chunk_id] += 1.0 / (rrf_k + rank)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

        result = []
        for cid in sorted_ids:
            chunk = all_chunks[cid]
            chunk.score = rrf_scores[cid]
            result.append(chunk)
        return result

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Cross-encoder re-ranking: score (query, chunk) pairs together."""
        if not candidates:
            return []
        pairs  = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for chunk, score in ranked[:top_k]:
            chunk.score = float(score)
            result.append(chunk)
        return result

    def _parse_chroma_results(self, raw: dict) -> list[RetrievedChunk]:
        """Convert raw ChromaDB query results to RetrievedChunk objects."""
        chunks = []
        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas", [[]])[0] or [{}] * len(docs)
        distances = raw.get("distances", [[]])[0] or [0.0] * len(docs)
        ids       = raw.get("ids", [[]])[0]

        for doc, meta, dist, cid in zip(docs, metas, distances, ids):
            meta = meta or {}
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=meta.get("page"),
                chunk_id=cid,
                score=1.0 - float(dist),   # convert distance to similarity
            ))
        return chunks

    def _get_metadata(self, chunk_id: str) -> dict:
        """Fetch metadata for a chunk by ID."""
        result = self._collection.get(ids=[chunk_id], include=["metadatas"])
        metas = result.get("metadatas", [{}])
        return metas[0] if metas else {}

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered sources for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page = f", page {chunk.page}" if chunk.page is not None else ""
            parts.append(
                f"[Source {i}] ({chunk.source}{page})\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_rag_chain(self):
        """Build the LCEL generation chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about a document.

RULES:
- Answer using ONLY the provided context
- Cite sources using [Source N] after each claim you make
- If multiple sources support a claim, cite all of them: [Source 1][Source 3]
- If the answer is not in the context, say exactly: "I cannot find this in the provided document."
- Be concise and precise

Context:
{context}"""),
            ("human", "{question}"),
        ])
        return prompt | self._llm | StrOutputParser()

    # ── Convenience methods ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return indexing statistics."""
        return {
            "collection":    self.collection_name,
            "total_chunks":  self._collection.count(),
            "bm25_indexed":  len(self._corpus_texts),
            "parent_chunks": len(self._parent_map),
        }
