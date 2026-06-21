"""
rag_service.py — Core AI Service Logic

A self-contained, lightweight RAG engine (simpler than Phase 04's full hybrid
pipeline, to keep this production project focused on PRODUCTIONIZING rather
than re-implementing retrieval). Uses OpenAI embeddings + ChromaDB.

This is the "business logic" layer that main.py (FastAPI) wraps with auth,
rate limiting, streaming, and observability — and that evaluation.py
evaluates with RAGAS.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import chromadb

from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL   = "gpt-4o-mini"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100


@dataclass
class SourceChunk:
    text: str
    source: str
    score: float = 0.0

@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceChunk]
    num_chunks: int
    latency_s: float = 0.0


class RAGService:
    """
    Production-facing RAG service.

    Usage:
        service = RAGService(collection_name="prod_docs", persist_dir="./chroma_db")
        service.index("handbook.pdf")
        answer = service.ask("What is the refund policy?")
    """

    def __init__(
        self,
        collection_name: str = "production_docs",
        persist_dir: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        self._client = OpenAI()
        self._async_client = AsyncOpenAI()

        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Indexing ────────────────────────────────────────────────────────────

    def index(self, source: str) -> dict:
        """Load, chunk, embed, and index a document (PDF, TXT, or URL)."""
        docs = self._load(source)
        if not docs:
            raise ValueError(f"No content loaded from: {source}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        embeddings = self._embed(texts)

        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_idx": i} for i in range(len(chunks))]

        self._collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas,
        )

        return {
            "source": source,
            "chunks_indexed": len(chunks),
            "total_in_collection": self._collection.count(),
        }

    def _load(self, source: str):
        if source.startswith(("http://", "https://")):
            return WebBaseLoader(source).load()
        if source.lower().endswith(".pdf"):
            return PyPDFLoader(source).load()
        return TextLoader(source, encoding="utf-8").load()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    # ── Retrieval + generation ─────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> list[SourceChunk]:
        if self._collection.count() == 0:
            raise RuntimeError("No documents indexed. Call .index(source) first.")

        query_vec = self._embed([query])[0]
        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            chunks.append(SourceChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                score=1.0 - dist,
            ))
        return chunks

    def ask(self, question: str, k: int = 4) -> RAGAnswer:
        """Synchronous full RAG: retrieve + generate."""
        t0 = time.time()
        chunks = self.retrieve(question, k=k)
        context = self._format_context(chunks)
        answer = self._generate(question, context)

        return RAGAnswer(
            answer=answer,
            sources=chunks,
            num_chunks=len(chunks),
            latency_s=round(time.time() - t0, 2),
        )

    async def ask_stream(self, question: str, k: int = 4):
        """
        Async generator version — yields tokens as they arrive.
        Used by the FastAPI streaming endpoint.
        """
        chunks = self.retrieve(question, k=k)   # retrieval is sync (fast, local)
        context = self._format_context(chunks)

        stream = await self._async_client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._build_messages(question, context),
            stream=True,
            temperature=0,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._build_messages(question, context),
            temperature=0,
        )
        return response.choices[0].message.content

    def _build_messages(self, question: str, context: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided context. If the answer isn't "
                    "in the context, say 'I cannot find this in the provided documents.'\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]

    def _format_context(self, chunks: list[SourceChunk]) -> str:
        return "\n\n---\n\n".join(
            f"[{i+1}] ({c.source})\n{c.text}" for i, c in enumerate(chunks)
        )

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "total_chunks": self._collection.count(),
        }
