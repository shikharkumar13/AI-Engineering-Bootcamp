"""
doc_chat.py — Multi-turn Document Q&A with Session Memory

Demonstrates all Phase 03 concepts:
  - LCEL chain composition with |
  - RunnableWithMessageHistory for memory
  - Document loaders (text, PDF, URL, CSV)
  - RecursiveCharacterTextSplitter
  - LangSmith automatic tracing
  - Streaming responses
  - Multiple concurrent sessions
"""

import os
import time
from pathlib import Path
from typing import Generator, Iterator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders — each handles a different source type
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
)

load_dotenv()


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 15_000   # max document chars to include in system prompt
CHUNK_SIZE        = 1_000    # characters per chunk when splitting
CHUNK_OVERLAP     = 200      # overlap between chunks
DEFAULT_MODEL     = "gpt-4o-mini"


# ── System prompt template ─────────────────────────────────────────────────────
# Uses MessagesPlaceholder for conversation history injection.
# {context} is filled with the loaded document content.

SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions \
about the document provided below.

Rules:
- Base your answers only on the document content
- If the answer is not in the document, clearly say so
- Be concise and precise
- When quoting the document, use quotation marks

Document:
---------
{context}
---------"""


# ── DocChat class ──────────────────────────────────────────────────────────────

class DocChat:
    """
    Multi-turn Q&A over a loaded document with persistent session memory.

    Every LLM call is automatically traced in LangSmith if env vars are set.

    Usage:
        chat = DocChat(session_id="user-alice")
        chat.load("research_paper.pdf")      # or a URL, .txt, .csv
        reply = chat.chat("What is the main finding?")
        reply = chat.chat("Can you elaborate on that?")  # remembers context
        chat.show_history()
    """

    def __init__(
        self,
        session_id: str = "default",
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        self.session_id   = session_id
        self._context     = ""          # the loaded document text
        self._doc_info    = {}          # metadata about the loaded document
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}

        self._llm    = ChatOpenAI(model=model, temperature=temperature, streaming=True)
        self._parser = StrOutputParser()
        self._chain  = self._build_chain()

    # ── Document loading ───────────────────────────────────────────────────────

    def load(self, source: str, chunk_size: int = CHUNK_SIZE) -> dict:
        """
        Load a document from any supported source.

        Supported sources:
            - File path ending in .pdf  → PyPDFLoader
            - File path ending in .csv  → CSVLoader
            - File path (any other)     → TextLoader
            - URL (http/https)          → WebBaseLoader

        Args:
            source:     Path, URL, or file path
            chunk_size: Characters per chunk (smaller = more precise retrieval later)

        Returns:
            dict with 'chunks', 'total_chars', 'source', 'type'
        """
        docs = self._load_source(source)
        if not docs:
            raise ValueError(f"No content loaded from: {source}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Build context string (join all chunks, up to MAX_CONTEXT_CHARS)
        # In Phase 04, we will replace this with semantic retrieval
        full_text = "\n\n---\n\n".join(c.page_content for c in chunks)
        self._context = full_text[:MAX_CONTEXT_CHARS]

        self._doc_info = {
            "source": source,
            "type":   self._detect_source_type(source),
            "chunks": len(chunks),
            "total_chars": len(full_text),
            "context_chars": len(self._context),
            "truncated": len(full_text) > MAX_CONTEXT_CHARS,
        }

        if self._doc_info["truncated"]:
            print(f"  ⚠ Document truncated: {len(full_text):,} chars → "
                  f"{MAX_CONTEXT_CHARS:,} chars included in context.")
            print("  Tip: Phase 04 (RAG) will handle arbitrarily large documents.")

        print(f"  ✓ Loaded '{source}' — {len(chunks)} chunks, "
              f"{len(self._context):,} chars in context.")
        return self._doc_info

    def _load_source(self, source: str) -> list[Document]:
        """Choose the right loader based on source type."""
        src_type = self._detect_source_type(source)

        if src_type == "url":
            loader = WebBaseLoader(source)
        elif src_type == "pdf":
            loader = PyPDFLoader(source)
        elif src_type == "csv":
            loader = CSVLoader(source)
        else:
            # Default: plain text
            loader = TextLoader(source, encoding="utf-8")

        return loader.load()

    @staticmethod
    def _detect_source_type(source: str) -> str:
        if source.startswith(("http://", "https://")):
            return "url"
        ext = Path(source).suffix.lower()
        return {"pdf": "pdf", ".csv": "csv"}.get(ext, "text")

    # ── LCEL chain construction ────────────────────────────────────────────────

    def _build_chain(self) -> RunnableWithMessageHistory:
        """
        Build the memory-aware LCEL chain.

        Flow:
          user input dict
            → inject document context into prompt
            → ChatPromptTemplate (with MessagesPlaceholder for history)
            → ChatOpenAI
            → StrOutputParser
          wrapped by RunnableWithMessageHistory (auto-manages history per session)
        """

        # Step 1: Prompt template with context slot and history placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),   # ← history injected here
            ("human", "{question}"),
        ])

        # Step 2: Inject the document context before the prompt
        # RunnablePassthrough.assign adds the 'context' key to the input dict
        inject_context = RunnablePassthrough.assign(
            context=RunnableLambda(lambda _: self._context)
        )

        # Step 3: Base chain (no memory yet)
        base_chain = inject_context | prompt | self._llm | self._parser

        # Step 4: Wrap with memory management
        return RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="question",   # user message key in input dict
            history_messages_key="history",  # MessagesPlaceholder variable name
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create the history object for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def _config(self, session_id: str | None = None) -> dict:
        """Build the LangChain config dict for a given session."""
        return {"configurable": {"session_id": session_id or self.session_id}}

    # ── Chat interface ─────────────────────────────────────────────────────────

    def chat(self, question: str, session_id: str | None = None) -> str:
        """
        Send a question and get a response.
        Conversation history for this session is maintained automatically.

        Args:
            question:   The user's question
            session_id: Override the default session_id

        Returns:
            The model's response as a string
        """
        if not self._context:
            raise RuntimeError("No document loaded. Call .load(source) first.")

        return self._chain.invoke(
            {"question": question},
            config=self._config(session_id),
        )

    def stream(self, question: str, session_id: str | None = None) -> Iterator[str]:
        """
        Stream the response token by token.
        Use this in CLI apps and web backends for better UX.

        Usage:
            for token in chat.stream("What is the main finding?"):
                print(token, end="", flush=True)
        """
        if not self._context:
            raise RuntimeError("No document loaded. Call .load(source) first.")

        yield from self._chain.stream(
            {"question": question},
            config=self._config(session_id),
        )

    # ── Session management ─────────────────────────────────────────────────────

    def new_session(self, session_id: str) -> "DocChat":
        """
        Switch to a different session (different conversation history).
        The document stays loaded; only the conversation resets.
        """
        self.session_id = session_id
        return self  # enables chaining: chat.new_session("alice").chat("hello")

    def clear_history(self, session_id: str | None = None) -> None:
        """Clear conversation history for a session."""
        sid = session_id or self.session_id
        if sid in self._sessions:
            self._sessions[sid].clear()
            print(f"  ✓ History cleared for session '{sid}'")

    def get_history(self, session_id: str | None = None) -> list[dict]:
        """
        Return the conversation history as a list of dicts.
        Each dict has 'role' ('human' or 'ai') and 'content'.
        """
        sid = session_id or self.session_id
        history = self._get_session_history(sid)
        result = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "human",   "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "ai",      "content": msg.content})
        return result

    def show_history(self, session_id: str | None = None) -> None:
        """Pretty-print the conversation history."""
        history = self.get_history(session_id)
        sid = session_id or self.session_id
        print(f"\n── Conversation history (session: {sid}) ─────────")
        if not history:
            print("  (empty)")
            return
        for turn in history:
            role  = "You" if turn["role"] == "human" else "AI"
            print(f"\n  {role}: {turn['content'][:200]}"
                  + ("..." if len(turn["content"]) > 200 else ""))
        print()

    def session_stats(self, session_id: str | None = None) -> dict:
        """Return stats for the current session."""
        history = self.get_history(session_id)
        return {
            "session_id":    session_id or self.session_id,
            "turns":         len([m for m in history if m["role"] == "human"]),
            "total_messages": len(history),
            "document":      self._doc_info.get("source", "none"),
        }


# ── Standalone LCEL demos ─────────────────────────────────────────────────────
# These show LCEL concepts in isolation, outside the DocChat class.

def demo_basic_lcel():
    """Show the | operator composing prompt, model, and parser."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical writer. One sentence only."),
        ("human",  "Explain {concept}."),
    ])
    chain  = prompt | llm | StrOutputParser()

    print("Basic LCEL chain:")
    result = chain.invoke({"concept": "gradient descent"})
    print(f"  {result}")
    return result


def demo_parallel_lcel():
    """Show RunnableParallel running two chains on the same input."""
    from langchain_core.runnables import RunnableParallel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    summary_chain  = ChatPromptTemplate.from_messages([
        ("system", "Summarize in one sentence."),
        ("human",  "{text}"),
    ]) | llm | parser

    keywords_chain = ChatPromptTemplate.from_messages([
        ("system", "List 5 key technical terms as comma-separated values."),
        ("human",  "{text}"),
    ]) | llm | parser

    parallel = RunnableParallel(summary=summary_chain, keywords=keywords_chain)
    text     = ("The transformer architecture introduced in 'Attention is All You Need' "
                "uses multi-head self-attention and positional encodings to process "
                "sequences without recurrence, enabling massively parallel training.")

    print("\nParallel LCEL (two chains, one input):")
    result = parallel.invoke({"text": text})
    print(f"  Summary:  {result['summary']}")
    print(f"  Keywords: {result['keywords']}")
    return result


def demo_streaming_lcel():
    """Show token-by-token streaming with LCEL."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Write a two-sentence explanation of {topic}."),
    ])
    chain  = prompt | llm | StrOutputParser()

    print("\nStreaming LCEL (tokens appear live):")
    print("  ", end="")
    for token in chain.stream({"topic": "the attention mechanism"}):
        print(token, end="", flush=True)
    print()
