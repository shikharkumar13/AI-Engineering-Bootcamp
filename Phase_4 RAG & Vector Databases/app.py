"""
app.py — Streamlit UI for the RAG Document Q&A App

Run with:
    streamlit run app.py

Features:
  - Upload any PDF or text file
  - Choose retrieval strategy (hybrid, similarity, mmr, bm25)
  - Toggle HyDE and source display
  - Chat interface with conversation history
  - Source chunks displayed alongside each answer
  - Indexing stats in the sidebar
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Session state initialization ───────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 RAG Document Q&A")
    st.caption("Phase 04 — Hybrid Retrieval + Re-ranking")
    st.divider()

    # Document upload
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Supported: PDF, TXT",
        type=["pdf", "txt"],
        help="Upload the document you want to query."
    )

    # URL input
    url_input = st.text_input(
        "Or paste a URL",
        placeholder="https://en.wikipedia.org/wiki/...",
    )

    # Index button
    if st.button("📥 Index Document", type="primary", use_container_width=True):
        source = None

        if uploaded_file:
            # Save to temp file
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.getvalue())
                source = f.name
        elif url_input.strip():
            source = url_input.strip()

        if source:
            with st.spinner("Loading embedding model and indexing document..."):
                try:
                    engine = RAGEngine(
                        collection_name="streamlit_rag",
                        persist_dir="./streamlit_chroma_db",
                    )
                    stats = engine.index(source, reset=True)
                    st.session_state.engine      = engine
                    st.session_state.doc_stats   = stats
                    st.session_state.messages    = []
                    st.session_state.engine_ready = True
                    st.success(f"✓ Indexed {stats['chunks_indexed']} chunks!")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.warning("Please upload a file or enter a URL.")

    st.divider()

    # Retrieval settings
    st.subheader("2. Retrieval Settings")

    strategy = st.selectbox(
        "Retrieval strategy",
        options=["hybrid", "similarity", "mmr", "bm25"],
        index=0,
        help=(
            "hybrid: BM25 + vector search merged with RRF\n"
            "similarity: dense vector cosine similarity\n"
            "mmr: diversity-aware similarity\n"
            "bm25: sparse keyword matching"
        ),
    )

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=10, value=5,
        help="How many relevant chunks to retrieve and show the LLM."
    )

    use_hyde = st.toggle(
        "Use HyDE",
        value=False,
        help="Generate a hypothetical answer first, then search with it. "
             "Better for abstract queries; slower."
    )

    show_sources = st.toggle(
        "Show source chunks",
        value=True,
        help="Display the retrieved chunks alongside each answer."
    )

    st.divider()

    # Document stats
    if st.session_state.doc_stats:
        st.subheader("📊 Index Stats")
        s = st.session_state.doc_stats
        st.metric("Source", Path(s["source"]).name if "/" in s["source"] else s["source"])
        st.metric("Chunks indexed", s["chunks_indexed"])
        st.metric("Parent-child", "Yes" if s["parent_child"] else "No")

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

col_chat, col_sources = st.columns([3, 2]) if show_sources else (st.container(), None)

with col_chat:
    st.header("💬 Chat")

    if not st.session_state.engine_ready:
        st.info("👈 Upload a document in the sidebar to begin.")
    else:
        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if question := st.chat_input("Ask a question about your document..."):

            # Display user message
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(
                    f"Retrieving with {strategy} {'+ HyDE' if use_hyde else ''}..."
                ):
                    try:
                        response = st.session_state.engine.ask(
                            question=question,
                            k=top_k,
                            strategy=strategy,
                            use_hyde=use_hyde,
                        )
                        st.write(response.answer)

                        # Store response with sources for the history
                        st.session_state.messages.append({
                            "role":     "assistant",
                            "content":  response.answer,
                            "sources":  response.sources,
                            "strategy": response.strategy,
                        })

                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", "content": error_msg
                        })

        # Show sources for the last response
        if show_sources and st.session_state.messages:
            last = st.session_state.messages[-1]
            if last["role"] == "assistant" and "sources" in last:
                with st.expander(
                    f"📚 Sources ({len(last['sources'])} chunks · {last.get('strategy', '')})",
                    expanded=True,
                ):
                    for i, src in enumerate(last["sources"], 1):
                        page = f" · page {src.page}" if src.page is not None else ""
                        score = f" · score {src.score:.3f}" if src.score else ""
                        st.caption(f"**[Source {i}]** {src.source}{page}{score}")
                        st.text(src.text[:400] + ("..." if len(src.text) > 400 else ""))
                        if i < len(last["sources"]):
                            st.divider()

# ── Source panel (right column when show_sources=True) ─────────────────────────

if show_sources and col_sources:
    with col_sources:
        st.header("🔍 Retrieval Pipeline")
        st.caption(
            "**How hybrid retrieval works:**\n\n"
            "1. **BM25** finds chunks with keyword matches\n"
            "2. **Vector search** finds semantically similar chunks\n"
            "3. **RRF** merges both rankings\n"
            "4. **Cross-encoder** re-ranks the top candidates\n"
            "5. **Parent chunks** replace child chunks for more context"
        )
        st.divider()

        if not st.session_state.engine_ready:
            st.info("Index a document to see retrieval details here.")
        else:
            st.metric(
                "Total indexed chunks",
                st.session_state.engine.stats()["total_chunks"]
            )

# ── Missing Path import ─────────────────────────────────────────────────────────

from pathlib import Path  # needed for sidebar display
