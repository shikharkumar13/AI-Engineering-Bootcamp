"""
streamlit_app.py — Production Frontend

Calls the FastAPI backend over HTTP (not direct AI logic calls) — this is
the architectural separation that lets backend and frontend scale and
deploy independently, exactly as discussed in Phase 08 Section 3.2.

Run with:
    streamlit run streamlit_app.py

Configure the backend URL and API key via environment variables or
Streamlit secrets (.streamlit/secrets.toml):
    API_URL = "http://localhost:8000"
    API_KEY = "demo-key-123"
"""

import os
import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "demo-key-123")

st.set_page_config(page_title="Production AI Service", page_icon="🤖", layout="wide")


# ── Session state ───────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None


# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Production AI Service")
    st.caption("Phase 08 — FastAPI backend + Docker + RAGAS + Langfuse")
    st.divider()

    st.subheader("Backend connection")
    st.code(f"API_URL = {API_URL}", language="text")

    # Health check
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        st.success(f"✓ Backend reachable ({health['status']})")
    except Exception:
        st.error("✗ Cannot reach backend. Is main.py running?")

    try:
        ready = httpx.get(f"{API_URL}/ready", timeout=5.0).json()
        if ready["ready"]:
            st.success(f"✓ Ready — {ready['checks'].get('indexed_chunks', 0)} chunks indexed")
        else:
            st.warning("⚠ Backend not ready — index a document first")
    except Exception:
        pass

    st.divider()

    # Document upload
    st.subheader("📄 Index a Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if st.button("Index Document", type="primary", use_container_width=True):
        if uploaded_file:
            with st.spinner("Indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = httpx.post(
                        f"{API_URL}/index",
                        files=files,
                        headers={"X-API-Key": API_KEY},
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success(f"✓ Indexed {result['chunks_indexed']} chunks")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.warning("Upload a file first.")

    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ───────────────────────────────────────────────────────────────

st.header("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (score: {src['score']:.3f})")
                    st.text(src["excerpt"])

if prompt := st.chat_input("Ask a question about the indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # Stream from the FastAPI backend
            with httpx.stream(
                "POST", f"{API_URL}/chat/stream",
                json={"message": prompt, "k": 4},
                headers={"X-API-Key": API_KEY},
                timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    error_detail = response.json().get("detail", "Unknown error")
                    full_response = f"⚠ Error: {error_detail}"
                    placeholder.error(full_response)
                else:
                    for chunk in response.iter_text():
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

        except httpx.ConnectError:
            full_response = "⚠ Cannot connect to backend. Is main.py running?"
            placeholder.error(full_response)
        except Exception as e:
            full_response = f"⚠ Error: {e}"
            placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # After streaming, also fetch the non-streaming response once for sources +
    # trace_id (since streaming endpoint doesn't return structured metadata)
    if not full_response.startswith("⚠"):
        try:
            full_result = httpx.post(
                f"{API_URL}/chat",
                json={"message": prompt, "k": 4},
                headers={"X-API-Key": API_KEY},
                timeout=30.0,
            ).json()
            st.session_state.messages[-1]["sources"] = full_result.get("sources", [])
            st.session_state.last_trace_id = full_result.get("trace_id")
        except Exception:
            pass


# ── Feedback ────────────────────────────────────────────────────────────────────

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key="thumbs_up"):
            if st.session_state.last_trace_id:
                httpx.post(
                    f"{API_URL}/feedback",
                    json={"trace_id": st.session_state.last_trace_id, "is_positive": True},
                    headers={"X-API-Key": API_KEY},
                )
                st.toast("Feedback recorded — thanks!")
    with col2:
        if st.button("👎", key="thumbs_down"):
            if st.session_state.last_trace_id:
                httpx.post(
                    f"{API_URL}/feedback",
                    json={"trace_id": st.session_state.last_trace_id, "is_positive": False},
                    headers={"X-API-Key": API_KEY},
                )
                st.toast("Feedback recorded — we'll review this.")
