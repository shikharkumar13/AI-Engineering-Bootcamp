"""
demo.py — Phase 03: LangChain & Orchestration

Run: python demo.py

Demonstrates:
  1. LCEL chain composition (| operator)
  2. Parallel chains (RunnableParallel)
  3. Streaming with LCEL
  4. Document loading from different sources
  5. Text splitting and chunking
  6. Multi-turn chat with memory
  7. Multiple sessions (different conversation histories)
  8. Session stats and history inspection
"""

import os
import time
import textwrap
from pathlib import Path
from dotenv import load_dotenv

from doc_chat import DocChat, demo_basic_lcel, demo_parallel_lcel, demo_streaming_lcel
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DIVIDER = "═" * 62
SAMPLE_TEXT_PATH = "sample_doc.txt"


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def create_sample_document():
    """Create a sample text document for demos."""
    content = textwrap.dedent("""
    The Transformer Architecture: A Technical Overview
    ==================================================

    Introduction
    ------------
    The Transformer is a deep learning model architecture introduced in the 2017
    paper "Attention Is All You Need" by Vaswani et al. It has become the
    foundation for large language models including GPT, BERT, and Claude.

    Unlike recurrent neural networks (RNNs), Transformers process entire sequences
    simultaneously using a mechanism called self-attention. This enables massively
    parallel training on modern GPU/TPU hardware.

    Self-Attention Mechanism
    ------------------------
    The core of the Transformer is multi-head self-attention. For each token in
    the input sequence, attention computes three vectors:
    - Query (Q): what this token is looking for
    - Key (K): what this token offers to others
    - Value (V): the actual information this token contains

    Attention scores are computed as: softmax(QK^T / sqrt(d_k)) * V

    The scaling factor sqrt(d_k) prevents the dot products from growing too large
    in high dimensions, which would push softmax into regions with tiny gradients.

    Multi-head attention runs this process h times in parallel with different
    learned weight matrices, allowing the model to attend to information from
    different representation subspaces simultaneously.

    Positional Encoding
    -------------------
    Since self-attention is permutation-invariant (it treats the input as a set,
    not a sequence), Transformers add positional encodings to the input embeddings.
    The original paper used sinusoidal functions of different frequencies.

    Modern models like GPT use learned positional embeddings instead.

    Architecture Variants
    ---------------------
    Three main Transformer variants exist:
    1. Encoder-only (BERT): best for understanding tasks (classification, NER)
    2. Decoder-only (GPT, Claude): best for generation tasks (text completion)
    3. Encoder-Decoder (T5, BART): best for sequence-to-sequence tasks (translation)

    Training at Scale
    -----------------
    Transformers are trained using next-token prediction (decoder models) or
    masked language modeling (encoder models) on internet-scale text corpora.
    GPT-3 was trained on 45TB of text data; GPT-4 on significantly more.

    The key insight: the same pre-training objective on enough data produces
    models that generalize to many downstream tasks without task-specific training.
    This is called emergent capability.

    Limitations
    -----------
    - Quadratic complexity in sequence length (attention over N tokens costs O(N^2))
    - Fixed context window (though modern models push this to millions of tokens)
    - Hallucination: models generate plausible but incorrect information
    - No persistent memory between conversations
    """).strip()

    Path(SAMPLE_TEXT_PATH).write_text(content)
    return SAMPLE_TEXT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: LCEL composition
# ─────────────────────────────────────────────────────────────────────────────

def demo_1_lcel():
    section("DEMO 1 — LCEL Chain Composition (the | operator)")
    demo_basic_lcel()
    demo_parallel_lcel()
    demo_streaming_lcel()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: Document loading
# ─────────────────────────────────────────────────────────────────────────────

def demo_2_document_loading():
    section("DEMO 2 — Document Loading")

    # Text file
    print("\n▸ TextLoader — loading a .txt file")
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(SAMPLE_TEXT_PATH)
    docs   = loader.load()
    print(f"  Documents loaded: {len(docs)}")
    print(f"  Characters:       {len(docs[0].page_content):,}")
    print(f"  Metadata:         {docs[0].metadata}")
    print(f"  Preview:          '{docs[0].page_content[:100]}...'")

    # Web page
    print("\n▸ WebBaseLoader — loading a URL")
    from langchain_community.document_loaders import WebBaseLoader
    try:
        loader = WebBaseLoader("https://en.wikipedia.org/wiki/Attention_(machine_learning)")
        web_docs = loader.load()
        print(f"  Documents loaded: {len(web_docs)}")
        print(f"  Characters:       {len(web_docs[0].page_content):,}")
        print(f"  Source:           {web_docs[0].metadata.get('source')}")
    except Exception as e:
        print(f"  (Skipped — network error: {type(e).__name__})")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3: Text splitting
# ─────────────────────────────────────────────────────────────────────────────

def demo_3_text_splitting():
    section("DEMO 3 — Text Splitting")

    from langchain_community.document_loaders import TextLoader

    docs    = TextLoader(SAMPLE_TEXT_PATH).load()
    text    = docs[0].page_content

    print(f"\n  Original document: {len(text):,} chars")

    # Compare different chunk sizes
    for chunk_size, overlap in [(500, 50), (1000, 200), (2000, 400)]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_text(text)
        print(f"\n  chunk_size={chunk_size}, overlap={overlap}:")
        print(f"    → {len(chunks)} chunks, avg {sum(len(c) for c in chunks)//len(chunks)} chars each")

    # Show what a chunk actually looks like
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)  # uses split_documents to preserve metadata

    print(f"\n  Chunk 1 (with metadata preserved):")
    print(f"    Metadata:  {chunks[0].metadata}")
    print(f"    Content:   '{chunks[0].page_content[:200]}...'")
    print(f"\n  Chunk 2 (notice the 50-char overlap with chunk 1):")
    print(f"    Content:   '{chunks[1].page_content[:200]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4: Basic Q&A with document
# ─────────────────────────────────────────────────────────────────────────────

def demo_4_basic_qa():
    section("DEMO 4 — Document Q&A")

    chat = DocChat(session_id="demo-basic")
    chat.load(SAMPLE_TEXT_PATH)

    questions = [
        "What is the main advantage of Transformers over RNNs?",
        "What are Q, K, and V vectors in attention?",
        "What is the scaling factor used in attention, and why?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        print(f"  A: ", end="")
        for token in chat.stream(q):
            print(token, end="", flush=True)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5: Multi-turn conversation with memory
# ─────────────────────────────────────────────────────────────────────────────

def demo_5_memory():
    section("DEMO 5 — Multi-turn Conversation Memory")

    chat = DocChat(session_id="memory-demo")
    chat.load(SAMPLE_TEXT_PATH)

    # Turn 1 — ask about a concept
    print("\n  Turn 1:")
    print(f"  User: What are the three Transformer architecture variants?")
    r1 = chat.chat("What are the three Transformer architecture variants?")
    print(f"  AI:   {r1[:250]}...")

    # Turn 2 — follow-up (tests memory: "the second one" refers back to Turn 1)
    print("\n  Turn 2 (follow-up — refers back to Turn 1):")
    print(f"  User: Which of those is used for text generation tasks like GPT?")
    r2 = chat.chat("Which of those is used for text generation tasks like GPT?")
    print(f"  AI:   {r2[:250]}")

    # Turn 3 — another follow-up
    print("\n  Turn 3:")
    print(f"  User: And what is the limitation related to sequence length?")
    r3 = chat.chat("And what is the limitation related to sequence length?")
    print(f"  AI:   {r3[:250]}")

    # Show the full conversation history
    chat.show_history()

    print(f"\n  Session stats: {chat.session_stats()}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6: Multiple sessions
# ─────────────────────────────────────────────────────────────────────────────

def demo_6_multiple_sessions():
    section("DEMO 6 — Multiple Independent Sessions")

    # One DocChat instance, two sessions — completely separate histories
    chat = DocChat()
    chat.load(SAMPLE_TEXT_PATH)

    print("\n  Simulating two users asking different questions about the same doc:")

    print("\n  [Session: alice]")
    r = chat.chat("What problem does self-attention solve?", session_id="alice")
    print(f"  AI: {r[:180]}...")

    print("\n  [Session: bob]")
    r = chat.chat("How was GPT-3 trained?", session_id="bob")
    print(f"  AI: {r[:180]}...")

    print("\n  [Session: alice — follow-up, remembers her first question]")
    r = chat.chat("Can you give me a more technical explanation?", session_id="alice")
    print(f"  AI: {r[:180]}...")

    print("\n  [Session: bob — his follow-up, separate history from alice]")
    r = chat.chat("What is the difference between pre-training and fine-tuning?", session_id="bob")
    print(f"  AI: {r[:180]}...")

    # Inspect each session's history independently
    print("\n  Alice's history:")
    for turn in chat.get_history("alice"):
        role = "User" if turn["role"] == "human" else "AI"
        print(f"    {role}: {turn['content'][:100]}...")

    print("\n  Bob's history:")
    for turn in chat.get_history("bob"):
        role = "User" if turn["role"] == "human" else "AI"
        print(f"    {role}: {turn['content'][:100]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7: URL loading and Q&A
# ─────────────────────────────────────────────────────────────────────────────

def demo_7_url_loading():
    section("DEMO 7 — Loading from a URL")

    chat = DocChat(session_id="url-demo")

    try:
        info = chat.load("https://en.wikipedia.org/wiki/Attention_(machine_learning)")
        print(f"\n  Loaded: {info['chunks']} chunks, {info['total_chars']:,} chars")

        print("\n  Q: What are the main applications of attention mechanisms?")
        r = chat.chat("What are the main applications of attention mechanisms?")
        print(f"  A: {r[:300]}")

    except Exception as e:
        print(f"  (Skipped — network/content error: {type(e).__name__}: {e})")
        print("  → Try running this with an internet connection.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 8: LangSmith tracing visibility
# ─────────────────────────────────────────────────────────────────────────────

def demo_8_langsmith():
    section("DEMO 8 — LangSmith Tracing")

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    if tracing == "true":
        print(f"\n  ✓ LangSmith tracing is ACTIVE")
        print(f"    Project: {project}")
        print(f"    Every chain.invoke() above created a trace.")
        print(f"    View at: https://smith.langchain.com")
        print(f"\n  Making one more call to generate a clearly labeled trace...")

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser

        chain  = (
            ChatPromptTemplate.from_messages([("human", "{q}")])
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )

        result = chain.invoke(
            {"q": "What is Layer Normalization in Transformers?"},
            config={
                "run_name":  "phase03-langsmith-demo",
                "tags":      ["demo", "phase03"],
                "metadata":  {"demo": "langsmith", "user": "learner"},
            }
        )
        print(f"  Response: {result[:150]}...")
        print(f"\n  Find this trace by searching 'phase03-langsmith-demo' in LangSmith.")
    else:
        print("\n  LangSmith tracing is NOT active.")
        print("  To enable, add to your .env file:")
        print("    LANGCHAIN_TRACING_V2=true")
        print("    LANGCHAIN_API_KEY=ls__your_key")
        print("  Get a free key at: https://smith.langchain.com")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  Phase 03 — LangChain & Orchestration: All Demos")
    print(DIVIDER)

    # Create the sample document for demos
    create_sample_document()
    print(f"  ✓ Sample document created: {SAMPLE_TEXT_PATH}")

    demo_1_lcel()
    demo_2_document_loading()
    demo_3_text_splitting()
    demo_4_basic_qa()
    demo_5_memory()
    demo_6_multiple_sessions()
    demo_7_url_loading()
    demo_8_langsmith()

    print(f"\n{DIVIDER}")
    print("  ✅ Phase 03 complete.")
    print("  You built: multi-turn document Q&A with LCEL + session memory.")
    print("  Next: Phase 04 — RAG & Vector Databases")
    print("  (for documents larger than what fits in the context window)")
    print(DIVIDER)
