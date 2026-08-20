"""
Project 2 — Research Copilot
==============================
Combines three phases instead of rebuilding any of them:

  - Phase 4's `RAGEngine` for hybrid retrieval (BM25 + vector + re-rank)
    over the user's own indexed documents.
  - Phase 3's conversational-memory *pattern* — a per-session list of prior
    turns woven back into the next query — applied on top of Phase 4's
    retrieval (RAGEngine itself is single-shot; this is what makes it
    multi-turn).
  - Phase 5's `ResearchAgent` as a fallback: when local retrieval isn't
    confident it actually covers the question (top re-ranked score below a
    threshold), the query is handed to the web-research agent instead of
    letting the RAG chain guess from weak context.

Usage:
    from copilot_graph import ResearchCopilot

    copilot = ResearchCopilot()
    copilot.load_document("handbook.pdf")

    turn = copilot.chat("What's the refund policy?")
    print(turn.answer, turn.source)

    turn = copilot.chat("What about international orders?")  # remembers context
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ── Wire up sibling phase folders so we can import their code directly ────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_5 AI Agents & LangGraph"))
sys.path.insert(0, str(_REPO_ROOT / "Phase_4 RAG & Vector Databases"))

from rag_engine import RAGEngine, RetrievedChunk   # Phase 4
from research_agent import ResearchAgent            # Phase 5

DEFAULT_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class CopilotTurn:
    session_id: str
    message: str
    answer: str
    source: str                                  # "local docs (...)" or "web (...)"
    sources: List[RetrievedChunk] = field(default_factory=list)
    web_notes_count: int = 0


class ResearchCopilot:
    """
    Multi-turn chat over your own documents, with a live web-research
    fallback when local retrieval doesn't confidently cover the question.
    """

    def __init__(
        self,
        collection_name: str = "research_copilot",
        persist_dir: str = "./chroma_db",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.rag = RAGEngine(collection_name=collection_name, persist_dir=persist_dir)
        self.agent = ResearchAgent()
        self.confidence_threshold = confidence_threshold
        self.histories: dict[str, list[dict]] = {}

    # ── Indexing (delegates straight to Phase 4) ────────────────────────

    def load_document(self, source: str, reset: bool = False) -> dict:
        """Index a PDF/text/URL/CSV so it's searchable — see Phase 4's RAGEngine.index()."""
        return self.rag.index(source, reset=reset)

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(self, message: str, session_id: str = "default") -> CopilotTurn:
        history = self.histories.setdefault(session_id, [])
        contextualized_query = self._contextualize(message, history)

        rag_response = self.rag.ask(contextualized_query, k=5, strategy="hybrid")
        top_score = rag_response.sources[0].score if rag_response.sources else 0.0

        if top_score < self.confidence_threshold:
            # Local docs don't confidently cover this — break out to live web research
            # instead of letting the RAG chain answer from weak/irrelevant context.
            agent_result = self.agent.research(message, thread_id=session_id, verbose=False)
            turn = CopilotTurn(
                session_id=session_id,
                message=message,
                answer=agent_result["final_message"],
                source="web (Phase 5 research agent — local docs had no confident match)",
                sources=[],
                web_notes_count=len(agent_result["notes"]),
            )
        else:
            turn = CopilotTurn(
                session_id=session_id,
                message=message,
                answer=rag_response.answer,
                source=f"local docs ({rag_response.strategy}, "
                f"top score {top_score:.2f})",
                sources=rag_response.sources,
            )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": turn.answer})
        return turn

    def _contextualize(self, message: str, history: list[dict], max_turns: int = 4) -> str:
        """Weave the last few turns into the query — the Phase 3 memory pattern,
        applied here since RAGEngine's own chain is single-shot with no history."""
        if not history:
            return message
        recent = history[-max_turns:]
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in recent)
        return f"Conversation so far:\n{convo}\n\nNew question: {message}"

    # ── Session management (mirrors Phase 3's DocChat API) ─────────────

    def get_history(self, session_id: str = "default") -> list[dict]:
        return self.histories.get(session_id, [])

    def show_history(self, session_id: str = "default"):
        for turn in self.get_history(session_id):
            print(f"  {turn['role']}: {turn['content'][:200]}")

    def clear_history(self, session_id: str = "default"):
        self.histories[session_id] = []
