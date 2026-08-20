"""
crew.py — Content Desk core logic
====================================
Combines:

  - Phase 6's `ContentFactory` — the research → write → edit → parallel
    multi-platform-format crew — used directly, not reimplemented.
  - Phase 7's fine-tuned support-ticket model, called over plain HTTP,
    exactly the way Phase 7's own `demo_client.py` does. It is deliberately
    NOT imported as a Python module: Phase 7's other files
    (`train.py`, `inference_api.py`) pull in CUDA-only packages
    (`unsloth`, `bitsandbytes`) that won't import on a machine without a
    GPU, so the only portable way to use that model from here is the same
    way any other HTTP client would — which is also exactly how it'd be
    consumed in a real deployment, where the fine-tuned model lives on its
    own GPU-backed service.

`compare_support_response()` exists to make Phase 7's own decision
framework concrete: fine-tuning wins for a narrow, structured task (support
tickets, forced into Issue/Resolution/Next Steps); a general-purpose crew
agent is what you reach for on an open-ended task instead. Running the same
support message through both, side by side, is that comparison.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Phase_6 Multi-Agent Systems"))

from content_factory import ContentFactory, PipelineResult   # Phase 6

# Set this to wherever Phase 7's inference_api.py is being served from
# (e.g. an ngrok URL from the Colab instance running it). Left unset, the
# fine-tuned side of compare_support_response() is skipped gracefully —
# same behavior as Phase 7's own demo_client.py when the API isn't up.
FINETUNED_MODEL_URL = os.getenv("FINETUNED_MODEL_URL", "").rstrip("/")


@dataclass
class SupportComparison:
    message: str
    finetuned_available: bool
    finetuned_response: Optional[dict]
    crew_response: str
    crew_latency_s: float


class ContentDesk:
    """Core AI logic for the content desk — kept separate from main.py's
    HTTP layer, the same separation Phase 8's rag_service.py/main.py use."""

    def __init__(self, verbose: bool = False):
        self.factory = ContentFactory(verbose=verbose)

    # ── Capability 1: multi-agent content generation (Phase 6) ─────────

    def generate(self, topic: str) -> PipelineResult:
        """Full research → write → edit → {blog, linkedin, twitter} pipeline."""
        return self.factory.run(topic)

    # ── Capability 2: fine-tune vs. general-agent comparison (Phase 7) ─

    def compare_support_response(self, message: str) -> SupportComparison:
        finetuned_response = None
        finetuned_available = False

        if FINETUNED_MODEL_URL:
            try:
                r = httpx.post(
                    f"{FINETUNED_MODEL_URL}/chat",
                    json={"message": message},
                    timeout=30.0,
                )
                r.raise_for_status()
                finetuned_response = r.json()
                finetuned_available = True
            except Exception:
                # Same graceful degrade as Phase 7's demo_client.py when the
                # GPU-backed API isn't reachable — this is expected whenever
                # FINETUNED_MODEL_URL points at a Colab session that isn't running.
                finetuned_available = False

        t0 = time.time()
        crew_response = self._general_agent_response(message)
        crew_latency_s = round(time.time() - t0, 2)

        return SupportComparison(
            message=message,
            finetuned_available=finetuned_available,
            finetuned_response=finetuned_response,
            crew_response=crew_response,
            crew_latency_s=crew_latency_s,
        )

    def _general_agent_response(self, message: str) -> str:
        """A single general-purpose agent with no domain fine-tuning —
        the baseline Phase 7's decision framework compares against."""
        from crewai import Agent, Task, Crew, Process

        agent = Agent(
            role="Support Assistant",
            goal="Resolve the customer's issue clearly and politely",
            backstory="A general-purpose support agent with no domain-specific fine-tuning.",
            verbose=False,
        )
        task = Task(
            description=f"Respond to this customer support message:\n\n{message}",
            expected_output="A clear, professional support response, 2-4 sentences.",
            agent=agent,
        )
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()
        return result.raw
