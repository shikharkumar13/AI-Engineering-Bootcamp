"""
demo.py — Research Copilot

Indexes a small local text file, asks a question it can answer from that
file, then asks a question it can't — watch it fall back to live web
research instead of guessing from irrelevant context.

Run:
    python demo.py
"""

import os
import tempfile

from copilot_graph import ResearchCopilot

SAMPLE_DOC = """
Acme Corp Return Policy

Customers may return any unopened item within 30 days of purchase for a
full refund. Opened items are eligible for store credit only, within 14
days. Items marked "final sale" cannot be returned under any circumstances.

Refunds are processed to the original payment method within 5-7 business
days of the returned item being received at our warehouse.

Acme Corp only ships within the continental United States at this time.
"""


def main():
    print("=" * 60)
    print("RESEARCH COPILOT — Phase 3 (memory) + Phase 4 (RAG) + Phase 5 (agent)")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write(SAMPLE_DOC)
        doc_path = f.name

    copilot = ResearchCopilot(persist_dir=os.path.join(tempfile.gettempdir(), "copilot_demo_db"))

    print("\nIndexing sample return-policy document...")
    copilot.load_document(doc_path, reset=True)

    print("\n--- Turn 1: answerable from local docs ---")
    turn = copilot.chat("What's the return window for an unopened item?")
    print(f"Source: {turn.source}")
    print(f"Answer: {turn.answer}")

    print("\n--- Turn 2: follow-up, relies on memory of turn 1 ---")
    turn = copilot.chat("What about if I already opened it?")
    print(f"Source: {turn.source}")
    print(f"Answer: {turn.answer}")

    print("\n--- Turn 3: NOT in local docs — should fall back to web research ---")
    turn = copilot.chat("What is the current exchange rate from USD to EUR?")
    print(f"Source: {turn.source}")
    print(f"Answer: {turn.answer}")

    print("\n--- Session history ---")
    copilot.show_history()

    os.unlink(doc_path)


if __name__ == "__main__":
    main()
