"""
demo.py — Smart Inbox Triage

Runs a handful of sample support tickets through InboxTriager and prints
the extracted fields plus the drafted reply for each one, then a session
cost report.

Run:
    python demo.py
"""

from triage import InboxTriager

SAMPLE_TICKETS = [
    """
    Subject: Charged twice this month!!

    Hi, I just noticed TWO charges of $49.99 on my card this month for my
    Pro subscription. This is the second time this has happened. I need
    this fixed today or I'm cancelling and disputing the charge with my bank.

    - Maria
    """,
    """
    Subject: How do I export my data?

    Hello, quick question — is there a way to export all my project data
    as a CSV or JSON file? Couldn't find it in the settings menu. Thanks!
    """,
    """
    Subject: App crashes on startup after update

    Since updating to version 4.2 this morning, the app crashes immediately
    on launch on my iPhone 13, iOS 17.4. I've tried reinstalling twice.
    Attached the crash log. This is blocking my whole team from working.
    """,
]


def main():
    print("=" * 60)
    print("SMART INBOX TRIAGE — Phase 1 (LLM client) + Phase 2 (extraction)")
    print("=" * 60)

    triager = InboxTriager()
    results = triager.batch_triage(SAMPLE_TICKETS)

    for i, r in enumerate(results, start=1):
        print(f"\n{'-' * 60}")
        print(f"Ticket {i}")
        print(f"{'-' * 60}")
        print(f"  Category:  {r.extraction.category.value}")
        print(f"  Priority:  {r.extraction.priority.value}")
        print(f"  Sentiment: {r.extraction.sentiment.value}")
        print(f"  Summary:   {r.extraction.summary}")
        if r.extraction.action_items:
            print("  Action items:")
            for item in r.extraction.action_items:
                print(f"    - {item.task}")
        print(f"\n  Draft reply ({r.draft_reply_provider}, "
              f"${r.draft_reply_cost_usd:.6f}):")
        print(f"  {r.draft_reply}")

    print(f"\n{'=' * 60}")
    triager.report()


if __name__ == "__main__":
    main()
