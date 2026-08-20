# advice_journal.py
#
# The Phase 00 Part 6 capstone project — a Daily Advice Journal.
# See the full guide for a step-by-step explanation of every concept used
# here: functions, try/except, classes, file I/O, and control flow.
#
# Setup:
#   pip install requests
# Run:
#   python advice_journal.py     (Windows)
#   python3 advice_journal.py    (Mac)

import json
import os
from datetime import datetime
import requests


def fetch_advice():
    """
    Calls the free Advice Slip API and returns a single piece of advice
    as plain text. Returns None if anything goes wrong, so the calling
    code can decide what to do next instead of crashing.
    """
    try:
        response = requests.get("https://api.adviceslip.com/advice", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["slip"]["advice"]
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch advice right now: {e}")
        return None


class AdviceJournal:
    """Manages saving and loading advice entries to a local file."""

    def __init__(self, filepath="journal.jsonl"):
        self.filepath = filepath

    def add_entry(self, advice_text):
        """Save a new advice entry, with the current date and time."""
        entry = {
            "advice": advice_text,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_entries(self):
        """Read every saved entry back from the file."""
        if not os.path.exists(self.filepath):
            return []

        entries = []
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def print_all_entries(self):
        """Display every saved entry in a friendly format."""
        entries = self.load_entries()
        if not entries:
            print("No entries saved yet.")
            return
        print(f"\nYou have {len(entries)} saved entries:\n")
        for i, entry in enumerate(entries, start=1):
            print(f"  {i}. [{entry['saved_at']}] {entry['advice']}")


def main():
    journal = AdviceJournal()

    print("=" * 50)
    print("  Daily Advice Journal")
    print("=" * 50)

    while True:
        print("\nWhat would you like to do?")
        print("  1. Get new advice and save it")
        print("  2. View all saved advice")
        print("  3. Quit")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            advice = fetch_advice()
            if advice:
                print(f"\nToday's advice: {advice}")
                journal.add_entry(advice)
                print("(saved to your journal)")
        elif choice == "2":
            journal.print_all_entries()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
