"""
demo.py — Travel Itinerary Planner walkthrough

Plans two trips (calling plan_trip() twice on purpose, to prove the
thread_id fix doesn't leak state between calls) and asks one follow-up
question about a destination.

Run: python demo.py
"""

from itinerary_planner import TravelPlanner


def print_itinerary(itinerary):
    print(f"  {itinerary.destination}: {itinerary.num_days} day(s)")
    for day in itinerary.days:
        theme = f" - {day.theme}" if day.theme else ""
        print(f"    Day {day.day_number}{theme}")
        for activity in day.activities:
            time = f" ({activity.estimated_time})" if activity.estimated_time else ""
            print(f"      - [{activity.category}] {activity.name}{time}: {activity.description}")
    if itinerary.practical_tips:
        print("    Practical tips:")
        for tip in itinerary.practical_tips:
            print(f"      - {tip}")
    if itinerary.sources:
        print(f"    Sources: {itinerary.sources}")


def main():
    planner = TravelPlanner()

    print("=" * 60)
    print("1. Plan a 3-day trip to Kyoto, Japan")
    print("=" * 60)
    kyoto_trip = planner.plan_trip("Kyoto, Japan", num_days=3, interests=["history", "food"])
    print_itinerary(kyoto_trip)

    print()
    print("=" * 60)
    print("2. Plan a SECOND trip to the same destination - proves the")
    print("   thread_id fix: this run must not repeat or bleed into run 1")
    print("=" * 60)
    kyoto_trip_2 = planner.plan_trip("Kyoto, Japan", num_days=2, interests=["nature"])
    print_itinerary(kyoto_trip_2)

    print()
    print("=" * 60)
    print("3. Ask a one-off follow-up question")
    print("=" * 60)
    question = "What's the best way to get around Kyoto without a car?"
    answer = planner.ask_about_destination("Kyoto, Japan", question)
    print(f"  Q: {question}")
    print(f"  A: {answer}")


if __name__ == "__main__":
    main()
