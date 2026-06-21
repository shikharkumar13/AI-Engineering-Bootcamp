# phase00_part4_advice_solution.py
#
# A worked solution to the Section 9 checkpoint project.
# Try writing your own version FIRST — this is here to check your work
# against afterward, not to copy before attempting it yourself.

import requests


def get_random_advice() -> str:
    """
    Calls the free Advice Slip API and returns a single piece of advice
    as plain text.

    The API's response looks like this:
        {"slip": {"id": 117, "advice": "Be kind to yourself today."}}
    so to get just the text, we need response.json()["slip"]["advice"]
    """
    response = requests.get("https://api.adviceslip.com/advice")

    if response.status_code != 200:
        return f"Something went wrong (status code {response.status_code})"

    data = response.json()
    return data["slip"]["advice"]


if __name__ == "__main__":
    advice = get_random_advice()
    print(f"Today's advice: {advice}")
