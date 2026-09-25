import re

def extract_numbers(address):

    if not isinstance(address, str):
        return []

    return re.findall(r"\d+", address)

LANDMARK_WORDS = [
    "near",
    "opp",
    "opposite",
    "behind",
    "beside",
    "next"
]

def extract_landmarks(text):

    text = text.lower()

    results = []

    for word in LANDMARK_WORDS:

        if word in text:
            results.append(word)

    return " ".join(results)