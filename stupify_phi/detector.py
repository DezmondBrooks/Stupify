import re

DEFAULT_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
}

def detect_fields(row: dict, patterns: dict = DEFAULT_PATTERNS) -> dict:
    matches = {}
    for field, value in row.items():
        for name, pattern in patterns.items():
            if isinstance(value, str) and re.search(pattern, value):
                matches[field] = name
    return matches
