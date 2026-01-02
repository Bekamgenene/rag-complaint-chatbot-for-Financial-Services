import re
from typing import List

__all__: List[str] = ["clean_text"]

_pattern = re.compile(r"[^a-z0-9\s]")
_boilerplate_patterns = [
    re.compile(r"i am writing to file a complaint[\s\S]*", flags=re.IGNORECASE),
    re.compile(r"dear (?:sir|madam|to whom it may concern)[\s\S]*", flags=re.IGNORECASE),
]


def _remove_boilerplate(text: str) -> str:
    """Remove common boilerplate phrases that add noise to embeddings."""
    lowered = text.lower()
    for pat in _boilerplate_patterns:
        lowered = pat.sub(" ", lowered)
    return lowered


def clean_text(text: str) -> str:
    """Standard text cleaning for embeddings.

    Steps
    -----
    1. Lower-case the text.
    2. Remove boilerplate sentences (very rough heuristics).
    3. Strip special characters, keeping only [a-z0-9 ] and consolidating spaces.
    4. Trim whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = _remove_boilerplate(text)
    text = _pattern.sub(" ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
