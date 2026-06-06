from __future__ import annotations

import re


MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "﹣": "-",
        "－": "-",
    }
)


def cleanup_math_text(markdown: str) -> str:
    cleaned = markdown.translate(MINUS_TRANSLATION)
    cleaned = re.sub(r"(\d)\s+_\._\s+(\d)", r"\1.\2", cleaned)
    cleaned = re.sub(r"(\d)\s+_/_\s+(\d)", r"\1/\2", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned
