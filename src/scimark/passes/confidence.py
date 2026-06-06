from __future__ import annotations

import re


COMMENT = "<!-- scimark: low-confidence-math -->"
PATTERNS = [
    r"_\[",
    r"\[\+\]",
    r"\[−\]",
    r"\[\(\]",
    r"\[\)\]",
    r"_D\[\+\]",
    r"_rD\[−\]",
]


def _looks_like_table(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and "|" in lines[0] and "|" in lines[1]


def _is_suspicious_math_paragraph(block: str) -> bool:
    pattern_hits = sum(len(re.findall(pattern, block)) for pattern in PATTERNS)
    word_count = len(re.findall(r"\b\w+\b", block))
    bracket_underscore_count = block.count("_") + sum(block.count(char) for char in "[]()")
    weird_fragment_count = len(re.findall(r"[_\[\]\(\)\+\-/=]{2,}", block))
    isolated_symbol_count = len(re.findall(r"(?<!\w)[_\[\]\(\)\+\-/=](?!\w)", block))
    ratio = bracket_underscore_count / max(word_count, 1)

    return (
        pattern_hits >= 2
        or ratio >= 0.6
        or (pattern_hits >= 1 and weird_fragment_count >= 3)
        or (weird_fragment_count >= 4 and isolated_symbol_count >= 4)
    )


def annotate_low_confidence_math(markdown: str) -> tuple[str, int]:
    blocks = re.split(r"\n\s*\n", markdown.strip())
    annotated_blocks: list[str] = []
    count = 0

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        if stripped.startswith("```") or stripped.startswith("~~~") or _looks_like_table(stripped):
            annotated_blocks.append(block)
            continue

        if _is_suspicious_math_paragraph(stripped):
            annotated_blocks.append(f"{COMMENT}\n{block}")
            count += 1
        else:
            annotated_blocks.append(block)

    return "\n\n".join(annotated_blocks).rstrip() + "\n", count
