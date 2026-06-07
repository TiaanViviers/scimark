from __future__ import annotations

import re

from scimark.markdown_blocks import MarkdownBlock, parse_markdown_blocks


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


def is_suspicious_math_paragraph(block: str) -> bool:
    return _is_suspicious_math_paragraph(block)


def _is_existing_confidence_comment(block: MarkdownBlock) -> bool:
    return block.block_type == "unknown" and block.text.strip() == COMMENT


def annotate_low_confidence_math(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    if not lines:
        return markdown, 0

    blocks = parse_markdown_blocks(markdown)
    insertion_points: list[int] = []

    for block_index, block in enumerate(blocks):
        if block.block_type != "paragraph":
            continue

        if not _is_suspicious_math_paragraph(block.text):
            continue

        if block_index > 0 and _is_existing_confidence_comment(blocks[block_index - 1]):
            continue

        insertion_points.append(block.start_line)

    for offset, line_index in enumerate(insertion_points):
        lines.insert(line_index + offset, COMMENT)

    return "\n".join(lines).rstrip() + "\n", len(insertion_points)
