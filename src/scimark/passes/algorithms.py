from __future__ import annotations

import re
from dataclasses import dataclass

from scimark.markdown_blocks import MarkdownBlock, parse_markdown_blocks


COMMENT = "<!-- scimark: low-confidence-algorithm -->"
ALGORITHM_HEADING_RE = re.compile(r"^\s*(Algorithm\s+\d+)\s*[:.]?(?:\s+.*)?$")
ALGORITHM_BODY_PATTERNS = [
    re.compile(r"^\s*(?:Input|Output|Require|Ensure)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:for|while)\b.*\bdo\b", re.IGNORECASE),
    re.compile(r"^\s*(?:if|else if)\b.*\bthen\b", re.IGNORECASE),
    re.compile(r"^\s*(?:else|end|return|repeat|until)\b", re.IGNORECASE),
]


@dataclass(slots=True)
class AlgorithmRegion:
    start_line: int
    end_line: int
    label: str | None = None


def _normalize_line(line: str) -> str:
    normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    normalized = re.sub(r"__(.*?)__", r"\1", normalized)
    normalized = normalized.replace("*", "").replace("_", "")
    return " ".join(normalized.split())


def _is_comment_block(block: MarkdownBlock) -> bool:
    stripped = block.text.strip()
    return block.block_type == "unknown" and stripped.startswith("<!--") and stripped.endswith("-->")


def _algorithm_label(block: MarkdownBlock) -> str | None:
    if block.block_type not in {"paragraph", "heading"} or not block.lines:
        return None

    match = ALGORITHM_HEADING_RE.fullmatch(_normalize_line(block.lines[0]))
    if match is None:
        return None

    return match.group(1)


def _algorithm_signal_count(block: MarkdownBlock) -> int:
    if block.block_type not in {"paragraph", "list", "code_fence"}:
        return 0

    count = 0
    for line in block.lines:
        normalized = _normalize_line(line)
        if any(pattern.search(normalized) for pattern in ALGORITHM_BODY_PATTERNS):
            count += 1
    return count


def _is_algorithm_anchor(block: MarkdownBlock) -> bool:
    signal_count = _algorithm_signal_count(block)
    if signal_count >= 2:
        return True
    return block.block_type in {"list", "code_fence"} and signal_count >= 1


def _is_algorithm_neighbor_block(block: MarkdownBlock) -> bool:
    return block.block_type in {"image", "unknown"} or _algorithm_signal_count(block) > 0


def find_algorithm_regions(markdown: str) -> list[AlgorithmRegion]:
    blocks = parse_markdown_blocks(markdown)
    regions: list[AlgorithmRegion] = []
    index = 0

    while index < len(blocks):
        block = blocks[index]
        label = _algorithm_label(block)

        if label is None and not _is_algorithm_anchor(block):
            index += 1
            continue

        start_index = index
        end_index = index
        index += 1

        while index < len(blocks):
            next_block = blocks[index]
            if _algorithm_label(next_block) is not None:
                break
            if _is_algorithm_neighbor_block(next_block):
                end_index = index
                index += 1
                continue
            break

        regions.append(
            AlgorithmRegion(
                start_line=blocks[start_index].start_line,
                end_line=blocks[end_index].end_line,
                label=label,
            )
        )

    return regions


def annotate_algorithm_blocks(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    if not lines:
        return markdown, 0

    blocks = parse_markdown_blocks(markdown)
    regions = find_algorithm_regions(markdown)
    if not regions:
        return markdown, 0

    block_start_to_previous: dict[int, MarkdownBlock | None] = {}
    for index, block in enumerate(blocks):
        block_start_to_previous[block.start_line] = blocks[index - 1] if index > 0 else None

    insertion_points: list[int] = []
    for region in regions:
        previous = block_start_to_previous.get(region.start_line)
        if previous is not None and previous.block_type == "unknown" and previous.text.strip() == COMMENT:
            continue
        insertion_points.append(region.start_line)

    for offset, line_index in enumerate(insertion_points):
        lines.insert(line_index + offset, COMMENT)

    return "\n".join(lines).rstrip() + "\n", len(insertion_points)
