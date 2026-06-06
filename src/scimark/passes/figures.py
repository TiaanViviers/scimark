from __future__ import annotations

import re


IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*]\([^)]+\)\s*$")
CAPTION_LINE_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s+\d+\s*[:.].*")
SCIMARK_COMMENT_RE = re.compile(r"^\s*<!--\s*scimark:.*?-->\s*$")


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_comment(line: str) -> bool:
    return bool(SCIMARK_COMMENT_RE.fullmatch(line))


def _is_image(line: str) -> bool:
    return bool(IMAGE_LINE_RE.fullmatch(line))


def _is_caption(line: str) -> bool:
    return bool(CAPTION_LINE_RE.fullmatch(line))


def _is_ignorable(line: str) -> bool:
    return _is_blank(line) or _is_comment(line)


def _normalize_figure_block(block_lines: list[str], caption_line: str) -> list[str]:
    normalized = [line for line in block_lines if not _is_blank(line)]
    normalized.append(caption_line)
    return normalized


def _find_nearby_image_block_before(lines: list[str], caption_index: int) -> tuple[int, int] | None:
    index = caption_index - 1
    if index < 0:
        return None

    seen_image = False
    while index >= 0 and (_is_ignorable(lines[index]) or _is_image(lines[index])):
        seen_image = seen_image or _is_image(lines[index])
        index -= 1

    if not seen_image:
        return None

    block_start = index + 1
    while block_start < caption_index and _is_blank(lines[block_start]):
        block_start += 1

    return block_start, caption_index


def _find_nearby_image_block_after(lines: list[str], caption_index: int) -> tuple[int, int] | None:
    index = caption_index + 1
    if index >= len(lines):
        return None

    seen_image = False
    while index < len(lines) and (_is_ignorable(lines[index]) or _is_image(lines[index])):
        seen_image = seen_image or _is_image(lines[index])
        index += 1

    if not seen_image:
        return None

    return caption_index, index - 1


def keep_captions_with_figures(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    if not lines:
        return markdown, 0

    adjusted = 0
    index = 0

    while index < len(lines):
        if not _is_caption(lines[index]):
            index += 1
            continue

        before_block = _find_nearby_image_block_before(lines, index)
        if before_block is not None:
            block_start, caption_index = before_block
            block_lines = lines[block_start:caption_index]
            normalized = _normalize_figure_block(block_lines, lines[caption_index])
            if normalized != lines[block_start : caption_index + 1]:
                lines[block_start : caption_index + 1] = normalized
                adjusted += 1
            index = block_start + len(normalized)
            continue

        after_block = _find_nearby_image_block_after(lines, index)
        if after_block is not None:
            caption_index, block_end = after_block
            block_lines = lines[caption_index + 1 : block_end + 1]
            normalized = _normalize_figure_block(block_lines, lines[caption_index])
            replacement = normalized
            if replacement != lines[caption_index : block_end + 1]:
                lines[caption_index : block_end + 1] = replacement
                adjusted += 1
            index = caption_index + len(replacement)
            continue

        index += 1

    return "\n".join(lines).rstrip() + "\n", adjusted
