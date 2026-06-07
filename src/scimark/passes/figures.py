from __future__ import annotations

import re

from scimark.markdown_blocks import MarkdownBlock, parse_markdown_blocks


CAPTION_LINE_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s+\d+\s*[:.].*")
SCIMARK_COMMENT_RE = re.compile(r"^\s*<!--\s*scimark:.*?-->\s*$")


def _is_comment_line(line: str) -> bool:
    return bool(SCIMARK_COMMENT_RE.fullmatch(line))


def _is_caption_block(block: MarkdownBlock) -> bool:
    return (
        block.block_type == "paragraph"
        and len(block.lines) == 1
        and bool(CAPTION_LINE_RE.fullmatch(block.lines[0]))
    )


def _is_comment_block(block: MarkdownBlock) -> bool:
    return block.block_type == "unknown" and all(_is_comment_line(line) for line in block.lines)


def _is_figure_neighbor_block(block: MarkdownBlock) -> bool:
    return block.block_type == "image" or _is_comment_block(block)


def _find_nearby_figure_blocks_before(
    blocks: list[MarkdownBlock], caption_index: int
) -> tuple[int, int] | None:
    index = caption_index - 1
    if index < 0:
        return None

    seen_image = False
    while index >= 0 and _is_figure_neighbor_block(blocks[index]):
        seen_image = seen_image or blocks[index].block_type == "image"
        index -= 1

    if not seen_image:
        return None

    return index + 1, caption_index


def _find_nearby_figure_blocks_after(
    blocks: list[MarkdownBlock], caption_index: int
) -> tuple[int, int] | None:
    index = caption_index + 1
    if index >= len(blocks):
        return None

    seen_image = False
    while index < len(blocks) and _is_figure_neighbor_block(blocks[index]):
        seen_image = seen_image or blocks[index].block_type == "image"
        index += 1

    if not seen_image:
        return None

    return caption_index, index - 1


def _normalized_figure_block_lines(blocks: list[MarkdownBlock], caption_line: str) -> list[str]:
    normalized: list[str] = []
    for block in blocks:
        normalized.extend(block.lines)
    normalized.append(caption_line)
    return normalized


def keep_captions_with_figures(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    if not lines:
        return markdown, 0

    adjusted = 0
    changed = True

    while changed:
        changed = False
        blocks = parse_markdown_blocks("\n".join(lines))

        for block_index, block in enumerate(blocks):
            if not _is_caption_block(block):
                continue

            before_group = _find_nearby_figure_blocks_before(blocks, block_index)
            if before_group is not None:
                start_index, caption_index = before_group
                group_blocks = blocks[start_index:caption_index]
                normalized = _normalized_figure_block_lines(group_blocks, block.lines[0])
                slice_start = blocks[start_index].start_line
                slice_end = block.end_line + 1
                if normalized != lines[slice_start:slice_end]:
                    lines[slice_start:slice_end] = normalized
                    adjusted += 1
                    changed = True
                    break

            after_group = _find_nearby_figure_blocks_after(blocks, block_index)
            if after_group is not None:
                caption_index, end_index = after_group
                group_blocks = blocks[caption_index + 1 : end_index + 1]
                normalized = _normalized_figure_block_lines(group_blocks, block.lines[0])
                slice_start = block.start_line
                slice_end = blocks[end_index].end_line + 1
                if normalized != lines[slice_start:slice_end]:
                    lines[slice_start:slice_end] = normalized
                    adjusted += 1
                    changed = True
                    break

    return "\n".join(lines).rstrip() + "\n", adjusted
