from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


BlockType = Literal[
    "heading",
    "paragraph",
    "image",
    "table",
    "code_fence",
    "list",
    "blockquote",
    "picture_text",
    "unknown",
]

IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*]\([^)]+\)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+\.)\s+")
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$")
SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


@dataclass(slots=True)
class MarkdownBlock:
    block_type: BlockType
    start_line: int
    end_line: int
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _fence_delimiter(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("```"):
        return "```"
    if stripped.startswith("~~~"):
        return "~~~"
    return None


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_line(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(SEPARATOR_CELL_RE.fullmatch(cell.replace(" ", "")) for cell in cells)


def _is_table_start(lines: list[str], index: int) -> bool:
    return index + 1 < len(lines) and "|" in lines[index] and _is_separator_line(lines[index + 1])


def _is_picture_text_start(line: str) -> bool:
    return "Start of picture text" in line


def _is_picture_text_end(line: str) -> bool:
    return "End of picture text" in line


def _is_unknown_start(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (COMMENT_RE.fullmatch(stripped) is not None or stripped.startswith("<"))


def parse_markdown_blocks(markdown: str) -> list[MarkdownBlock]:
    lines = markdown.splitlines()
    blocks: list[MarkdownBlock] = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        start = index
        line = lines[index]

        if _fence_delimiter(line):
            delimiter = _fence_delimiter(line)
            index += 1
            while index < len(lines):
                if lines[index].strip().startswith(delimiter):
                    index += 1
                    break
                index += 1
            blocks.append(MarkdownBlock("code_fence", start, index - 1, lines[start:index]))
            continue

        if _is_picture_text_start(line):
            index += 1
            while index < len(lines):
                if _is_picture_text_end(lines[index]):
                    index += 1
                    break
                index += 1
            blocks.append(MarkdownBlock("picture_text", start, index - 1, lines[start:index]))
            continue

        if _is_table_start(lines, index):
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                index += 1
            blocks.append(MarkdownBlock("table", start, index - 1, lines[start:index]))
            continue

        if HEADING_RE.match(line):
            blocks.append(MarkdownBlock("heading", start, start, [line]))
            index += 1
            continue

        if IMAGE_LINE_RE.fullmatch(line):
            blocks.append(MarkdownBlock("image", start, start, [line]))
            index += 1
            continue

        if LIST_ITEM_RE.match(line):
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if not next_line.strip():
                    break
                if LIST_ITEM_RE.match(next_line) or next_line.startswith(" ") or next_line.startswith("\t"):
                    index += 1
                    continue
                break
            blocks.append(MarkdownBlock("list", start, index - 1, lines[start:index]))
            continue

        if BLOCKQUOTE_RE.match(line):
            index += 1
            while index < len(lines) and lines[index].strip() and BLOCKQUOTE_RE.match(lines[index]):
                index += 1
            blocks.append(MarkdownBlock("blockquote", start, index - 1, lines[start:index]))
            continue

        if _is_unknown_start(line):
            index += 1
            while index < len(lines) and lines[index].strip() and _is_unknown_start(lines[index]):
                index += 1
            blocks.append(MarkdownBlock("unknown", start, index - 1, lines[start:index]))
            continue

        index += 1
        while index < len(lines):
            next_line = lines[index]
            if not next_line.strip():
                break
            if (
                _fence_delimiter(next_line)
                or _is_picture_text_start(next_line)
                or _is_table_start(lines, index)
                or HEADING_RE.match(next_line)
                or IMAGE_LINE_RE.fullmatch(next_line)
                or LIST_ITEM_RE.match(next_line)
                or BLOCKQUOTE_RE.match(next_line)
                or _is_unknown_start(next_line)
            ):
                break
            index += 1

        blocks.append(MarkdownBlock("paragraph", start, index - 1, lines[start:index]))

    return blocks
