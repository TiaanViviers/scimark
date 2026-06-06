from __future__ import annotations

import re
from dataclasses import dataclass

from scimark.document import TableStats


SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


@dataclass(slots=True)
class TableBlock:
    start_line: int
    end_line: int
    lines: list[str]


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


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def find_tables(markdown: str) -> list[TableBlock]:
    lines = markdown.splitlines()
    tables: list[TableBlock] = []
    in_code_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            index += 1
            continue

        if in_code_fence or index + 1 >= len(lines):
            index += 1
            continue

        if _is_table_row(line) and _is_separator_line(lines[index + 1]):
            end = index + 2
            while end < len(lines) and _is_table_row(lines[end]):
                end += 1
            tables.append(TableBlock(start_line=index, end_line=end - 1, lines=lines[index:end]))
            index = end
            continue

        index += 1

    return tables


def analyze_table(lines: list[str]) -> TableStats:
    parsed_rows = [_split_table_row(line) for line in lines]
    content_rows = [row for row, line in zip(parsed_rows, lines, strict=False) if not _is_separator_line(line)]
    row_lengths = [len(row) for row in content_rows]
    columns = max(row_lengths, default=0)
    consistent_columns = len(set(row_lengths)) <= 1
    empty_cells = sum(1 for row in content_rows for cell in row if not cell)
    total_cells = sum(len(row) for row in content_rows)
    br_tag_count = sum(cell.lower().count("<br>") for row in content_rows for cell in row)
    empty_cell_ratio = empty_cells / total_cells if total_cells else 0.0
    low_confidence = (
        not consistent_columns
        or empty_cell_ratio >= 0.35
        or br_tag_count >= max(2, columns)
        or (br_tag_count >= 2 and empty_cells > 0)
        or columns <= 1
    )

    return TableStats(
        rows=len(content_rows),
        columns=columns,
        consistent_columns=consistent_columns,
        empty_cells=empty_cells,
        empty_cell_ratio=round(empty_cell_ratio, 3),
        br_tag_count=br_tag_count,
        low_confidence=low_confidence,
    )


def annotate_low_confidence_tables(markdown: str) -> tuple[str, list[TableStats]]:
    tables = find_tables(markdown)
    if not tables:
        return markdown, []

    lines = markdown.splitlines()
    output_lines: list[str] = []
    cursor = 0
    stats: list[TableStats] = []

    for table in tables:
        output_lines.extend(lines[cursor:table.start_line])
        table_stats = analyze_table(table.lines)
        stats.append(table_stats)
        if table_stats.low_confidence:
            output_lines.append("<!-- scimark: low-confidence-table -->")
        output_lines.extend(table.lines)
        cursor = table.end_line + 1

    output_lines.extend(lines[cursor:])
    return "\n".join(output_lines).rstrip() + "\n", stats
