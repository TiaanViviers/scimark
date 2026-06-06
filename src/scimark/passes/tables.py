from __future__ import annotations

import re
from dataclasses import dataclass

from scimark.document import TableStats


SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
LARGE_CELL_CHAR_THRESHOLD = 80
LARGE_CELL_SEGMENT_THRESHOLD = 4


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


def _cell_segments(cell: str) -> list[str]:
    segments = [segment.strip() for segment in BR_TAG_RE.split(cell) if segment.strip()]
    if segments:
        return segments
    if cell.strip():
        return [cell.strip()]
    return []


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
    unusually_large_cells = 0
    repeated_stacked_values = 0
    stacked_row_mismatches = 0
    max_stacked_segments = 0

    for row in content_rows:
        stacked_counts: list[int] = []

        for cell in row:
            segments = _cell_segments(cell)
            segment_count = len(segments)
            max_stacked_segments = max(max_stacked_segments, segment_count)

            if segment_count >= LARGE_CELL_SEGMENT_THRESHOLD or len(cell.strip()) >= LARGE_CELL_CHAR_THRESHOLD:
                unusually_large_cells += 1

            if segment_count >= 3:
                repeated_stacked_values += 1

            if cell.strip():
                if segment_count > 1:
                    stacked_counts.append(segment_count)

        if len(stacked_counts) >= 2 and len(set(stacked_counts)) > 1:
            stacked_row_mismatches += 1

    low_confidence_reasons: list[str] = []
    if not consistent_columns:
        low_confidence_reasons.append("inconsistent-columns")
    if columns <= 1:
        low_confidence_reasons.append("too-few-columns")
    if empty_cell_ratio >= 0.35:
        low_confidence_reasons.append("many-empty-cells")
    if stacked_row_mismatches > 0:
        low_confidence_reasons.append("stacked-row-mismatch")
    if repeated_stacked_values > 0 and empty_cells > 0 and br_tag_count >= 2:
        low_confidence_reasons.append("sparse-stacked-cells")
    if repeated_stacked_values >= max(2, columns) and br_tag_count >= max(4, columns):
        low_confidence_reasons.append("multi-value-cells")
    if unusually_large_cells >= max(1, columns // 2) and repeated_stacked_values > 0:
        low_confidence_reasons.append("large-cells")
    if br_tag_count >= max(8, rows := len(content_rows)) and empty_cells > 0:
        low_confidence_reasons.append("dense-br-layout")

    low_confidence = bool(low_confidence_reasons)

    return TableStats(
        rows=len(content_rows),
        columns=columns,
        consistent_columns=consistent_columns,
        empty_cells=empty_cells,
        empty_cell_ratio=round(empty_cell_ratio, 3),
        br_tag_count=br_tag_count,
        unusually_large_cells=unusually_large_cells,
        repeated_stacked_values=repeated_stacked_values,
        stacked_row_mismatches=stacked_row_mismatches,
        max_stacked_segments=max_stacked_segments,
        low_confidence=low_confidence,
        low_confidence_reasons=low_confidence_reasons,
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
