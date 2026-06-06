from __future__ import annotations

from scimark.passes.tables import find_tables


def _fence_delimiter(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("```"):
        return "```"
    if stripped.startswith("~~~"):
        return "~~~"
    return None


def strip_page_number_lines(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    table_line_numbers = {
        line_number
        for table in find_tables(markdown)
        for line_number in range(table.start_line, table.end_line + 1)
    }
    cleaned_lines: list[str] = []
    removed = 0
    in_code_fence = False

    for line_number, line in enumerate(lines):
        stripped = line.strip()

        if _fence_delimiter(line):
            in_code_fence = not in_code_fence
            cleaned_lines.append(line)
            continue

        if (
            not in_code_fence
            and line_number not in table_line_numbers
            and stripped.isdigit()
        ):
            removed += 1
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result + ("\n" if result else ""), removed
