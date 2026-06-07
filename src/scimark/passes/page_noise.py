from __future__ import annotations

from scimark.markdown_blocks import parse_markdown_blocks


def strip_page_number_lines(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    protected_line_numbers = {
        line_number
        for block in parse_markdown_blocks(markdown)
        if block.block_type in {"table", "code_fence"}
        for line_number in range(block.start_line, block.end_line + 1)
    }
    cleaned_lines: list[str] = []
    removed = 0

    for line_number, line in enumerate(lines):
        stripped = line.strip()

        if (
            line_number not in protected_line_numbers
            and stripped.isdigit()
        ):
            removed += 1
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result + ("\n" if result else ""), removed
