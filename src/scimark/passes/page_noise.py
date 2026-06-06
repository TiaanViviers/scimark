from __future__ import annotations


def strip_page_number_lines(markdown: str) -> tuple[str, int]:
    lines = markdown.splitlines()
    cleaned_lines: list[str] = []
    removed = 0
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            cleaned_lines.append(line)
            continue

        if not in_code_fence and stripped.isdigit():
            removed += 1
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result + ("\n" if result else ""), removed
