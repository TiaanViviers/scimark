from scimark.passes.page_noise import strip_page_number_lines


def test_strip_page_number_lines_removes_standalone_digits() -> None:
    markdown = """Intro

2

Body

14
"""

    cleaned, removed = strip_page_number_lines(markdown)

    assert removed == 2
    assert "\n2\n" not in cleaned
    assert "\n14\n" not in cleaned
    assert "Intro" in cleaned
    assert "Body" in cleaned


def test_strip_page_number_lines_ignores_code_fences() -> None:
    markdown = """Intro

```text
3
```
"""

    cleaned, removed = strip_page_number_lines(markdown)

    assert removed == 0
    assert "3\n```" in cleaned


def test_strip_page_number_lines_ignores_markdown_tables() -> None:
    markdown = """| Page | Value |
| --- | --- |
| 4 | kept |
| 12 | also kept |
"""

    cleaned, removed = strip_page_number_lines(markdown)

    assert removed == 0
    assert "| 4 | kept |" in cleaned
    assert "| 12 | also kept |" in cleaned
