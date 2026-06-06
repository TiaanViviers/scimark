from scimark.passes.page_noise import strip_page_number_lines


def test_strip_page_number_lines_ignores_code_fences() -> None:
    markdown = """Intro

2

```text
3
```

| Page | Value |
| --- | --- |
| 4 | kept |
"""

    cleaned, removed = strip_page_number_lines(markdown)

    assert removed == 1
    assert "\n2\n" not in cleaned
    assert "3\n```" in cleaned
    assert "| 4 | kept |" in cleaned
