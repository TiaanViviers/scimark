from scimark.markdown_blocks import parse_markdown_blocks


def test_parse_markdown_blocks_classifies_core_block_types() -> None:
    markdown = """# Heading

Paragraph one
still paragraph.

![](_assets/paper/figure-1.png)

| A | B |
| --- | --- |
| 1 | 2 |

```text
code
```

- item one
- item two

> quoted
> text

<!-- scimark: low-confidence-math -->

----- Start of picture text -----
noise
----- End of picture text -----
"""

    blocks = parse_markdown_blocks(markdown)

    assert [block.block_type for block in blocks] == [
        "heading",
        "paragraph",
        "image",
        "table",
        "code_fence",
        "list",
        "blockquote",
        "unknown",
        "picture_text",
    ]
    assert blocks[1].start_line == 2
    assert blocks[1].end_line == 3
    assert blocks[3].lines[0] == "| A | B |"


def test_parse_markdown_blocks_preserves_line_numbers_for_tables_and_code() -> None:
    markdown = """Intro

| A | B |
| --- | --- |
| 1 | 2 |

```python
print("x")
```
"""

    blocks = parse_markdown_blocks(markdown)

    table_block = next(block for block in blocks if block.block_type == "table")
    code_block = next(block for block in blocks if block.block_type == "code_fence")

    assert (table_block.start_line, table_block.end_line) == (2, 4)
    assert (code_block.start_line, code_block.end_line) == (6, 8)
