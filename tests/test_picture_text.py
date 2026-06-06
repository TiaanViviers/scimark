from scimark.passes.picture_text import remove_picture_text_blocks


def test_remove_picture_text_blocks() -> None:
    markdown = """Intro paragraph.

----- Start of picture text -----
32
16 Basic algorithm
----- End of picture text -----

Conclusion.
"""

    cleaned, count = remove_picture_text_blocks(markdown)

    assert count == 1
    assert "Start of picture text" not in cleaned
    assert cleaned == "Intro paragraph.\n\nConclusion.\n"


def test_remove_picture_text_blocks_with_markdown_formatting() -> None:
    markdown = """Before.

**----- Start of picture text -----**<br>
Multi-Head Attention<br>**----- End of picture text -----**<br>

After.
"""

    cleaned, count = remove_picture_text_blocks(markdown)

    assert count == 1
    assert "Multi-Head Attention" not in cleaned
    assert cleaned == "Before.\n\nAfter.\n"
