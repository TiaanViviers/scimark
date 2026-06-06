from scimark.passes.figures import keep_captions_with_figures


def test_keeps_caption_adjacent_after_image() -> None:
    markdown = """Paragraph.

![](_assets/paper/figure-1.png)

Figure 1: Example figure.
"""

    cleaned, adjusted = keep_captions_with_figures(markdown)

    assert adjusted == 1
    assert cleaned == "Paragraph.\n\n![](_assets/paper/figure-1.png)\nFigure 1: Example figure.\n"


def test_moves_caption_below_nearby_image_block() -> None:
    markdown = """Figure 2: Example with comment.

<!-- scimark: low-confidence-math -->
![](_assets/paper/figure-2a.png)

<!-- scimark: low-confidence-math -->
![](_assets/paper/figure-2b.png)
"""

    cleaned, adjusted = keep_captions_with_figures(markdown)

    assert adjusted == 1
    assert cleaned == "\n".join(
        [
            "<!-- scimark: low-confidence-math -->",
            "![](_assets/paper/figure-2a.png)",
            "<!-- scimark: low-confidence-math -->",
            "![](_assets/paper/figure-2b.png)",
            "Figure 2: Example with comment.",
            "",
        ]
    )


def test_ignores_regular_prose() -> None:
    markdown = """Figure 3 is discussed below.

This paragraph is not a caption.
"""

    cleaned, adjusted = keep_captions_with_figures(markdown)

    assert adjusted == 0
    assert cleaned == markdown
