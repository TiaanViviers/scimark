from scimark.passes.confidence import annotate_low_confidence_math
from scimark.passes.math_cleanup import cleanup_math_text


def test_cleanup_math_text() -> None:
    markdown = "The score is 0 _._ 9 and the split is 1 _/_ 4 . Also 5 − 3."

    cleaned = cleanup_math_text(markdown)

    assert "0.9" in cleaned
    assert "1/4" in cleaned
    assert "4." in cleaned
    assert "5 - 3" in cleaned


def test_annotate_low_confidence_math() -> None:
    markdown = """This paragraph is readable.

The update uses _[ x _D[+] 1 ] with [+] and [−] terms that look broken.
"""

    annotated, count = annotate_low_confidence_math(markdown)

    assert count == 1
    assert "<!-- scimark: low-confidence-math -->" in annotated
