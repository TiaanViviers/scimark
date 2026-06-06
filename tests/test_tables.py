from scimark.passes.tables import analyze_table, annotate_low_confidence_tables, find_tables


def test_detects_markdown_tables() -> None:
    markdown = """| Model | Score |
| --- | --- |
| A | 0.91 |
| B | 0.89 |
"""

    tables = find_tables(markdown)

    assert len(tables) == 1
    stats = analyze_table(tables[0].lines)
    assert stats.rows == 3
    assert stats.columns == 2
    assert stats.consistent_columns is True
    assert stats.low_confidence is False


def test_annotates_low_confidence_table() -> None:
    markdown = """| Metric | Value |
| --- | --- |
| Accuracy |  |
| Recall | bad<br>split<br>cell |
"""

    annotated, stats = annotate_low_confidence_tables(markdown)

    assert len(stats) == 1
    assert stats[0].low_confidence is True
    assert annotated.startswith("<!-- scimark: low-confidence-table -->")
