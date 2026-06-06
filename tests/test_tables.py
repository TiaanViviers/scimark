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
    assert stats.unusually_large_cells == 0
    assert stats.repeated_stacked_values == 0
    assert stats.stacked_row_mismatches == 0
    assert stats.low_confidence_reasons == []


def test_multirow_header_table_can_still_be_high_confidence() -> None:
    markdown = """|Layer Type|Complexity per Layer|Sequential|Maximum Path Length|
|---|---|---|---|
|||Operations||
|Self-Attention|_O_(_n_2 _· d_)|_O_(1)|_O_(1)|
|Recurrent|_O_(_n · d_2)|_O_(_n_)|_O_(_n_)|
|Convolutional|_O_(_k · n · d_2)|_O_(1)|_O_(_logk_(_n_))|
|Self-Attention (restricted)|_O_(_r · n · d_)|_O_(1)|_O_(_n/r_)|
"""

    tables = find_tables(markdown)

    assert len(tables) == 1
    stats = analyze_table(tables[0].lines)
    assert stats.low_confidence is False
    assert stats.empty_cells >= 1
    assert stats.stacked_row_mismatches == 0


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
    assert "sparse-stacked-cells" in stats[0].low_confidence_reasons


def test_stacked_multi_value_table_is_low_confidence() -> None:
    markdown = """|Model|BLEU<br>EN-DE<br>EN-FR|Training Cost (FLOPs)|
|---|---|---|
|||EN-DE<br>EN-FR|
|ByteNet [15]<br>Deep-Att + PosUnk [32]<br>GNMT + RL [31]<br>ConvS2S [8]<br>MoE[26]|23.75<br>39.2<br>24.6<br>39.92<br>25.16<br>40.46<br>26.03<br>40.56|1_._0_·_1020<br>2_._3_·_1019<br>1_._4_·_1020<br>9_._6_·_1018<br>1_._5_·_1020<br>2_._0_·_1019<br>1_._2_·_1020|
|Transformer (base model)<br>Transformer (big)|27.3<br>38.1<br>**28.4**<br>**41.0**|**3****_._3****_·_ 1018**<br>2_._3_·_1019|
"""

    tables = find_tables(markdown)

    assert len(tables) == 1
    stats = analyze_table(tables[0].lines)
    assert stats.low_confidence is True
    assert stats.repeated_stacked_values >= 3
    assert stats.stacked_row_mismatches >= 1
    assert stats.max_stacked_segments >= 4
    assert "stacked-row-mismatch" in stats.low_confidence_reasons
