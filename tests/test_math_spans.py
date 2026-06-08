from scimark.layout import RawLayoutBox, RawLayoutDocument, RawLayoutLine, RawLayoutPage, RawLayoutSpan
from scimark.math_spans import (
    FormulaBoxKind,
    LineKind,
    LineRun,
    MathRegion,
    PaperMathDiagnostics,
    RegionKind,
    RegionBlock,
    RegionSource,
    SegmentAnalysis,
    analyze_formula_box_region,
    build_paper_math_diagnostics,
    build_page_math_diagnostics,
    build_math_debug_report,
    build_page_regions,
    build_region_review_report,
    build_page_segments,
    classify_line,
    extract_line_features,
    find_embedded_display_math_runs,
    find_math_regions,
    is_math_like_box,
    normalize_span_token,
    serialize_display_math_line,
    serialize_page,
    serialize_page_with_region_promotion,
    serialize_region_block,
    serialize_region,
    serialize_line,
    should_use_experimental_math_on_page,
    should_use_region_serializer,
    split_line_by_baseline,
)


def _span(
    text: str,
    *,
    x0: float,
    x1: float,
    baseline: float = 10.0,
    size: float = 10.0,
    flags: int = 0,
) -> RawLayoutSpan:
    return RawLayoutSpan(
        text=text,
        bbox=(x0, baseline - size, x1, baseline),
        origin=(x0, baseline),
        size=size,
        flags=flags,
        char_flags=0,
        font="Test",
        block=0,
        line=0,
        direction=(1.0, 0.0),
    )


def test_serialize_line_recovers_simple_subscript_and_superscript() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 40.0, 20.0),
        spans=[
            _span("W", x0=0.0, x1=8.0, baseline=10.0, size=10.0),
            _span("i", x0=8.2, x1=11.0, baseline=12.4, size=7.0),
            _span("Q", x0=11.2, x1=17.0, baseline=7.6, size=7.0),
        ],
    )

    assert serialize_line(line) == "W_i^Q"


def test_normalize_span_token_recovers_bracketed_parser_tokens() -> None:
    assert normalize_span_token("[x][i]") == "xi"
    assert normalize_span_token("[+][(]") == "+("
    assert normalize_span_token("[) = [˜]") == ")=˜"


def test_serialize_line_demotes_invalid_script_cluster_to_plain_inline_math() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 60.0, 20.0),
        spans=[
            _span("r", x0=0.0, x1=4.0, baseline=10.0, size=10.0),
            _span("D", x0=4.2, x1=8.0, baseline=12.2, size=7.0),
            _span("[+][(]", x0=8.1, x1=16.0, baseline=7.7, size=7.0),
            _span("[y]", x0=16.1, x1=20.0, baseline=7.7, size=7.0),
            _span("[)]", x0=20.1, x1=24.0, baseline=7.7, size=7.0),
        ],
    )

    assert serialize_line(line) == "r_D+(y)"


def test_serialize_line_keeps_valid_script_cluster_grouped() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 60.0, 20.0),
        spans=[
            _span("x", x0=0.0, x1=4.0, baseline=10.0, size=10.0),
            _span("i", x0=4.2, x1=7.0, baseline=12.2, size=7.0),
            _span("+1", x0=7.2, x1=12.0, baseline=12.2, size=7.0),
        ],
    )

    assert serialize_line(line) == "x_{i+1}"


def test_extract_line_features_reports_math_heavy_line_signals() -> None:
    line = RawLayoutLine(
        bbox=(40.0, 0.0, 220.0, 20.0),
        spans=[
            _span("r", x0=40.0, x1=44.0),
            _span("D", x0=44.2, x1=48.0, baseline=12.2, size=7.0),
            _span("[+][(]", x0=48.2, x1=60.0, baseline=7.7, size=7.0),
            _span("[y]", x0=60.2, x1=66.0, baseline=7.7, size=7.0),
            _span("[)]", x0=66.2, x1=72.0, baseline=7.7, size=7.0),
            _span("=[0]", x0=74.0, x1=88.0),
        ],
    )

    features = extract_line_features(
        line,
        serialized_text=serialize_line(line),
        page_width=300.0,
        box_bbox=(20.0, 0.0, 260.0, 24.0),
    )

    assert features.operator_count >= 3
    assert features.script_fraction > 0.3
    assert features.symbol_density > 0.15


def test_classify_line_marks_dense_math_as_display_math() -> None:
    line = RawLayoutLine(
        bbox=(70.0, 0.0, 190.0, 20.0),
        spans=[
            _span("r", x0=70.0, x1=74.0),
            _span("D", x0=74.2, x1=78.0, baseline=12.2, size=7.0),
            _span("[+][(]", x0=78.2, x1=90.0, baseline=7.7, size=7.0),
            _span("[y]", x0=90.2, x1=96.0, baseline=7.7, size=7.0),
            _span("[)]", x0=96.2, x1=102.0, baseline=7.7, size=7.0),
            _span("-", x0=106.0, x1=110.0),
            _span("ω", x0=114.0, x1=120.0),
            _span("D", x0=120.2, x1=124.0, baseline=12.2, size=7.0),
            _span("=[0]", x0=128.0, x1=142.0),
        ],
    )

    classification = classify_line(
        line,
        serialized_text=serialize_line(line),
        page_width=300.0,
        box_bbox=(20.0, 0.0, 260.0, 24.0),
    )

    assert classification.kind == LineKind.DISPLAY_MATH
    assert classification.confidence >= 0.75
    assert "many_operators" in classification.reasons


def test_classify_line_marks_proof_leadin_as_theorem_proof() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 180.0, 20.0),
        spans=[_span("Proof. The key is again consider y", x0=0.0, x1=160.0)],
    )

    classification = classify_line(line, serialized_text=serialize_line(line))

    assert classification.kind == LineKind.THEOREM_PROOF
    assert "starts_with_proof" in classification.reasons


def test_classify_line_does_not_treat_hyphenated_prose_as_math_or_algorithm() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 220.0, 20.0),
        spans=[_span("algorithm 14 and real-world applications remain prose.", x0=0.0, x1=200.0)],
    )

    classification = classify_line(line, serialized_text=serialize_line(line))

    assert classification.kind == LineKind.PROSE


def test_classify_line_treats_author_affiliation_line_as_prose() -> None:
    line = RawLayoutLine(
        bbox=(80.0, 0.0, 240.0, 20.0),
        spans=[
            _span("Diederik", x0=80.0, x1=120.0),
            _span("P.", x0=124.0, x1=136.0),
            _span("Kingma", x0=140.0, x1=178.0),
            _span("*", x0=182.0, x1=186.0, baseline=7.5, size=7.0, flags=1),
        ],
    )

    classification = classify_line(
        line,
        serialized_text=serialize_line(line),
        page_width=320.0,
        box_bbox=(40.0, 0.0, 280.0, 24.0),
    )

    assert classification.kind == LineKind.PROSE
    assert "author_affiliation_like" in classification.reasons


def test_classify_line_treats_diagram_label_with_scripts_as_non_math() -> None:
    line = RawLayoutLine(
        bbox=(120.0, 0.0, 210.0, 20.0),
        spans=[
            _span("Discretize", x0=120.0, x1=170.0),
            _span("∆", x0=176.0, x1=184.0),
            _span("SRAMGPU", x0=184.5, x1=210.0, baseline=12.2, size=7.0),
        ],
    )

    classification = classify_line(
        line,
        serialized_text=serialize_line(line),
        page_width=400.0,
        box_bbox=(80.0, 0.0, 240.0, 24.0),
    )

    assert classification.kind in {LineKind.PROSE, LineKind.UNKNOWN}
    assert "diagram_label_like" in classification.reasons


def test_classify_line_treats_bibliography_entry_as_prose() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 360.0, 20.0),
        spans=[
            _span(
                "25 Chris Donahue, Julian Mc Auley, and Miller Puckette. In: The International Conference on Learning Representations (ICLR). 2019.",
                x0=0.0,
                x1=340.0,
            )
        ],
    )

    classification = classify_line(line, serialized_text=serialize_line(line))

    assert classification.kind == LineKind.PROSE
    assert "bibliography_entry_like" in classification.reasons


def test_serialize_display_math_line_adds_math_spacing() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 180.0, 20.0),
        spans=[
            _span("r", x0=0.0, x1=4.0),
            _span("D", x0=4.2, x1=8.0, baseline=12.2, size=7.0),
            _span("[+][(]", x0=8.2, x1=20.0, baseline=7.7, size=7.0),
            _span("[y]", x0=20.2, x1=26.0, baseline=7.7, size=7.0),
            _span("[)]", x0=26.2, x1=32.0, baseline=7.7, size=7.0),
            _span("-ω", x0=36.0, x1=44.0),
            _span("D", x0=44.2, x1=48.0, baseline=12.2, size=7.0),
            _span("=[0]", x0=52.0, x1=66.0),
        ],
    )

    assert serialize_display_math_line(line) == "r_D+(y) - ω_D = 0"


def test_split_line_by_baseline_separates_interleaved_visual_lines() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 120.0, 30.0),
        spans=[
            _span("Property", x0=0.0, x1=30.0, baseline=10.0, size=10.0),
            _span("introduced", x0=0.0, x1=36.0, baseline=20.0, size=10.0),
            _span("of", x0=32.0, x1=40.0, baseline=10.0, size=10.0),
            _span("the", x0=38.0, x1=50.0, baseline=20.0, size=10.0),
            _span("Extended", x0=42.0, x1=72.0, baseline=10.0, size=10.0),
            _span("function", x0=52.0, x1=80.0, baseline=20.0, size=10.0),
        ],
    )

    virtual_lines = split_line_by_baseline(line)

    assert len(virtual_lines) == 2
    assert serialize_line(virtual_lines[0]) == "Property of Extended"
    assert serialize_line(virtual_lines[1]) == "introduced the function"


def test_is_math_like_box_detects_formula_and_scripted_text() -> None:
    formula_box = RawLayoutBox(source_index=0, boxclass="formula", bbox=(0.0, 0.0, 40.0, 20.0))
    text_box = RawLayoutBox(
        source_index=1,
        boxclass="text",
        bbox=(0.0, 0.0, 60.0, 20.0),
        textlines=[
            RawLayoutLine(
                bbox=(0.0, 0.0, 60.0, 20.0),
                spans=[
                    _span("d", x0=0.0, x1=6.0),
                    _span("k", x0=6.2, x1=10.0, baseline=12.0, size=7.0),
                ],
            )
        ],
    )

    assert is_math_like_box(formula_box) is True
    assert is_math_like_box(text_box) is True


def test_find_math_regions_groups_adjacent_math_text_and_formula_boxes() -> None:
    page = RawLayoutPage(
        page_number=12,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 80.0, 20.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 80.0, 20.0),
                        spans=[_span("Lemma", x0=0.0, x1=20.0), _span("A.2.", x0=22.0, x1=40.0)],
                    )
                ],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 80.0, 42.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 44.0, 120.0, 64.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 44.0, 120.0, 64.0),
                        spans=[
                            _span("x", x0=0.0, x1=4.0),
                            _span("i", x0=4.2, x1=7.0, baseline=12.2, size=7.0),
                        ],
                    )
                ],
            ),
        ],
    )

    regions = find_math_regions(page)

    assert len(regions) == 1
    assert regions[0].start_box_index == 0
    assert regions[0].end_box_index == 2
    assert [box.boxclass for box in regions[0].boxes] == ["text", "formula", "text"]


def test_serialize_region_keeps_formula_box_as_explicit_placeholder() -> None:
    region = find_math_regions(
        RawLayoutPage(
            page_number=1,
            width=400.0,
            height=600.0,
            boxes=[
                RawLayoutBox(
                    source_index=0,
                    boxclass="text",
                    bbox=(0.0, 0.0, 60.0, 20.0),
                    textlines=[
                        RawLayoutLine(
                            bbox=(0.0, 0.0, 60.0, 20.0),
                            spans=[_span("Proof.", x0=0.0, x1=24.0)],
                        )
                    ],
                ),
                RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 60.0, 42.0)),
                RawLayoutBox(
                    source_index=2,
                    boxclass="text",
                    bbox=(0.0, 44.0, 60.0, 64.0),
                    textlines=[
                        RawLayoutLine(
                            bbox=(0.0, 44.0, 60.0, 64.0),
                            spans=[_span("x", x0=0.0, x1=4.0), _span("i", x0=4.2, x1=7.0, baseline=12.2, size=7.0)],
                        )
                    ],
                ),
            ],
        )
    )[0]

    serialized = serialize_region(region)

    assert "Proof." in serialized
    assert "[[FORMULA_BOX]]" in serialized
    assert "x_i" in serialized


def test_serialize_region_stitches_hyphenated_prose_across_boxes() -> None:
    region = MathRegion(
        page_number=1,
        start_box_index=0,
        end_box_index=1,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 100.0, 20.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 100.0, 20.0),
                        spans=[_span("We have in-", x0=0.0, x1=50.0)],
                    )
                ],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(0.0, 22.0, 120.0, 42.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 22.0, 120.0, 42.0),
                        spans=[_span("troduced the extension.", x0=0.0, x1=90.0)],
                    )
                ],
            ),
        ],
    )

    assert serialize_region(region) == "We have introduced the extension."


def test_serialize_region_separates_prose_from_dense_math_box() -> None:
    region = find_math_regions(
        RawLayoutPage(
            page_number=1,
            width=400.0,
            height=600.0,
            boxes=[
                RawLayoutBox(
                    source_index=0,
                    boxclass="text",
                    bbox=(0.0, 0.0, 120.0, 20.0),
                    textlines=[
                        RawLayoutLine(
                            bbox=(0.0, 0.0, 120.0, 20.0),
                            spans=[_span("Proof. The key is", x0=0.0, x1=80.0)],
                        )
                    ],
                ),
                RawLayoutBox(
                    source_index=1,
                    boxclass="text",
                    bbox=(0.0, 22.0, 160.0, 42.0),
                    textlines=[
                        RawLayoutLine(
                            bbox=(0.0, 22.0, 160.0, 42.0),
                            spans=[
                                _span("r", x0=0.0, x1=4.0),
                                _span("D", x0=4.2, x1=8.0, baseline=12.2, size=7.0),
                                _span("[+][(]", x0=8.1, x1=16.0, baseline=7.7, size=7.0),
                                _span("[y]", x0=16.1, x1=20.0, baseline=7.7, size=7.0),
                                _span("[)]", x0=20.1, x1=24.0, baseline=7.7, size=7.0),
                                _span("=[0]", x0=24.2, x1=35.0),
                            ],
                        )
                    ],
                ),
            ],
        )
    )[0]

    assert serialize_region(region) == "Proof. The key is\nr_D+(y) = 0"


def test_build_page_segments_and_serialize_page_use_reordered_column_flow() -> None:
    page = RawLayoutPage(
        page_number=11,
        width=600.0,
        height=800.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="section-header",
                bbox=(40.0, 10.0, 560.0, 40.0),
                textlines=[RawLayoutLine(bbox=(40.0, 10.0, 560.0, 40.0), spans=[_span("APPENDIX", x0=40.0, x1=100.0)])],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(330.0, 60.0, 560.0, 100.0),
                textlines=[RawLayoutLine(bbox=(330.0, 60.0, 560.0, 100.0), spans=[_span("Right proof content.", x0=330.0, x1=420.0)])],
            ),
            RawLayoutBox(
                source_index=2,
                boxclass="section-header",
                bbox=(40.0, 50.0, 260.0, 80.0),
                textlines=[RawLayoutLine(bbox=(40.0, 50.0, 260.0, 80.0), spans=[_span("A.1 Weighted Quantile", x0=40.0, x1=160.0)])],
            ),
            RawLayoutBox(
                source_index=3,
                boxclass="text",
                bbox=(40.0, 90.0, 260.0, 150.0),
                textlines=[RawLayoutLine(bbox=(40.0, 90.0, 260.0, 150.0), spans=[_span("Left column intro.", x0=40.0, x1=120.0)])],
            ),
        ],
    )

    segments = build_page_segments(page)
    serialized = serialize_page(page)

    assert [segment.kind for segment in segments] == ["prose", "prose", "prose", "prose"]
    assert serialized == "APPENDIX\n\nA.1 Weighted Quantile\n\nLeft column intro.\n\nRight proof content."


def test_build_page_regions_groups_proof_display_and_followup_prose() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 180.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 180.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 24.0, 180.0, 40.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 44.0, 220.0, 64.0),
                textlines=[RawLayoutLine(bbox=(0.0, 44.0, 220.0, 64.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=100.0)])],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert regions[0].kind == RegionKind.THEOREM_PROOF_BLOCK
    assert any(region.kind == RegionKind.DISPLAY_MATH_BLOCK for region in regions)
    rendered = "\n\n".join(serialize_region_block(region, page_width=page.width) for region in regions)
    assert "Proof. The key is:" in rendered
    assert "[[DISPLAY_FORMULA:" in rendered
    assert "Therefore x_i=0." in rendered


def test_build_page_regions_splits_embedded_display_math_run_from_mixed_region() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 240.0, 80.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 220.0, 20.0),
                        spans=[_span("Proof. The key is to consider y.", x0=0.0, x1=160.0)],
                    ),
                    RawLayoutLine(
                        bbox=(60.0, 24.0, 220.0, 44.0),
                        spans=[
                            _span("r", x0=60.0, x1=64.0),
                            _span("D", x0=64.2, x1=68.0, baseline=36.2, size=7.0),
                            _span("[+][(]", x0=68.2, x1=80.0, baseline=31.7, size=7.0),
                            _span("[y]", x0=80.2, x1=86.0, baseline=31.7, size=7.0),
                            _span("[)]", x0=86.2, x1=92.0, baseline=31.7, size=7.0),
                            _span("-ω", x0=96.0, x1=104.0),
                            _span("D", x0=104.2, x1=108.0, baseline=36.2, size=7.0),
                            _span("=[0]", x0=112.0, x1=126.0),
                        ],
                    ),
                    RawLayoutLine(
                        bbox=(0.0, 48.0, 220.0, 68.0),
                        spans=[_span("Therefore x_i = 0.", x0=0.0, x1=90.0)],
                    ),
                ],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert [region.kind for region in regions] == [
        RegionKind.THEOREM_PROOF_BLOCK,
        RegionKind.DISPLAY_MATH_BLOCK,
        RegionKind.THEOREM_PROOF_BLOCK,
    ]
    assert should_use_region_serializer(regions[1], page_width=page.width) is True
    assert should_use_region_serializer(regions[0]) is False
    assert should_use_region_serializer(regions[2]) is False


def test_build_page_regions_splits_embedded_display_math_run_from_formula_mixed_region() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(40.0, 24.0, 280.0, 56.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 42.0, 260.0, 86.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(60.0, 42.0, 220.0, 62.0),
                        spans=[
                            _span("r", x0=60.0, x1=64.0),
                            _span("D", x0=64.2, x1=68.0, baseline=54.2, size=7.0),
                            _span("[+][(]", x0=68.2, x1=80.0, baseline=49.7, size=7.0),
                            _span("[y]", x0=80.2, x1=86.0, baseline=49.7, size=7.0),
                            _span("[)]", x0=86.2, x1=92.0, baseline=49.7, size=7.0),
                            _span("-ω", x0=96.0, x1=104.0),
                            _span("D", x0=104.2, x1=108.0, baseline=54.2, size=7.0),
                            _span("=[0]", x0=112.0, x1=126.0),
                        ],
                    ),
                    RawLayoutLine(
                        bbox=(0.0, 66.0, 220.0, 86.0),
                        spans=[_span("This means x_i = 0.", x0=0.0, x1=110.0)],
                    ),
                ],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert regions[0].kind == RegionKind.THEOREM_PROOF_BLOCK
    assert any(region.source == RegionSource.FORMULA_BOX_SPLIT for region in regions)


def test_build_page_regions_does_not_split_embedded_display_run_inside_caption_mixed_region() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 260.0, 90.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 240.0, 20.0),
                        spans=[_span("Figure 1: Overview.", x0=0.0, x1=100.0)],
                    ),
                    RawLayoutLine(
                        bbox=(80.0, 24.0, 220.0, 44.0),
                        spans=[
                            _span("W", x0=80.0, x1=86.0),
                            _span("∈", x0=90.0, x1=98.0),
                            _span("R", x0=102.0, x1=108.0),
                            _span("d×d", x0=110.0, x1=140.0),
                        ],
                    ),
                    RawLayoutLine(
                        bbox=(0.0, 48.0, 240.0, 68.0),
                        spans=[_span("We only train A and B.", x0=0.0, x1=120.0)],
                    ),
                ],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert len(regions) == 1
    assert all(region.source != RegionSource.EMBEDDED_DISPLAY_RUN for region in regions)


def test_build_page_regions_can_split_display_run_after_caption_box_in_mixed_region() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 260.0, 20.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 240.0, 20.0),
                        spans=[_span("Figure 1: Overview.", x0=0.0, x1=100.0)],
                    ),
                ],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(0.0, 24.0, 260.0, 90.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 24.0, 220.0, 44.0),
                        spans=[_span("Proof. The key is to consider y.", x0=0.0, x1=160.0)],
                    ),
                    RawLayoutLine(
                        bbox=(60.0, 48.0, 220.0, 68.0),
                        spans=[
                            _span("r", x0=60.0, x1=64.0),
                            _span("D", x0=64.2, x1=68.0, baseline=60.2, size=7.0),
                            _span("[+][(]", x0=68.2, x1=80.0, baseline=55.7, size=7.0),
                            _span("[y]", x0=80.2, x1=86.0, baseline=55.7, size=7.0),
                            _span("[)]", x0=86.2, x1=92.0, baseline=55.7, size=7.0),
                            _span("-ω", x0=96.0, x1=104.0),
                            _span("D", x0=104.2, x1=108.0, baseline=60.2, size=7.0),
                            _span("=[0]", x0=112.0, x1=126.0),
                        ],
                    ),
                    RawLayoutLine(
                        bbox=(0.0, 72.0, 220.0, 90.0),
                        spans=[_span("Therefore x_i = 0.", x0=0.0, x1=90.0)],
                    ),
                ],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert any(region.source == RegionSource.EMBEDDED_DISPLAY_RUN for region in regions)
    assert any(region.kind == RegionKind.CAPTION for region in regions)


def test_build_page_regions_merges_back_low_value_formula_box_split() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(80.0, 24.0, 140.0, 38.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 42.0, 220.0, 62.0),
                textlines=[RawLayoutLine(bbox=(0.0, 42.0, 220.0, 62.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=100.0)])],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert all(region.source != RegionSource.FORMULA_BOX_SPLIT for region in regions)
    assert len(regions) == 1
    assert regions[0].kind == RegionKind.THEOREM_PROOF_BLOCK


def test_build_page_regions_merges_low_value_formula_box_past_adjacent_display_block() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("A.1", x0=0.0, x1=30.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(40.0, 24.0, 280.0, 56.0)),
            RawLayoutBox(source_index=2, boxclass="formula", bbox=(80.0, 60.0, 140.0, 74.0)),
            RawLayoutBox(
                source_index=3,
                boxclass="text",
                bbox=(0.0, 78.0, 260.0, 118.0),
                textlines=[
                    RawLayoutLine(bbox=(0.0, 78.0, 260.0, 98.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)]),
                    RawLayoutLine(bbox=(0.0, 100.0, 260.0, 118.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=100.0)]),
                ],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert any(
        region.source == RegionSource.FORMULA_BOX_SPLIT and region.kind == RegionKind.DISPLAY_MATH_BLOCK
        for region in regions
    )
    assert all(
        not (
            region.source == RegionSource.FORMULA_BOX_SPLIT
            and region.kind == RegionKind.DISPLAY_MATH_BLOCK
            and any(segment.segment.boxes[0].bbox == (80.0, 60.0, 140.0, 74.0) for segment in region.segments if segment.segment.boxes)
        )
        for region in regions
    )


def test_build_page_regions_keeps_display_sized_formula_box_split() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(40.0, 24.0, 280.0, 56.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 60.0, 220.0, 80.0),
                textlines=[RawLayoutLine(bbox=(0.0, 60.0, 220.0, 80.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=100.0)])],
            ),
        ],
    )

    regions = build_page_regions(page)

    assert any(region.source == RegionSource.FORMULA_BOX_SPLIT for region in regions)
    assert any(region.kind == RegionKind.DISPLAY_MATH_BLOCK for region in regions)


def test_find_embedded_display_math_runs_reports_rejected_sentence_fragment() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 240.0, 40.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(60.0, 0.0, 220.0, 20.0),
                        spans=[_span("ϵ to ϵ+^1_b.", x0=60.0, x1=140.0)],
                    )
                ],
            ),
        ],
    )

    segment = build_page_regions(page)[0].segments[0]
    runs = find_embedded_display_math_runs(segment)

    assert len(runs) == 1
    assert runs[0].accepted is False
    assert "sentence_fragment" in runs[0].rejected_reasons


def test_should_use_region_serializer_only_promotes_high_confidence_display_math() -> None:
    display_page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(source_index=0, boxclass="formula", bbox=(0.0, 0.0, 180.0, 18.0)),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 20.0, 180.0, 38.0)),
        ],
    )
    proof_page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 180.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 180.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 24.0, 180.0, 40.0)),
        ],
    )

    display_region = build_page_regions(display_page)[0]
    proof_region = build_page_regions(proof_page)[0]

    assert display_region.kind == RegionKind.DISPLAY_MATH_BLOCK
    assert should_use_region_serializer(display_region, page_width=display_page.width) is False
    assert proof_region.kind == RegionKind.THEOREM_PROOF_BLOCK
    assert should_use_region_serializer(proof_region) is False


def test_analyze_formula_box_region_classifies_case_definition_as_useful() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 120.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 120.0, 20.0), spans=[_span("When y < x1:", x0=0.0, x1=60.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 140.0, 36.0)),
        ],
    )

    region = build_page_regions(page)[1]
    analysis = analyze_formula_box_region(
        region,
        page_width=page.width,
        previous_context="When y < x1:",
        next_context="",
    )

    assert analysis.kind == FormulaBoxKind.CASE_DEFINITION
    assert analysis.should_standalone is True
    assert analysis.should_promote is True


def test_analyze_formula_box_region_classifies_tiny_equation_as_inline_noise() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 120.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 120.0, 20.0), spans=[_span("Proof. The key is:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(80.0, 24.0, 140.0, 38.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 42.0, 220.0, 62.0),
                textlines=[RawLayoutLine(bbox=(0.0, 42.0, 220.0, 62.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=100.0)])],
            ),
        ],
    )

    split_region = build_page_regions(page)[0]
    formula_segment = next(segment for segment in split_region.segments if segment.has_formula_boxes)
    analysis = analyze_formula_box_region(
        RegionBlock(
            page_number=1,
            kind=RegionKind.DISPLAY_MATH_BLOCK,
            segments=[
                SegmentAnalysis(
                    segment=formula_segment.segment,
                    lines=[],
                    base_kind=RegionKind.DISPLAY_MATH_BLOCK,
                    confidence=0.96,
                    reasons=["formula_boxes"],
                    has_formula_boxes=True,
                )
            ],
            source=RegionSource.FORMULA_BOX_SPLIT,
        ),
        page_width=page.width,
        previous_context="Proof. The key is:",
        next_context="Therefore x_i=0.",
    )

    assert analysis.kind == FormulaBoxKind.INLINE_NOISE
    assert analysis.should_standalone is False
    assert analysis.should_promote is False


def test_serialize_page_with_region_promotion_promotes_safe_display_math_region() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(source_index=0, boxclass="formula", bbox=(80.0, 0.0, 320.0, 18.0)),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(80.0, 20.0, 320.0, 38.0)),
        ],
    )

    stable = serialize_page(page)
    promoted = serialize_page_with_region_promotion(page)

    assert stable == "[[DISPLAY_FORMULA_BLOCK x2: derivation]]"
    assert promoted == "[[DISPLAY_FORMULA_BLOCK x2: derivation]]"


def test_should_use_region_serializer_rejects_sentence_fragment_display_math() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(80.0, 0.0, 260.0, 20.0),
                textlines=[RawLayoutLine(bbox=(80.0, 0.0, 260.0, 20.0), spans=[_span("ϵ to ϵ+^1_b.", x0=80.0, x1=180.0)])],
            ),
        ],
    )

    region = build_page_regions(page)[0]

    assert region.kind == RegionKind.DISPLAY_MATH_BLOCK
    assert should_use_region_serializer(region) is False


def test_serialize_page_with_region_promotion_keeps_theorem_proof_region_on_stable_path() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 120.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 120.0, 20.0), spans=[_span("When y < x1:", x0=0.0, x1=60.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 140.0, 36.0)),
        ],
    )

    stable = serialize_page(page)
    promoted = serialize_page_with_region_promotion(page)

    assert stable == promoted


def test_should_use_region_serializer_rejects_display_math_next_to_caption() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("Figure 1: Overview.", x0=0.0, x1=100.0)])],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(80.0, 26.0, 220.0, 46.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(80.0, 26.0, 220.0, 46.0),
                        spans=[
                            _span("W", x0=80.0, x1=86.0),
                            _span("∈", x0=90.0, x1=98.0),
                            _span("R", x0=102.0, x1=108.0),
                            _span("d×d", x0=110.0, x1=140.0),
                        ],
                    )
                ],
            ),
        ],
    )

    regions = build_page_regions(page)
    assert len(regions) >= 2
    assert regions[0].kind == RegionKind.CAPTION
    assert regions[1].kind == RegionKind.DISPLAY_MATH_BLOCK
    assert should_use_region_serializer(
        regions[1],
        page_width=page.width,
        previous_context="Figure 1: Overview.",
        previous_region_kind=regions[0].kind,
        previous_region_source=regions[0].source,
    ) is False


def test_serialize_page_keeps_formula_region_between_prose_sections() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 120.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 120.0, 20.0), spans=[_span("Proof. The key is", x0=0.0, x1=80.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 120.0, 50.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 52.0, 160.0, 72.0),
                textlines=[RawLayoutLine(bbox=(0.0, 52.0, 160.0, 72.0), spans=[_span("Therefore x_i = 0.", x0=0.0, x1=90.0)])],
            ),
        ],
    )

    assert serialize_page(page) == "Proof. The key is\n[[DISPLAY_FORMULA: derivation]]\nTherefore x_i=0."


def test_serialize_page_uses_case_definition_formula_stub() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 120.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 120.0, 20.0), spans=[_span("When y < x1:", x0=0.0, x1=60.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 140.0, 36.0)),
        ],
    )

    assert serialize_page(page) == "When y < x1:\n\n[[DISPLAY_FORMULA: case-definition]]"


def test_serialize_page_groups_consecutive_formula_boxes_into_block_stub() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 160.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 160.0, 20.0), spans=[_span("The constraints are:", x0=0.0, x1=90.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 160.0, 34.0)),
            RawLayoutBox(source_index=2, boxclass="formula", bbox=(0.0, 36.0, 160.0, 48.0)),
        ],
    )

    assert serialize_page(page) == "The constraints are:\n\n[[DISPLAY_FORMULA_BLOCK x2: constraint]]"


def test_serialize_page_suppresses_redundant_equation_number_after_numbered_formula_stub() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 180.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 180.0, 20.0), spans=[_span("We define", x0=0.0, x1=40.0)])],
            ),
            RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 180.0, 38.0)),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 40.0, 40.0, 52.0),
                textlines=[RawLayoutLine(bbox=(0.0, 40.0, 40.0, 52.0), spans=[_span("(24)", x0=0.0, x1=18.0)])],
            ),
        ],
    )

    assert serialize_page(page) == "We define\n\n[[DISPLAY_FORMULA: numbered-equation (24)]]"


def test_serialize_page_normalizes_repeatable_surface_boundary_artifacts() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 250.0, 20.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 250.0, 20.0),
                        spans=[
                            _span(
                                "Our task is given a series of inputD, to estimater+(y) andr-(y) fory ∈X.",
                                x0=0.0,
                                x1=220.0,
                            )
                        ],
                    )
                ],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(0.0, 24.0, 300.0, 44.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 24.0, 300.0, 44.0),
                        spans=[_span("Given a quantile summary Q(D),wecallitis valid only when definedin S.", x0=0.0, x1=250.0)],
                    )
                ],
            ),
            RawLayoutBox(
                source_index=2,
                boxclass="text",
                bbox=(0.0, 48.0, 340.0, 68.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 48.0, 340.0, 68.0),
                        spans=[_span("Given a small multi-setD with S ={x1} and D ={(x1, w1)} and ˜ωDare defined.", x0=0.0, x1=320.0)],
                    )
                ],
            ),
            RawLayoutBox(
                source_index=3,
                boxclass="text",
                bbox=(0.0, 72.0, 360.0, 92.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 72.0, 360.0, 92.0),
                        spans=[_span("The function The ˜ωDto X is copied.", x0=0.0, x1=220.0)],
                    )
                ],
            ),
        ],
    )

    assert serialize_page(page) == (
        "Our task is given a series of input D, to estimate r+(y) and r-(y) for y ∈ X.\n\n"
        "Given a quantile summary Q(D), we call it valid only when defined in S.\n\n"
        "Given a small multi-set D with S = {x1} and D = {(x1, w1)} and ˜ωD are defined.\n"
        "The function\nThe ˜ωD to X is copied."
    )


def test_serialize_page_normalizes_dense_math_prose_boundary_artifacts() -> None:
    page = RawLayoutPage(
        page_number=1,
        width=500.0,
        height=700.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 320.0, 20.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(0.0, 0.0, 320.0, 20.0),
                        spans=[_span("Lemma A.2. Quantile summary Q(D)=(S, r) isan valid form.", x0=0.0, x1=280.0)],
                    )
                ],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(0.0, 24.0, 360.0, 44.0),
                textlines=[
                        RawLayoutLine(
                            bbox=(0.0, 24.0, 360.0, 44.0),
                            spans=[_span("Notethattherank andx∗=x1, thenreturnx1; Eq.(21) forQ′ and Q′is stable.", x0=0.0, x1=330.0)],
                        )
                    ],
                ),
        ],
    )

    assert serialize_page(page) == (
        "Lemma A.2. Quantile summary Q(D) = (S, r) is an valid form.\n"
        "Note that the rank and x∗ = x1, then return x1; Eq. (21) for Q′ and Q′ is stable."
    )


def test_serialize_line_normalizes_dense_math_prose_tokens() -> None:
    line = RawLayoutLine(
        bbox=(0.0, 0.0, 360.0, 20.0),
        spans=[_span("Input: Q(D)=(S, r) where S =x1 and ω˜Darefunctionsin S, thenreturnx1. Notethattherank holds.", x0=0.0, x1=340.0)],
    )

    assert serialize_line(line) == (
        "Input: Q(D)=(S, r) where S = x1 and ω˜D are functions in S, then return x1. "
        "Note that the rank holds."
    )


def test_build_math_debug_report_includes_prototype_and_span_roles() -> None:
    document = RawLayoutDocument(
        source_pdf="/tmp/sample.pdf",
        page_count=1,
        pages=[
            RawLayoutPage(
                page_number=1,
                width=400.0,
                height=600.0,
                boxes=[
                    RawLayoutBox(
                        source_index=0,
                        boxclass="text",
                        bbox=(0.0, 0.0, 60.0, 20.0),
                        textlines=[
                            RawLayoutLine(
                                bbox=(0.0, 0.0, 60.0, 20.0),
                                spans=[
                                    _span("W", x0=0.0, x1=8.0),
                                    _span("i", x0=8.2, x1=11.0, baseline=12.4, size=7.0),
                                    _span("Q", x0=11.2, x1=17.0, baseline=7.6, size=7.0),
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report = build_math_debug_report(document)

    assert "scimark math debug" in report
    assert "page 1" in report
    assert "page-prototype: W_i^Q" in report
    assert "prototype: W_i^Q" in report
    assert "region 0" in report


def test_build_region_review_report_includes_promotion_decisions() -> None:
    document = RawLayoutDocument(
        source_pdf="/tmp/sample.pdf",
        page_count=1,
        pages=[
            RawLayoutPage(
                page_number=1,
                width=400.0,
                height=600.0,
                boxes=[
                    RawLayoutBox(
                        source_index=0,
                        boxclass="text",
                        bbox=(0.0, 0.0, 180.0, 20.0),
                        textlines=[RawLayoutLine(bbox=(0.0, 0.0, 180.0, 20.0), spans=[_span("When y < x1:", x0=0.0, x1=60.0)])],
                    ),
                    RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 180.0, 40.0)),
                    RawLayoutBox(source_index=2, boxclass="formula", bbox=(0.0, 44.0, 180.0, 60.0)),
                ],
            )
        ],
    )

    report = build_region_review_report(document)

    assert "scimark region review" in report
    assert "REGION 0" in report
    assert "source=initial_grouping" in report
    assert "fallback=stable" in report
    assert "stable:" in report
    assert "lines: prose" in report
    assert "lines: formula_box" in report
    assert "theorem_proof_block" in report
    assert "specialized: [[DISPLAY_FORMULA_BLOCK x2: case-definition]]" in report


def test_build_paper_math_diagnostics_reports_region_and_suppression_counts() -> None:
    document = RawLayoutDocument(
        source_pdf="/tmp/sample.pdf",
        page_count=2,
        pages=[
            RawLayoutPage(
                page_number=1,
                width=400.0,
                height=600.0,
                boxes=[
                    RawLayoutBox(
                        source_index=0,
                        boxclass="text",
                        bbox=(40.0, 0.0, 260.0, 24.0),
                        textlines=[
                            RawLayoutLine(
                                bbox=(80.0, 0.0, 240.0, 20.0),
                                spans=[
                                    _span("Diederik", x0=80.0, x1=120.0),
                                    _span("P.", x0=124.0, x1=136.0),
                                    _span("Kingma", x0=140.0, x1=178.0),
                                    _span("*", x0=182.0, x1=186.0, baseline=7.5, size=7.0, flags=1),
                                ],
                            )
                        ],
                    ),
                    RawLayoutBox(
                        source_index=1,
                        boxclass="text",
                        bbox=(80.0, 30.0, 240.0, 54.0),
                        textlines=[
                            RawLayoutLine(
                                bbox=(120.0, 30.0, 210.0, 50.0),
                                spans=[
                                    _span("Discretize", x0=120.0, x1=170.0, baseline=40.0),
                                    _span("∆", x0=176.0, x1=184.0, baseline=40.0),
                                    _span("SRAMGPU", x0=184.5, x1=210.0, baseline=42.2, size=7.0),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            RawLayoutPage(
                page_number=2,
                width=400.0,
                height=600.0,
                boxes=[
                    RawLayoutBox(
                        source_index=0,
                        boxclass="text",
                        bbox=(0.0, 0.0, 180.0, 20.0),
                        textlines=[
                            RawLayoutLine(
                                bbox=(0.0, 0.0, 180.0, 20.0),
                                spans=[_span("When y < x1:", x0=0.0, x1=60.0)],
                            )
                        ],
                    ),
                    RawLayoutBox(source_index=1, boxclass="formula", bbox=(0.0, 22.0, 180.0, 40.0)),
                    RawLayoutBox(source_index=2, boxclass="formula", bbox=(0.0, 44.0, 180.0, 60.0)),
                ],
            ),
        ],
    )

    diagnostics = build_paper_math_diagnostics(document)

    assert diagnostics.pages_evaluated == 2
    assert diagnostics.display_math_regions == 1
    assert diagnostics.promoted_regions == 1
    assert diagnostics.fallback_regions == 0
    assert diagnostics.formula_box_split_regions == 1
    assert diagnostics.experimental_candidate_pages == 1
    assert diagnostics.reference_like_pages == 0
    assert diagnostics.figure_table_like_pages == 0
    assert diagnostics.author_affiliation_suppressions >= 1
    assert diagnostics.suspected_figure_label_suppressions >= 1


def test_build_page_math_diagnostics_marks_bibliography_like_page_as_non_candidate() -> None:
    page = RawLayoutPage(
        page_number=7,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 360.0, 120.0),
                textlines=[
                    RawLayoutLine(bbox=(0.0, 0.0, 360.0, 18.0), spans=[_span("[1] Smith et al. In: Proceedings of XYZ.", x0=0.0, x1=220.0)]),
                    RawLayoutLine(bbox=(0.0, 20.0, 360.0, 38.0), spans=[_span("[2] Doe et al. Journal of ABC, pp. 1-10.", x0=0.0, x1=240.0)]),
                    RawLayoutLine(bbox=(0.0, 40.0, 360.0, 58.0), spans=[_span("[3] Roe et al. DOI 10.1000/test.", x0=0.0, x1=210.0)]),
                    RawLayoutLine(bbox=(0.0, 60.0, 360.0, 78.0), spans=[_span("[4] Foo et al. arXiv 1234.5678.", x0=0.0, x1=180.0)]),
                    RawLayoutLine(bbox=(0.0, 80.0, 360.0, 98.0), spans=[_span("[5] Bar et al. Conference on Things.", x0=0.0, x1=210.0)]),
                    RawLayoutLine(bbox=(0.0, 100.0, 360.0, 118.0), spans=[_span("[6] Baz et al. ISBN 1234567890.", x0=0.0, x1=180.0)]),
                ],
            ),
        ],
    )

    diagnostics = build_page_math_diagnostics(page)

    assert diagnostics.reference_like is True
    assert diagnostics.experimental_candidate is False
    assert should_use_experimental_math_on_page(page) is False


def test_build_page_math_diagnostics_marks_math_heavy_page_as_candidate() -> None:
    page = RawLayoutPage(
        page_number=12,
        width=400.0,
        height=600.0,
        boxes=[
            RawLayoutBox(
                source_index=0,
                boxclass="text",
                bbox=(0.0, 0.0, 220.0, 20.0),
                textlines=[RawLayoutLine(bbox=(0.0, 0.0, 220.0, 20.0), spans=[_span("Proof. The key is to consider y.", x0=0.0, x1=160.0)])],
            ),
            RawLayoutBox(
                source_index=1,
                boxclass="text",
                bbox=(60.0, 24.0, 220.0, 44.0),
                textlines=[
                    RawLayoutLine(
                        bbox=(60.0, 24.0, 220.0, 44.0),
                        spans=[
                            _span("r", x0=60.0, x1=64.0),
                            _span("D", x0=64.2, x1=68.0, baseline=36.2, size=7.0),
                            _span("[+][(]", x0=68.2, x1=80.0, baseline=31.7, size=7.0),
                            _span("[y]", x0=80.2, x1=86.0, baseline=31.7, size=7.0),
                            _span("[)]", x0=86.2, x1=92.0, baseline=31.7, size=7.0),
                            _span("-ω", x0=96.0, x1=104.0),
                            _span("D", x0=104.2, x1=108.0, baseline=36.2, size=7.0),
                            _span("=[0]", x0=112.0, x1=126.0),
                        ],
                    ),
                ],
            ),
            RawLayoutBox(source_index=2, boxclass="formula", bbox=(40.0, 48.0, 280.0, 80.0)),
        ],
    )

    diagnostics = build_page_math_diagnostics(page)

    assert diagnostics.experimental_candidate is True
    assert diagnostics.reference_like is False
    assert should_use_experimental_math_on_page(page) is True
