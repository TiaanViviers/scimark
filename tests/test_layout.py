from pathlib import Path

import pymupdf

from scimark.layout import RawLayoutBox, RawLayoutPage, extract_raw_layout_document, parse_page_spec, reorder_page_boxes_for_reading


def test_parse_page_spec_supports_single_pages_and_ranges() -> None:
    assert parse_page_spec(None) is None
    assert parse_page_spec("3") == [3]
    assert parse_page_spec("3,1-2,5") == [1, 2, 3, 5]


def test_parse_page_spec_rejects_invalid_ranges() -> None:
    for value in ("0", "3-2", "", "1-0"):
        try:
            parse_page_spec(value)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {value!r}")


def test_extract_raw_layout_document_returns_serializable_span_data(tmp_path: Path) -> None:
    pdf_path = tmp_path / "layout.pdf"
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Lemma A.1")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "W_i^Q in readable text")
    document.save(pdf_path)
    document.close()

    raw_layout = extract_raw_layout_document(pdf_path, pages=[2])

    assert raw_layout.source_pdf == str(pdf_path.resolve())
    assert raw_layout.page_count == 2
    assert len(raw_layout.pages) == 1

    page = raw_layout.pages[0]
    assert page.page_number == 2
    assert page.boxes

    first_box = page.boxes[0]
    assert isinstance(first_box.boxclass, str)
    assert first_box.textlines
    assert first_box.textlines[0].spans

    first_span = first_box.textlines[0].spans[0]
    payload = raw_layout.to_dict()

    assert isinstance(first_span.baseline, float)
    assert "pages" in payload
    assert payload["pages"][0]["boxes"][0]["textlines"][0]["spans"][0]["text"]


def test_reorder_page_boxes_for_reading_restores_left_then_right_column_flow() -> None:
    page = RawLayoutPage(
        page_number=11,
        width=600.0,
        height=800.0,
        boxes=[
            RawLayoutBox(source_index=0, boxclass="section-header", bbox=(40.0, 10.0, 560.0, 40.0)),
            RawLayoutBox(source_index=1, boxclass="text", bbox=(330.0, 60.0, 560.0, 100.0)),
            RawLayoutBox(source_index=2, boxclass="section-header", bbox=(40.0, 50.0, 260.0, 80.0)),
            RawLayoutBox(source_index=3, boxclass="text", bbox=(40.0, 90.0, 260.0, 150.0)),
            RawLayoutBox(source_index=4, boxclass="text", bbox=(330.0, 120.0, 560.0, 180.0)),
        ],
    )

    ordered = reorder_page_boxes_for_reading(page)

    assert [box.source_index for box in ordered.boxes] == [0, 2, 3, 1, 4]
