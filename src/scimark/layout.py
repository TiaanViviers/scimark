from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RawLayoutSpan:
    text: str
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]
    size: float
    flags: int
    char_flags: int
    font: str
    block: int
    line: int
    direction: tuple[float, float]

    @property
    def baseline(self) -> float:
        return self.origin[1]

    @property
    def is_superscript(self) -> bool:
        return bool(self.flags & 1)

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2)

    @property
    def is_monospace(self) -> bool:
        return bool(self.flags & 8)

    @property
    def is_bold(self) -> bool:
        return bool((self.flags & 16) or (self.char_flags & 8))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline"] = self.baseline
        payload["is_superscript"] = self.is_superscript
        payload["is_italic"] = self.is_italic
        payload["is_monospace"] = self.is_monospace
        payload["is_bold"] = self.is_bold
        return payload


@dataclass(slots=True)
class RawLayoutLine:
    bbox: tuple[float, float, float, float]
    spans: list[RawLayoutSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(slots=True)
class RawLayoutBox:
    source_index: int
    boxclass: str
    bbox: tuple[float, float, float, float]
    textlines: list[RawLayoutLine] = field(default_factory=list)
    has_image: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "boxclass": self.boxclass,
            "bbox": self.bbox,
            "has_image": self.has_image,
            "textlines": [line.to_dict() for line in self.textlines],
        }


@dataclass(slots=True)
class RawLayoutPage:
    page_number: int
    width: float
    height: float
    boxes: list[RawLayoutBox] = field(default_factory=list)
    full_ocred: bool = False
    text_ocred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "full_ocred": self.full_ocred,
            "text_ocred": self.text_ocred,
            "boxes": [box.to_dict() for box in self.boxes],
        }


@dataclass(slots=True)
class RawLayoutDocument:
    source_pdf: str
    page_count: int
    pages: list[RawLayoutPage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": self.source_pdf,
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
        }


def parse_page_spec(page_spec: str | None) -> list[int] | None:
    if page_spec is None:
        return None

    pages: set[int] = set()
    for part in page_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start < 1 or end < 1 or end < start:
                raise ValueError(f"Invalid page range: {token}")
            pages.update(range(start, end + 1))
            continue

        page_number = int(token)
        if page_number < 1:
            raise ValueError(f"Invalid page number: {token}")
        pages.add(page_number)

    if not pages:
        raise ValueError("No pages selected.")

    return sorted(pages)


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return float(x0), float(y0), float(x1), float(y1)


def _build_span(span: dict[str, Any]) -> RawLayoutSpan:
    direction = span.get("dir", (1.0, 0.0))
    return RawLayoutSpan(
        text=str(span.get("text", "")),
        bbox=_normalize_bbox(span["bbox"]),
        origin=(float(span["origin"][0]), float(span["origin"][1])),
        size=float(span.get("size", 0.0)),
        flags=int(span.get("flags", 0)),
        char_flags=int(span.get("char_flags", 0)),
        font=str(span.get("font", "")),
        block=int(span.get("block", 0)),
        line=int(span.get("line", 0)),
        direction=(float(direction[0]), float(direction[1])),
    )


def _build_line(line: dict[str, Any]) -> RawLayoutLine:
    return RawLayoutLine(
        bbox=_normalize_bbox(line["bbox"]),
        spans=[_build_span(span) for span in line.get("spans", [])],
    )


def _build_box(box: Any, source_index: int) -> RawLayoutBox:
    bbox = (float(box.x0), float(box.y0), float(box.x1), float(box.y1))
    return RawLayoutBox(
        source_index=source_index,
        boxclass=str(box.boxclass),
        bbox=bbox,
        textlines=[_build_line(line) for line in box.textlines or []],
        has_image=box.image is not None,
    )


def reorder_page_boxes_for_reading(page: RawLayoutPage) -> RawLayoutPage:
    if len(page.boxes) < 2:
        return page

    midpoint = page.width / 2
    margin = max(18.0, page.width * 0.03)
    left: list[RawLayoutBox] = []
    right: list[RawLayoutBox] = []
    full_width: list[RawLayoutBox] = []

    for box in page.boxes:
        x0, _, x1, _ = box.bbox
        if x0 <= midpoint - margin and x1 >= midpoint + margin:
            full_width.append(box)
        elif x1 <= midpoint + margin:
            left.append(box)
        elif x0 >= midpoint - margin:
            right.append(box)
        elif (x0 + x1) / 2 <= midpoint:
            left.append(box)
        else:
            right.append(box)

    sort_key = lambda box: (box.bbox[1], box.bbox[0], box.source_index)
    left.sort(key=sort_key)
    right.sort(key=sort_key)
    full_width.sort(key=sort_key)

    if not right:
        ordered = full_width + left
        return RawLayoutPage(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            boxes=ordered,
            full_ocred=page.full_ocred,
            text_ocred=page.text_ocred,
        )

    ordered: list[RawLayoutBox] = []
    consumed_left = 0
    consumed_right = 0

    for wide_box in full_width:
        while consumed_left < len(left) and left[consumed_left].bbox[1] < wide_box.bbox[1]:
            ordered.append(left[consumed_left])
            consumed_left += 1
        while consumed_right < len(right) and right[consumed_right].bbox[1] < wide_box.bbox[1]:
            # preserve full-width boxes before right-column content at the same vertical band
            if right[consumed_right].bbox[1] + 4 < wide_box.bbox[1]:
                ordered.append(right[consumed_right])
                consumed_right += 1
            else:
                break
        ordered.append(wide_box)

    ordered.extend(left[consumed_left:])
    ordered.extend(right[consumed_right:])

    return RawLayoutPage(
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        boxes=ordered,
        full_ocred=page.full_ocred,
        text_ocred=page.text_ocred,
    )


def extract_raw_layout_document(
    pdf_path: Path,
    *,
    pages: list[int] | None = None,
) -> RawLayoutDocument:
    try:
        import pymupdf
        from pymupdf4llm.helpers import document_layout
    except ImportError as exc:
        raise RuntimeError("pymupdf4llm and PyMuPDF are required for layout extraction.") from exc

    requested_indexes = None if pages is None else [page_number - 1 for page_number in pages]

    with pymupdf.open(pdf_path) as document:
        parsed = document_layout.parse_document(
            document,
            pages=requested_indexes,
            write_images=False,
            embed_images=False,
            show_progress=False,
            force_text=True,
            use_ocr=True,
            force_ocr=False,
        )

    return RawLayoutDocument(
        source_pdf=str(pdf_path.resolve()),
        page_count=int(parsed.page_count),
        pages=[
            RawLayoutPage(
                page_number=int(page.page_number),
                width=float(page.width),
                height=float(page.height),
                boxes=[_build_box(box, source_index=index) for index, box in enumerate(page.boxes)],
                full_ocred=bool(page.full_ocred),
                text_ocred=bool(page.text_ocred),
            )
            for page in parsed.pages
        ],
    )
