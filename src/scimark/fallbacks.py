from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace
import re
from pathlib import Path
from typing import Any

from scimark.document import StructuralCandidate
from scimark.passes.algorithms import find_algorithm_regions
from scimark.passes.tables import analyze_table, find_tables
from scimark.pipeline import PageMarkdown


BODY_MARKER_RE = re.compile(
    r"\s+\*\*(?:Input|Output|Require|Ensure|for|while|if|end)\*\*",
    re.IGNORECASE,
)
VISUAL_BOX_CLASSES = {"formula", "picture"}
TEXTUAL_BOX_CLASSES = {"text", "caption", "section-header", "list-item"}
PADDING = 12
LOW_CONFIDENCE_TABLE_NOTE = "> Low confidence table detected. Cropped fallback image included below for safety."


def _line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for line in text.splitlines():
        offsets.append(cursor)
        cursor += len(line) + 1
    return offsets


def _region_char_span(text: str, start_line: int, end_line: int) -> tuple[int, int]:
    lines = text.splitlines()
    offsets = _line_offsets(text)
    start = offsets[start_line]
    end = offsets[end_line] + len(lines[end_line])
    return start, end


def _boxes_overlapping_span(
    page_boxes: list[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    overlapping: list[dict[str, Any]] = []
    for box in page_boxes:
        pos_start, pos_end = box.get("pos", (0, 0))
        if pos_end <= start or pos_start >= end:
            continue
        overlapping.append(box)
    return overlapping


def _x_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = min(a[2] - a[0], b[2] - b[0])
    if width <= 0:
        return 0.0
    return overlap / width


def _union_bbox(boxes: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    x0 = min(box["bbox"][0] for box in boxes)
    y0 = min(box["bbox"][1] for box in boxes)
    x1 = max(box["bbox"][2] for box in boxes)
    y1 = max(box["bbox"][3] for box in boxes)
    return x0, y0, x1, y1


def _expand_with_adjacent_visual_box(
    page_boxes: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected:
        return selected

    selected_indexes = {box["index"] for box in selected}
    union = _union_bbox(selected)
    max_index = max(box["index"] for box in selected)

    for box in page_boxes:
        if box["index"] <= max_index or box["index"] in selected_indexes:
            continue
        if box.get("class") not in VISUAL_BOX_CLASSES:
            continue
        bbox = box["bbox"]
        same_column = _x_overlap_ratio(union, bbox) >= 0.5
        close_below = bbox[1] - union[3] <= 48
        if same_column and close_below:
            return selected + [box]

    return selected


def _algorithm_crop_box(
    page: PageMarkdown, candidate: StructuralCandidate
) -> tuple[float, float, float, float] | None:
    source_text = page.raw_markdown or page.markdown
    regions = find_algorithm_regions(source_text)
    if not regions:
        return None

    matching_regions = [region for region in regions if region.label == candidate.label]
    region = matching_regions[0] if matching_regions else regions[0]

    start, end = _region_char_span(source_text, region.start_line, region.end_line)
    overlapping = _boxes_overlapping_span(page.page_boxes, start, end)
    if not overlapping:
        return None

    prioritized = [box for box in overlapping if box.get("class") in TEXTUAL_BOX_CLASSES]
    selected = prioritized or overlapping
    selected = _expand_with_adjacent_visual_box(page.page_boxes, selected)
    return _union_bbox(selected)


def _table_crop_box(page: PageMarkdown, table_index: int) -> tuple[float, float, float, float] | None:
    source_text = page.raw_markdown or page.markdown
    raw_tables = find_tables(source_text)
    low_conf_tables = [table for table in raw_tables if analyze_table(table.lines).low_confidence]
    if table_index >= len(low_conf_tables):
        return None

    table = low_conf_tables[table_index]
    start, end = _region_char_span(source_text, table.start_line, table.end_line)
    overlapping = _boxes_overlapping_span(page.page_boxes, start, end)
    if not overlapping:
        return None

    table_boxes = [box for box in overlapping if box.get("class") == "table"]
    selected = table_boxes or overlapping
    return _union_bbox(selected)


def _char_start_to_line(offsets: list[int], char_pos: int) -> int:
    return max(0, bisect_right(offsets, char_pos) - 1)


def _char_end_to_line(offsets: list[int], char_pos: int, line_count: int) -> int:
    return min(line_count - 1, max(0, bisect_right(offsets, char_pos - 1) - 1))


def _char_to_insertion_line(offsets: list[int], char_pos: int, line_count: int) -> int:
    return min(line_count, bisect_left(offsets, char_pos))


def reorder_algorithm_pages(pages: list[PageMarkdown]) -> list[PageMarkdown]:
    reordered: list[PageMarkdown] = []

    for page in pages:
        source_text = page.raw_markdown or page.markdown
        if not page.page_boxes:
            reordered.append(page)
            continue

        regions = find_algorithm_regions(source_text)
        if not regions:
            reordered.append(page)
            continue

        page_width = max(box["bbox"][2] for box in page.page_boxes)
        midpoint = page_width / 2
        offsets = _line_offsets(source_text)
        lines = source_text.splitlines()
        move_groups: list[tuple[int, int, int]] = []

        for region in regions:
            start_char, end_char = _region_char_span(source_text, region.start_line, region.end_line)
            overlapping = _boxes_overlapping_span(page.page_boxes, start_char, end_char)
            if not overlapping:
                continue

            selected = _expand_with_adjacent_visual_box(page.page_boxes, overlapping)
            bbox = _union_bbox(selected)
            if bbox[0] < midpoint:
                continue

            later_left_boxes = [
                box
                for box in page.page_boxes
                if box["pos"][0] >= max(box_item["pos"][1] for box_item in selected)
                and box["bbox"][0] < midpoint
            ]
            if not later_left_boxes:
                continue

            group_start_char = min(box["pos"][0] for box in selected)
            group_end_char = max(box["pos"][1] for box in selected)
            target_char = max(box["pos"][1] for box in later_left_boxes)

            start_line = _char_start_to_line(offsets, group_start_char)
            end_line = _char_end_to_line(offsets, group_end_char, len(lines))
            target_line = _char_to_insertion_line(offsets, target_char, len(lines))
            move_groups.append((start_line, end_line, target_line))

        if not move_groups:
            reordered.append(page)
            continue

        move_groups.sort()
        merged_groups: list[list[int]] = []
        for start_line, end_line, target_line in move_groups:
            if merged_groups and start_line <= merged_groups[-1][1] + 1:
                merged_groups[-1][1] = max(merged_groups[-1][1], end_line)
                merged_groups[-1][2] = max(merged_groups[-1][2], target_line)
            else:
                merged_groups.append([start_line, end_line, target_line])

        rewritten_lines = lines[:]
        for start_line, end_line, target_line in reversed(merged_groups):
            segment = rewritten_lines[start_line : end_line + 1]
            del rewritten_lines[start_line : end_line + 1]
            if target_line > end_line:
                target_line -= len(segment)
            rewritten_lines[target_line:target_line] = [""] + segment + [""]

        rewritten_markdown = "\n".join(rewritten_lines).rstrip() + "\n"
        reordered.append(replace(page, markdown=rewritten_markdown))

    return reordered


def render_algorithm_region_fallbacks(
    pdf_path: Path,
    fallback_dir: Path,
    pages: list[PageMarkdown],
    candidates: list[StructuralCandidate],
    *,
    dpi: int,
) -> int:
    algorithm_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "algorithm" and candidate.needs_fallback and candidate.source_page is not None
    ]
    if not algorithm_candidates:
        return 0

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed in the active environment.") from exc

    fallback_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
    pages_by_number = {page.page_number: page for page in pages}
    rendered_paths: set[Path] = set()

    with pymupdf.open(pdf_path) as document:
        for candidate in algorithm_candidates:
            assert candidate.source_page is not None
            page_number = candidate.source_page
            page_chunk = pages_by_number.get(page_number)
            page = document.load_page(page_number - 1)
            clip = _algorithm_crop_box(page_chunk, candidate) if page_chunk is not None else None

            if clip is None:
                clip_rect = page.rect
                output_name = f"page-{page_number:04d}.png"
            else:
                clip_rect = pymupdf.Rect(*clip) + (-PADDING, -PADDING, PADDING, PADDING)
                clip_rect = clip_rect & page.rect
                output_name = f"{candidate.block_id}.png"

            output_path = fallback_dir / output_name
            if output_path not in rendered_paths:
                page.get_pixmap(matrix=matrix, alpha=False, clip=clip_rect).save(output_path)
                rendered_paths.add(output_path)
            candidate.fallback_asset_path = str(output_path.resolve())

    return len(rendered_paths)


def render_table_region_fallbacks(
    pdf_path: Path,
    fallback_dir: Path,
    pages: list[PageMarkdown],
    candidates: list[StructuralCandidate],
    *,
    dpi: int,
) -> int:
    table_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "table" and candidate.needs_fallback and candidate.source_page is not None
    ]
    if not table_candidates:
        return 0

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed in the active environment.") from exc

    fallback_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
    pages_by_number = {page.page_number: page for page in pages}
    rendered_paths: set[Path] = set()

    with pymupdf.open(pdf_path) as document:
        grouped: dict[int, list[StructuralCandidate]] = {}
        for candidate in table_candidates:
            assert candidate.source_page is not None
            grouped.setdefault(candidate.source_page, []).append(candidate)

        for page_number, page_candidates in grouped.items():
            page = document.load_page(page_number - 1)
            page_chunk = pages_by_number.get(page_number)
            ordered_candidates = sorted(page_candidates, key=lambda candidate: candidate.start_line)

            for table_index, candidate in enumerate(ordered_candidates):
                clip = _table_crop_box(page_chunk, table_index) if page_chunk is not None else None

                if clip is None:
                    clip_rect = page.rect
                    output_name = f"page-{page_number:04d}.png"
                else:
                    clip_rect = pymupdf.Rect(*clip) + (-PADDING, -PADDING, PADDING, PADDING)
                    clip_rect = clip_rect & page.rect
                    output_name = f"{candidate.block_id}.png"

                output_path = fallback_dir / output_name
                if output_path not in rendered_paths:
                    page.get_pixmap(matrix=matrix, alpha=False, clip=clip_rect).save(output_path)
                    rendered_paths.add(output_path)
                candidate.fallback_asset_path = str(output_path.resolve())

    return len(rendered_paths)


def _extract_algorithm_heading(region_text: str, candidate: StructuralCandidate) -> str:
    for line in region_text.splitlines():
        if "Algorithm" not in line:
            continue
        heading = BODY_MARKER_RE.split(line.strip(), maxsplit=1)[0].rstrip()
        if heading:
            return heading
    if candidate.label:
        return f"**{candidate.label}:**"
    return "**Algorithm:**"


def apply_algorithm_fallbacks(
    markdown: str,
    candidates: list[StructuralCandidate],
    markdown_path: Path,
) -> str:
    algorithm_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "algorithm" and candidate.fallback_asset_path
    ]
    if not algorithm_candidates:
        return markdown

    lines = markdown.splitlines()
    regions = find_algorithm_regions(markdown)
    if not regions:
        return markdown

    replacements: list[tuple[int, int, str, StructuralCandidate]] = []
    for index, (region, candidate) in enumerate(zip(regions, algorithm_candidates, strict=False)):
        replace_start = region.start_line
        while replace_start > 0 and lines[replace_start - 1].strip().startswith("<!-- scimark:"):
            replace_start -= 1

        next_replace_start = len(lines)
        if index + 1 < len(regions):
            next_replace_start = regions[index + 1].start_line
            while (
                next_replace_start > 0
                and lines[next_replace_start - 1].strip().startswith("<!-- scimark:")
            ):
                next_replace_start -= 1

        replace_end = min(region.end_line, next_replace_start - 1)
        region_text = "\n".join(lines[region.start_line : replace_end + 1])
        heading = _extract_algorithm_heading(region_text, candidate)
        image_path = Path(candidate.fallback_asset_path)
        relative_path = image_path.relative_to(markdown_path.parent.resolve()).as_posix()
        replacements.append((replace_start, replace_end, f"![]({relative_path})", candidate))

    output_lines: list[str] = []
    cursor = 0
    for replace_start, replace_end, image_line, candidate in replacements:
        output_lines.extend(lines[cursor:replace_start])
        heading = _extract_algorithm_heading("\n".join(lines[replace_start : replace_end + 1]), candidate)
        candidate.start_line = len(output_lines)
        output_lines.extend([heading, "", image_line, ""])
        candidate.end_line = len(output_lines) - 1
        cursor = replace_end + 1

    output_lines.extend(lines[cursor:])
    return "\n".join(output_lines).rstrip() + "\n"


def apply_table_fallbacks(
    markdown: str,
    candidates: list[StructuralCandidate],
    markdown_path: Path,
) -> str:
    low_confidence_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "table" and candidate.needs_fallback and candidate.fallback_asset_path
    ]
    if not low_confidence_candidates:
        return markdown

    tables = find_tables(markdown)
    if not tables:
        return markdown

    low_confidence_tables = [
        table for table in tables if analyze_table(table.lines).low_confidence
    ]
    lines = markdown.splitlines()
    output_lines: list[str] = []
    cursor = 0

    for table, candidate in zip(low_confidence_tables, low_confidence_candidates, strict=False):
        replace_start = table.start_line
        while replace_start > 0 and lines[replace_start - 1].strip() == "<!-- scimark: low-confidence-table -->":
            replace_start -= 1

        output_lines.extend(lines[cursor:replace_start])
        output_lines.extend(table.lines)
        relative_path = Path(candidate.fallback_asset_path).relative_to(
            markdown_path.parent.resolve()
        ).as_posix()
        output_lines.extend(
            [
                "",
                LOW_CONFIDENCE_TABLE_NOTE,
                "",
                f"![]({relative_path})",
            ]
        )
        cursor = table.end_line + 1

    output_lines.extend(lines[cursor:])
    return "\n".join(output_lines).rstrip() + "\n"

