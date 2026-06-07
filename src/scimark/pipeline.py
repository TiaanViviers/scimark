from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scimark.document import DocumentStats
from scimark.passes.algorithms import annotate_algorithm_blocks
from scimark.passes.confidence import annotate_low_confidence_math
from scimark.passes.fix_paths import normalize_image_links
from scimark.passes.figures import keep_captions_with_figures
from scimark.passes.math_cleanup import cleanup_math_text
from scimark.passes.page_noise import strip_page_number_lines
from scimark.passes.picture_text import remove_picture_text_blocks
from scimark.passes.tables import annotate_low_confidence_tables
from scimark.structure import (
    collect_structural_candidates,
    finalize_document_stats,
    merge_page_stats,
)


@dataclass(slots=True)
class PipelineOptions:
    pdf_stem: str
    strip_picture_text: bool = True
    strip_page_numbers: bool = True


@dataclass(slots=True)
class PageMarkdown:
    page_number: int
    markdown: str
    raw_markdown: str = ""
    page_boxes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    markdown: str
    stats: DocumentStats


def _run_pipeline_single(markdown: str, options: PipelineOptions) -> PipelineResult:
    stats = DocumentStats()
    current = markdown

    if options.strip_picture_text:
        current, stats.picture_text_blocks_removed = remove_picture_text_blocks(current)

    if options.strip_page_numbers:
        current, stats.page_number_lines_removed = strip_page_number_lines(current)

    current = cleanup_math_text(current)
    current, stats.table_stats = annotate_low_confidence_tables(current)
    current, stats.low_confidence_math_regions = annotate_low_confidence_math(current)
    current, _ = normalize_image_links(current, options.pdf_stem)
    current, stats.figure_caption_adjustments = keep_captions_with_figures(current)
    current = cleanup_math_text(current)
    current, _ = annotate_algorithm_blocks(current)

    stats.structural_candidates = collect_structural_candidates(current)
    finalize_document_stats(stats)

    return PipelineResult(markdown=current, stats=stats)


def run_pipeline(markdown: str, options: PipelineOptions) -> PipelineResult:
    return _run_pipeline_single(markdown, options)


def run_pipeline_pages(pages: list[PageMarkdown], options: PipelineOptions) -> PipelineResult:
    if not pages:
        return PipelineResult(markdown="", stats=DocumentStats())

    combined_lines: list[str] = []
    aggregate = DocumentStats()
    line_offset = 0

    for page_index, page in enumerate(pages):
        page_result = _run_pipeline_single(page.markdown, options)
        page_lines = page_result.markdown.rstrip("\n").splitlines()

        merge_page_stats(
            aggregate,
            page_result.stats,
            line_offset=line_offset,
            source_page=page.page_number,
        )

        combined_lines.extend(page_lines)
        line_offset += len(page_lines)

        if page_index < len(pages) - 1:
            combined_lines.append("")
            line_offset += 1

    finalize_document_stats(aggregate)

    return PipelineResult(markdown="\n".join(combined_lines).rstrip() + "\n", stats=aggregate)
