from __future__ import annotations

from dataclasses import dataclass, replace

from scimark.document import DocumentStats
from scimark.passes.algorithms import annotate_algorithm_blocks
from scimark.passes.confidence import annotate_low_confidence_math
from scimark.passes.fix_paths import normalize_image_links
from scimark.passes.figures import keep_captions_with_figures
from scimark.passes.math_cleanup import cleanup_math_text
from scimark.passes.page_noise import strip_page_number_lines
from scimark.passes.picture_text import remove_picture_text_blocks
from scimark.passes.tables import annotate_low_confidence_tables
from scimark.structure import collect_structural_candidates


@dataclass(slots=True)
class PipelineOptions:
    pdf_stem: str
    strip_picture_text: bool = True
    strip_page_numbers: bool = True


@dataclass(slots=True)
class PageMarkdown:
    page_number: int
    markdown: str


@dataclass(slots=True)
class PipelineResult:
    markdown: str
    stats: DocumentStats


def _renumber_structural_candidates(stats: DocumentStats) -> None:
    kind_counts: dict[str, int] = {}

    for candidate in stats.structural_candidates:
        kind_counts[candidate.kind] = kind_counts.get(candidate.kind, 0) + 1
        index = kind_counts[candidate.kind]
        candidate.block_id = f"{candidate.kind}-{index}"
        if candidate.kind == "table":
            candidate.label = f"Table {index}"
        elif candidate.kind == "algorithm" and not candidate.label:
            candidate.label = f"Algorithm {index}"


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

    stats.tables_detected = len(stats.table_stats)
    stats.low_confidence_tables = sum(1 for table in stats.table_stats if table.low_confidence)
    stats.structural_candidates = collect_structural_candidates(current)
    _renumber_structural_candidates(stats)
    stats.algorithm_blocks_detected = sum(
        1 for candidate in stats.structural_candidates if candidate.kind == "algorithm"
    )

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

        aggregate.images_saved += page_result.stats.images_saved
        aggregate.picture_text_blocks_removed += page_result.stats.picture_text_blocks_removed
        aggregate.page_number_lines_removed += page_result.stats.page_number_lines_removed
        aggregate.figure_caption_adjustments += page_result.stats.figure_caption_adjustments
        aggregate.tables_detected += page_result.stats.tables_detected
        aggregate.low_confidence_tables += page_result.stats.low_confidence_tables
        aggregate.low_confidence_math_regions += page_result.stats.low_confidence_math_regions
        aggregate.table_stats.extend(page_result.stats.table_stats)

        for candidate in page_result.stats.structural_candidates:
            aggregate.structural_candidates.append(
                replace(
                    candidate,
                    start_line=candidate.start_line + line_offset,
                    end_line=candidate.end_line + line_offset,
                    source_page=page.page_number,
                )
            )

        combined_lines.extend(page_lines)
        line_offset += len(page_lines)

        if page_index < len(pages) - 1:
            combined_lines.append("")
            line_offset += 1

    aggregate.algorithm_blocks_detected = sum(
        1 for candidate in aggregate.structural_candidates if candidate.kind == "algorithm"
    )
    _renumber_structural_candidates(aggregate)

    return PipelineResult(markdown="\n".join(combined_lines).rstrip() + "\n", stats=aggregate)
