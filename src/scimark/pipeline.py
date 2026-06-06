from __future__ import annotations

from dataclasses import dataclass

from scimark.document import DocumentStats
from scimark.passes.confidence import annotate_low_confidence_math
from scimark.passes.fix_paths import normalize_image_links
from scimark.passes.figures import keep_captions_with_figures
from scimark.passes.math_cleanup import cleanup_math_text
from scimark.passes.page_noise import strip_page_number_lines
from scimark.passes.picture_text import remove_picture_text_blocks
from scimark.passes.tables import annotate_low_confidence_tables


@dataclass(slots=True)
class PipelineOptions:
    pdf_stem: str
    strip_picture_text: bool = True
    strip_page_numbers: bool = True


@dataclass(slots=True)
class PipelineResult:
    markdown: str
    stats: DocumentStats


def run_pipeline(markdown: str, options: PipelineOptions) -> PipelineResult:
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

    stats.tables_detected = len(stats.table_stats)
    stats.low_confidence_tables = sum(1 for table in stats.table_stats if table.low_confidence)

    return PipelineResult(markdown=current, stats=stats)
