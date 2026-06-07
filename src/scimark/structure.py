from __future__ import annotations

from dataclasses import replace

from scimark.document import DocumentStats, StructuralCandidate
from scimark.markdown_blocks import parse_markdown_blocks
from scimark.passes.algorithms import find_algorithm_regions
from scimark.passes.tables import analyze_table


def collect_table_candidates(markdown: str) -> list[StructuralCandidate]:
    candidates: list[StructuralCandidate] = []

    for index, block in enumerate(
        (block for block in parse_markdown_blocks(markdown) if block.block_type == "table"),
        start=1,
    ):
        stats = analyze_table(block.lines)
        candidates.append(
            StructuralCandidate(
                block_id=f"table-{index}",
                kind="table",
                start_line=block.start_line,
                end_line=block.end_line,
                label=f"Table {index}",
                low_confidence=stats.low_confidence,
                needs_fallback=stats.low_confidence,
                reasons=stats.low_confidence_reasons.copy(),
            )
        )

    return candidates


def collect_structural_candidates(markdown: str) -> list[StructuralCandidate]:
    candidates = collect_table_candidates(markdown)

    for index, region in enumerate(find_algorithm_regions(markdown), start=1):
        candidates.append(
            StructuralCandidate(
                block_id=f"algorithm-{index}",
                kind="algorithm",
                start_line=region.start_line,
                end_line=region.end_line,
                label=region.label or f"Algorithm {index}",
                low_confidence=True,
                needs_fallback=True,
                reasons=["algorithm-block", "requires-screenshot"],
            )
        )

    candidates.sort(key=lambda candidate: (candidate.start_line, candidate.end_line, candidate.kind))
    return candidates


def renumber_structural_candidates(candidates: list[StructuralCandidate]) -> None:
    kind_counts: dict[str, int] = {}

    for candidate in candidates:
        kind_counts[candidate.kind] = kind_counts.get(candidate.kind, 0) + 1
        index = kind_counts[candidate.kind]
        candidate.block_id = f"{candidate.kind}-{index}"
        if candidate.kind == "table":
            candidate.label = f"Table {index}"
        elif candidate.kind == "algorithm" and not candidate.label:
            candidate.label = f"Algorithm {index}"


def finalize_document_stats(stats: DocumentStats) -> None:
    stats.tables_detected = len(stats.table_stats)
    stats.low_confidence_tables = sum(1 for table in stats.table_stats if table.low_confidence)
    renumber_structural_candidates(stats.structural_candidates)
    stats.algorithm_blocks_detected = sum(
        1 for candidate in stats.structural_candidates if candidate.kind == "algorithm"
    )


def merge_page_stats(
    aggregate: DocumentStats,
    page_stats: DocumentStats,
    *,
    line_offset: int,
    source_page: int,
) -> None:
    aggregate.images_saved += page_stats.images_saved
    aggregate.picture_text_blocks_removed += page_stats.picture_text_blocks_removed
    aggregate.page_number_lines_removed += page_stats.page_number_lines_removed
    aggregate.figure_caption_adjustments += page_stats.figure_caption_adjustments
    aggregate.tables_detected += page_stats.tables_detected
    aggregate.low_confidence_tables += page_stats.low_confidence_tables
    aggregate.low_confidence_math_regions += page_stats.low_confidence_math_regions
    aggregate.table_stats.extend(page_stats.table_stats)

    for candidate in page_stats.structural_candidates:
        aggregate.structural_candidates.append(
            replace(
                candidate,
                start_line=candidate.start_line + line_offset,
                end_line=candidate.end_line + line_offset,
                source_page=source_page,
            )
        )
