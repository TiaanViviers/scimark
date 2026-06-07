from __future__ import annotations

from scimark.document import StructuralCandidate
from scimark.passes.algorithms import find_algorithm_regions
from scimark.passes.tables import analyze_table, find_tables


def collect_structural_candidates(markdown: str) -> list[StructuralCandidate]:
    candidates: list[StructuralCandidate] = []

    for index, table in enumerate(find_tables(markdown), start=1):
        stats = analyze_table(table.lines)
        candidates.append(
            StructuralCandidate(
                block_id=f"table-{index}",
                kind="table",
                start_line=table.start_line,
                end_line=table.end_line,
                label=f"Table {index}",
                low_confidence=stats.low_confidence,
                needs_fallback=stats.low_confidence,
                reasons=stats.low_confidence_reasons.copy(),
            )
        )

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
