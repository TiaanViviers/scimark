from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TableStats:
    rows: int
    columns: int
    consistent_columns: bool
    empty_cells: int
    empty_cell_ratio: float
    br_tag_count: int
    unusually_large_cells: int
    repeated_stacked_values: int
    stacked_row_mismatches: int
    max_stacked_segments: int
    low_confidence: bool
    low_confidence_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuralCandidate:
    block_id: str
    kind: str
    start_line: int
    end_line: int
    source_page: int | None = None
    label: str | None = None
    low_confidence: bool = False
    needs_fallback: bool = False
    fallback_asset_path: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentStats:
    images_saved: int = 0
    fallback_assets_generated: int = 0
    picture_text_blocks_removed: int = 0
    page_number_lines_removed: int = 0
    figure_caption_adjustments: int = 0
    tables_detected: int = 0
    algorithm_blocks_detected: int = 0
    low_confidence_tables: int = 0
    low_confidence_math_regions: int = 0
    table_stats: list[TableStats] = field(default_factory=list)
    structural_candidates: list[StructuralCandidate] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "images_saved": self.images_saved,
            "fallback_assets_generated": self.fallback_assets_generated,
            "picture_text_blocks_removed": self.picture_text_blocks_removed,
            "page_number_lines_removed": self.page_number_lines_removed,
            "figure_caption_adjustments": self.figure_caption_adjustments,
            "tables_detected": self.tables_detected,
            "algorithm_blocks_detected": self.algorithm_blocks_detected,
            "low_confidence_tables": self.low_confidence_tables,
            "low_confidence_math_regions": self.low_confidence_math_regions,
            "table_stats": [table.to_dict() for table in self.table_stats],
            "structural_candidates": [candidate.to_dict() for candidate in self.structural_candidates],
        }


@dataclass(slots=True)
class ManifestEntry:
    source_pdf: str
    output_markdown: str
    asset_dir: str
    status: str
    reason: str | None = None
    images_saved: int = 0
    fallback_assets_generated: int = 0
    picture_text_blocks_removed: int = 0
    page_number_lines_removed: int = 0
    figure_caption_adjustments: int = 0
    tables_detected: int = 0
    algorithm_blocks_detected: int = 0
    low_confidence_tables: int = 0
    low_confidence_math_regions: int = 0
    processing_time_seconds: float = 0.0
    table_stats: list[TableStats] = field(default_factory=list)
    structural_candidates: list[StructuralCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["table_stats"] = [table.to_dict() for table in self.table_stats]
        payload["structural_candidates"] = [
            candidate.to_dict() for candidate in self.structural_candidates
        ]
        return payload


@dataclass(slots=True)
class ConversionSummary:
    total_pdfs_discovered: int
    converted: int
    skipped: int
    errored: int
    manifest_path: str
    report_path: str
