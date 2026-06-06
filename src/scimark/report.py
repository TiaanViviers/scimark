from __future__ import annotations

import json
from pathlib import Path

from scimark.document import ManifestEntry


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    payload = [entry.to_dict() for entry in entries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    entries: list[ManifestEntry],
    total_pdfs_discovered: int,
    total_processing_time_seconds: float,
) -> None:
    converted = [entry for entry in entries if entry.status == "converted"]
    skipped = [entry for entry in entries if entry.status == "skipped"]
    errored = [entry for entry in entries if entry.status == "error"]

    all_table_stats = [table for entry in entries for table in entry.table_stats]

    payload = {
        "total_pdfs_discovered": total_pdfs_discovered,
        "pdfs_converted": len(converted),
        "pdfs_skipped": len(skipped),
        "pdfs_errored": len(errored),
        "total_images_saved": sum(entry.images_saved for entry in entries),
        "total_picture_text_blocks_removed": sum(
            entry.picture_text_blocks_removed for entry in entries
        ),
        "total_page_number_lines_removed": sum(entry.page_number_lines_removed for entry in entries),
        "total_figure_caption_adjustments": sum(
            entry.figure_caption_adjustments for entry in entries
        ),
        "total_markdown_tables_detected": sum(entry.tables_detected for entry in entries),
        "total_low_confidence_tables": sum(entry.low_confidence_tables for entry in entries),
        "total_unusually_large_table_cells": sum(
            table.unusually_large_cells for table in all_table_stats
        ),
        "total_repeated_stacked_values": sum(
            table.repeated_stacked_values for table in all_table_stats
        ),
        "total_stacked_row_mismatches": sum(
            table.stacked_row_mismatches for table in all_table_stats
        ),
        "total_low_confidence_math_regions": sum(
            entry.low_confidence_math_regions for entry in entries
        ),
        "total_processing_time_seconds": round(total_processing_time_seconds, 3),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
