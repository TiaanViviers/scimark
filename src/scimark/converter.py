from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from scimark.document import ConversionSummary, ManifestEntry
from scimark.paths import discover_pdfs
from scimark.pipeline import PageMarkdown, PipelineOptions, run_pipeline_pages
from scimark.report import write_manifest, write_report


@dataclass(slots=True)
class ConvertOptions:
    recursive: bool = False
    overwrite: bool = False
    dpi: int = 200
    strip_picture_text: bool = True
    strip_page_numbers: bool = True
    keep_raw: bool = False


def _render_pdf_to_markdown_pages(pdf_path: Path, asset_dir: Path, dpi: int) -> list[PageMarkdown]:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError("pymupdf4llm is not installed in the active environment.") from exc

    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        write_images=True,
        image_path=str(asset_dir),
        image_format="png",
        image_dpi=dpi,
    )

    pages: list[PageMarkdown] = []
    for page_index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.get("metadata", {}))
        page_number = int(metadata.get("page_number", page_index))
        pages.append(PageMarkdown(page_number=page_number, markdown=chunk.get("text", "")))

    return pages


def _count_images(asset_dir: Path) -> int:
    return sum(1 for path in asset_dir.glob("*.png") if path.is_file())


def convert_input(input_path: Path, output_dir: Path, options: ConvertOptions) -> ConversionSummary:
    discovered_pdfs = discover_pdfs(input_path, recursive=options.recursive)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets_root = output_dir / "_assets"
    metadata_root = output_dir / "_scimark"
    metadata_root.mkdir(parents=True, exist_ok=True)
    raw_root = metadata_root / "raw"

    manifest_entries: list[ManifestEntry] = []
    reserved_outputs: set[Path] = set()
    batch_start = time.perf_counter()

    for pdf_path in discovered_pdfs:
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        asset_dir = assets_root / pdf_path.stem
        raw_markdown_path = raw_root / f"{pdf_path.stem}.md"

        if markdown_path in reserved_outputs:
            manifest_entries.append(
                ManifestEntry(
                    source_pdf=str(pdf_path.resolve()),
                    output_markdown=str(markdown_path.resolve()),
                    asset_dir=str(asset_dir.resolve()),
                    status="skipped",
                    reason="duplicate output stem in this conversion batch",
                )
            )
            continue

        reserved_outputs.add(markdown_path)

        if markdown_path.exists() and not options.overwrite:
            manifest_entries.append(
                ManifestEntry(
                    source_pdf=str(pdf_path.resolve()),
                    output_markdown=str(markdown_path.resolve()),
                    asset_dir=str(asset_dir.resolve()),
                    status="skipped",
                    reason="output markdown already exists",
                )
            )
            continue

        started = time.perf_counter()

        try:
            if asset_dir.exists() and options.overwrite:
                shutil.rmtree(asset_dir)

            asset_dir.mkdir(parents=True, exist_ok=True)

            raw_pages = _render_pdf_to_markdown_pages(pdf_path, asset_dir, options.dpi)
            raw_markdown = "\n\n".join(page.markdown.rstrip("\n") for page in raw_pages).rstrip() + "\n"
            if options.keep_raw:
                raw_root.mkdir(parents=True, exist_ok=True)
                raw_markdown_path.write_text(raw_markdown, encoding="utf-8")

            pipeline_result = run_pipeline_pages(
                raw_pages,
                PipelineOptions(
                    pdf_stem=pdf_path.stem,
                    strip_picture_text=options.strip_picture_text,
                    strip_page_numbers=options.strip_page_numbers,
                ),
            )

            markdown_path.write_text(pipeline_result.markdown, encoding="utf-8")
            pipeline_result.stats.images_saved = _count_images(asset_dir)
            elapsed = time.perf_counter() - started

            manifest_entries.append(
                ManifestEntry(
                    source_pdf=str(pdf_path.resolve()),
                    output_markdown=str(markdown_path.resolve()),
                    asset_dir=str(asset_dir.resolve()),
                    status="converted",
                    images_saved=pipeline_result.stats.images_saved,
                    picture_text_blocks_removed=pipeline_result.stats.picture_text_blocks_removed,
                    page_number_lines_removed=pipeline_result.stats.page_number_lines_removed,
                    figure_caption_adjustments=pipeline_result.stats.figure_caption_adjustments,
                    tables_detected=pipeline_result.stats.tables_detected,
                    algorithm_blocks_detected=pipeline_result.stats.algorithm_blocks_detected,
                    low_confidence_tables=pipeline_result.stats.low_confidence_tables,
                    low_confidence_math_regions=pipeline_result.stats.low_confidence_math_regions,
                    processing_time_seconds=round(elapsed, 3),
                    table_stats=pipeline_result.stats.table_stats,
                    structural_candidates=pipeline_result.stats.structural_candidates,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            manifest_entries.append(
                ManifestEntry(
                    source_pdf=str(pdf_path.resolve()),
                    output_markdown=str(markdown_path.resolve()),
                    asset_dir=str(asset_dir.resolve()),
                    status="error",
                    reason=str(exc),
                    processing_time_seconds=round(elapsed, 3),
                )
            )

    manifest_path = metadata_root / "manifest.json"
    report_path = metadata_root / "report.json"
    total_time = time.perf_counter() - batch_start
    write_manifest(manifest_path, manifest_entries)
    write_report(report_path, manifest_entries, len(discovered_pdfs), total_time)

    converted = sum(1 for entry in manifest_entries if entry.status == "converted")
    skipped = sum(1 for entry in manifest_entries if entry.status == "skipped")
    errored = sum(1 for entry in manifest_entries if entry.status == "error")

    return ConversionSummary(
        total_pdfs_discovered=len(discovered_pdfs),
        converted=converted,
        skipped=skipped,
        errored=errored,
        manifest_path=str(manifest_path.resolve()),
        report_path=str(report_path.resolve()),
    )
