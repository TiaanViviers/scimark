from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scimark.converter import ConvertOptions, convert_input
from scimark.layout import extract_raw_layout_document, parse_page_spec
from scimark.math_spans import (
    build_paper_math_diagnostics,
    build_math_debug_report,
    build_region_review_report,
    serialize_page,
    serialize_page_with_region_promotion,
)
from scimark.paths import discover_pdfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scimark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert PDFs to Markdown.")
    convert_parser.add_argument("input_path", help="PDF file or directory of PDFs.")
    convert_parser.add_argument("--out", required=True, help="Directory to write Markdown outputs into.")
    convert_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover PDFs when input_path is a directory.",
    )
    convert_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown output files.",
    )
    convert_parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image DPI for extracted assets. Default: 200.",
    )
    convert_parser.add_argument(
        "--strip-picture-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove PyMuPDF4LLM picture text blocks. Enabled by default.",
    )
    convert_parser.add_argument(
        "--strip-page-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove standalone page-number lines. Enabled by default.",
    )
    convert_parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Store raw PyMuPDF4LLM Markdown in _scimark/raw/.",
    )
    convert_parser.add_argument(
        "--math-mode",
        choices=("legacy", "experimental"),
        default="legacy",
        help="Math serialization mode. 'experimental' selectively uses the span-based math path on candidate pages.",
    )

    dump_layout_parser = subparsers.add_parser(
        "dump-layout",
        help="Dump raw PyMuPDF4LLM layout spans/boxes as JSON for debugging.",
    )
    dump_layout_parser.add_argument("pdf_path", help="PDF file to inspect.")
    dump_layout_parser.add_argument("--out", required=True, help="JSON file to write.")
    dump_layout_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )

    debug_math_parser = subparsers.add_parser(
        "debug-math",
        help="Render a human-readable math/span debug report for selected PDF pages.",
    )
    debug_math_parser.add_argument("pdf_path", help="PDF file to inspect.")
    debug_math_parser.add_argument("--out", required=True, help="Text file to write.")
    debug_math_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )
    debug_math_parser.add_argument(
        "--all-boxes",
        action="store_true",
        help="Include non-math-like boxes in the debug report.",
    )

    prototype_math_parser = subparsers.add_parser(
        "prototype-math",
        help="Render the experimental reordered page-level math/prose prototype.",
    )
    prototype_math_parser.add_argument("pdf_path", help="PDF file to inspect.")
    prototype_math_parser.add_argument("--out", required=True, help="Text file to write.")
    prototype_math_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )

    promoted_prototype_math_parser = subparsers.add_parser(
        "prototype-math-promoted",
        help="Render the prototype using conservative region-level promotion for high-confidence display-math blocks.",
    )
    promoted_prototype_math_parser.add_argument("pdf_path", help="PDF file to inspect.")
    promoted_prototype_math_parser.add_argument("--out", required=True, help="Text file to write.")
    promoted_prototype_math_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )

    debug_regions_parser = subparsers.add_parser(
        "debug-regions",
        help="Render a region review report showing conservative region-serializer promotion decisions.",
    )
    debug_regions_parser.add_argument("pdf_path", help="PDF file to inspect.")
    debug_regions_parser.add_argument("--out", required=True, help="Text file to write.")
    debug_regions_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )

    eval_math_parser = subparsers.add_parser(
        "eval-math",
        help="Run baseline conversion plus experimental math outputs over a PDF or corpus.",
    )
    eval_math_parser.add_argument("input_path", help="PDF file or directory of PDFs.")
    eval_math_parser.add_argument("--out", required=True, help="Directory to write evaluation outputs into.")
    eval_math_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover PDFs when input_path is a directory.",
    )
    eval_math_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation outputs.",
    )
    eval_math_parser.add_argument(
        "--pages",
        help="1-based page selection like '12' or '12-14,18'. Defaults to all pages.",
    )
    eval_math_parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the normal baseline convert pass and only write experimental math outputs.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "convert":
            summary = convert_input(
                Path(args.input_path),
                Path(args.out),
                ConvertOptions(
                    recursive=args.recursive,
                    overwrite=args.overwrite,
                    dpi=args.dpi,
                    strip_picture_text=args.strip_picture_text,
                    strip_page_numbers=args.strip_page_numbers,
                    keep_raw=args.keep_raw,
                    math_mode=args.math_mode,
                ),
            )
            print(
                f"Discovered {summary.total_pdfs_discovered} PDF(s); "
                f"converted {summary.converted}, skipped {summary.skipped}, errored {summary.errored}."
            )
            print(f"Manifest: {summary.manifest_path}")
            print(f"Report: {summary.report_path}")
            return 0

        if args.command == "dump-layout":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(layout_document.to_dict(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote layout JSON: {output_path.resolve()}")
            return 0

        if args.command == "debug-math":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            report = build_math_debug_report(
                layout_document,
                include_all_boxes=args.all_boxes,
            )
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"Wrote math debug report: {output_path.resolve()}")
            return 0

        if args.command == "prototype-math":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            page_text = []
            for page in layout_document.pages:
                serialized = serialize_page(page)
                page_text.append(f"page {page.page_number}\n")
                if serialized:
                    page_text.append(serialized)
                    page_text.append("\n")
                page_text.append("\n")
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("".join(page_text).rstrip() + "\n", encoding="utf-8")
            print(f"Wrote math prototype: {output_path.resolve()}")
            return 0

        if args.command == "prototype-math-promoted":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            page_text = []
            for page in layout_document.pages:
                serialized = serialize_page_with_region_promotion(page)
                page_text.append(f"page {page.page_number}\n")
                if serialized:
                    page_text.append(serialized)
                    page_text.append("\n")
                page_text.append("\n")
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("".join(page_text).rstrip() + "\n", encoding="utf-8")
            print(f"Wrote promoted math prototype: {output_path.resolve()}")
            return 0

        if args.command == "debug-regions":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            report = build_region_review_report(layout_document)
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"Wrote region review report: {output_path.resolve()}")
            return 0

        if args.command == "eval-math":
            input_path = Path(args.input_path)
            output_root = Path(args.out)
            page_selection = parse_page_spec(args.pages)
            pdfs = discover_pdfs(input_path, recursive=args.recursive)
            output_root.mkdir(parents=True, exist_ok=True)

            baseline_dir = output_root / "baseline"
            if not args.skip_convert:
                print(f"Running baseline conversion for {len(pdfs)} PDF(s)...", flush=True)
                for index, pdf_path in enumerate(pdfs, start=1):
                    print(f"[baseline {index}/{len(pdfs)}] {pdf_path.name}", flush=True)
                    convert_input(
                        pdf_path,
                        baseline_dir,
                        ConvertOptions(
                            overwrite=args.overwrite,
                        ),
                    )
                    print(f"[baseline {index}/{len(pdfs)}] done: {pdf_path.stem}", flush=True)

            experimental_root = output_root / "experimental"
            summary: dict[str, object] = {
                "source_input": str(input_path.resolve()),
                "pdf_count": len(pdfs),
                "pages": args.pages or "all",
                "baseline_dir": str(baseline_dir.resolve()) if baseline_dir.exists() else None,
                "papers": [],
            }

            print(f"Running experimental math evaluation for {len(pdfs)} PDF(s)...", flush=True)
            for index, pdf_path in enumerate(pdfs, start=1):
                print(f"[experimental {index}/{len(pdfs)}] {pdf_path.name}", flush=True)
                layout_document = extract_raw_layout_document(pdf_path, pages=page_selection)
                paper_dir = experimental_root / pdf_path.stem
                paper_dir.mkdir(parents=True, exist_ok=True)
                diagnostics = build_paper_math_diagnostics(layout_document)

                prototype_parts: list[str] = []
                promoted_parts: list[str] = []
                for page in layout_document.pages:
                    prototype_parts.append(f"page {page.page_number}\n")
                    prototype_parts.append(serialize_page(page))
                    prototype_parts.append("\n\n")
                    promoted_parts.append(f"page {page.page_number}\n")
                    promoted_parts.append(serialize_page_with_region_promotion(page))
                    promoted_parts.append("\n\n")

                prototype_path = paper_dir / "prototype.txt"
                promoted_path = paper_dir / "promoted.txt"
                regions_path = paper_dir / "regions.txt"
                debug_path = paper_dir / "debug-math.txt"

                prototype_path.write_text("".join(prototype_parts).rstrip() + "\n", encoding="utf-8")
                promoted_path.write_text("".join(promoted_parts).rstrip() + "\n", encoding="utf-8")
                regions_path.write_text(build_region_review_report(layout_document), encoding="utf-8")
                debug_path.write_text(build_math_debug_report(layout_document), encoding="utf-8")

                summary["papers"].append(
                    {
                        "pdf": str(pdf_path.resolve()),
                        "prototype": str(prototype_path.resolve()),
                        "promoted": str(promoted_path.resolve()),
                        "regions": str(regions_path.resolve()),
                        "debug_math": str(debug_path.resolve()),
                        "diagnostics": diagnostics.to_dict(),
                    }
                )
                print(
                    "[experimental "
                    f"{index}/{len(pdfs)}] metrics: display={diagnostics.display_math_regions} "
                    f"promoted={diagnostics.promoted_regions} formula_splits={diagnostics.formula_box_split_regions} "
                    f"embedded_runs={diagnostics.embedded_display_run_regions}",
                    flush=True,
                )
                print(f"[experimental {index}/{len(pdfs)}] done: {pdf_path.stem}", flush=True)

            summary_path = output_root / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Evaluated {len(pdfs)} PDF(s)")
            print(f"Summary: {summary_path.resolve()}")
            print(f"Experimental outputs: {experimental_root.resolve()}")
            if not args.skip_convert:
                print(f"Baseline outputs: {baseline_dir.resolve()}")
            return 0
    except Exception as exc:
        print(f"scimark: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 1
