from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scimark.converter import ConvertOptions, convert_input
from scimark.layout import extract_raw_layout_document, parse_page_spec
from scimark.math_spans import build_math_debug_report, build_region_review_report, serialize_page


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

        if args.command == "debug-regions":
            page_selection = parse_page_spec(args.pages)
            layout_document = extract_raw_layout_document(Path(args.pdf_path), pages=page_selection)
            report = build_region_review_report(layout_document)
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"Wrote region review report: {output_path.resolve()}")
            return 0
    except Exception as exc:
        print(f"scimark: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 1
