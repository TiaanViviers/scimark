from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scimark.converter import ConvertOptions, convert_input


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "convert":
        parser.error(f"Unknown command: {args.command}")

    try:
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
    except Exception as exc:
        print(f"scimark: {exc}", file=sys.stderr)
        return 1

    print(
        f"Discovered {summary.total_pdfs_discovered} PDF(s); "
        f"converted {summary.converted}, skipped {summary.skipped}, errored {summary.errored}."
    )
    print(f"Manifest: {summary.manifest_path}")
    print(f"Report: {summary.report_path}")
    return 0
