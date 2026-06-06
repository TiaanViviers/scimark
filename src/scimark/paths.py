from __future__ import annotations

from pathlib import Path


def discover_pdfs(input_path: Path, recursive: bool = False) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(input_path.glob(pattern), key=lambda path: str(path).lower())
    return [path for path in pdfs if path.is_file()]
