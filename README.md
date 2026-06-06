# scimark

Lightweight scientific PDF to Markdown parser for academic papers. `scimark` uses `pymupdf4llm` for extraction, then applies a small deterministic cleanup pipeline to improve Markdown quality without adding heavyweight dependencies.

## Install

Activate your project environment first, then install in editable mode:

```bash
source /home/tiaan/environments/pyenv/bin/activate
python -m pip install -e .[dev]
```

## Usage

Convert one PDF:

```bash
scimark convert resources/Attention.pdf --out parsed/
```

Convert a directory:

```bash
scimark convert resources/ --out parsed/
```

Convert recursively:

```bash
scimark convert resources/ --out parsed/ --recursive
```

Optional flags:

- `--overwrite` replaces existing Markdown outputs.
- `--dpi 200` controls extracted image DPI.
- `--keep-raw` stores raw backend Markdown under `_scimark/raw/`.
- `--no-strip-picture-text` keeps picture-text blocks.
- `--no-strip-page-numbers` keeps standalone numeric page lines.

## Output Layout

```text
parsed/
├── Attention.md
├── XGboost.md
├── _assets/
│   ├── Attention/
│   └── XGboost/
└── _scimark/
    ├── manifest.json
    ├── report.json
    └── raw/
```

Image links are normalized to Markdown-relative paths such as:

```markdown
![](_assets/XGboost/XGBoost.pdf-0005-06.png)
```

## Pipeline

The first version keeps the pipeline intentionally small:

1. Convert the PDF with `pymupdf4llm`, writing PNG assets into `_assets/<pdf_stem>/`.
2. Remove picture-text blocks and standalone page-number lines by default.
3. Apply conservative math cleanup for a few obvious corruption patterns.
4. Detect Markdown tables, compute simple confidence metrics, and mark low-confidence tables.
5. Detect suspicious math-corrupted paragraphs and mark them for review.
6. Normalize image links and write `manifest.json` plus `report.json`.

## Testing

Unit tests cover the deterministic passes:

```bash
source /home/tiaan/environments/pyenv/bin/activate
PYTHONPATH=src pytest
```

For parser-quality checking, use the local `resources/` PDFs as a lightweight manual corpus:

```bash
scimark convert resources/ --out /tmp/scimark-smoke --overwrite
```

Then inspect:

- generated Markdown files in `/tmp/scimark-smoke/`
- `_scimark/report.json` for counts and confidence markers
- `_assets/` to confirm extracted figures/equations are linked correctly

## Known Limitations

- Table handling only scores confidence; it does not repair malformed tables yet.
- Math cleanup is intentionally conservative and does not reconstruct LaTeX.
- Duplicate PDF stems in the same conversion batch are skipped after the first match.
- End-to-end PDF conversion quality still depends on the underlying `pymupdf4llm` extraction.
