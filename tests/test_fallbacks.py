from pathlib import Path

import pymupdf

from scimark.document import StructuralCandidate
from scimark.fallbacks import (
    apply_algorithm_fallbacks,
    render_algorithm_region_fallbacks,
    reorder_algorithm_pages,
)
from scimark.pipeline import PageMarkdown


def test_render_algorithm_region_fallbacks_renders_distinct_crops_and_assigns_paths(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    page = PageMarkdown(
        page_number=1,
        markdown="""**Algorithm 1:** First algorithm

**Algorithm 2:** Second algorithm
""",
        page_boxes=[
            {"index": 0, "class": "caption", "bbox": (50, 100, 250, 120), "pos": (0, 31)},
            {"index": 1, "class": "formula", "bbox": (50, 122, 250, 180), "pos": (31, 32)},
            {"index": 2, "class": "caption", "bbox": (300, 100, 500, 120), "pos": (32, 64)},
            {"index": 3, "class": "formula", "bbox": (300, 122, 500, 180), "pos": (64, 65)},
        ],
    )

    candidates = [
        StructuralCandidate(
            block_id="algorithm-1",
            kind="algorithm",
            start_line=10,
            end_line=12,
            source_page=1,
            needs_fallback=True,
            label="Algorithm 1",
        ),
        StructuralCandidate(
            block_id="algorithm-2",
            kind="algorithm",
            start_line=20,
            end_line=22,
            source_page=1,
            needs_fallback=True,
            label="Algorithm 2",
        ),
    ]

    count = render_algorithm_region_fallbacks(
        pdf_path,
        tmp_path / "fallbacks",
        [page],
        candidates,
        dpi=72,
    )

    assert count == 2
    assert candidates[0].fallback_asset_path is not None
    assert candidates[0].fallback_asset_path != candidates[1].fallback_asset_path
    assert Path(candidates[0].fallback_asset_path).exists()
    assert Path(candidates[1].fallback_asset_path).exists()


def test_apply_algorithm_fallbacks_replaces_noisy_region_with_heading_and_custom_image(
    tmp_path: Path,
) -> None:
    markdown_path = tmp_path / "paper.md"
    fallback_path = tmp_path / "_fallbacks" / "paper" / "page-0003.png"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_bytes(b"png")

    markdown = """Before text.

<!-- scimark: low-confidence-algorithm -->
**Algorithm 2:** Approximate Algorithm for Split Finding **for** _k_ = 1 _**to** m_ **do** Propose candidates.

![](_assets/paper/algorithm-2.png)

After text.
"""
    candidates = [
        StructuralCandidate(
            block_id="algorithm-2",
            kind="algorithm",
            start_line=2,
            end_line=4,
            source_page=3,
            label="Algorithm 2",
            needs_fallback=True,
            fallback_asset_path=str(fallback_path.resolve()),
        )
    ]

    rewritten = apply_algorithm_fallbacks(markdown, candidates, markdown_path)

    assert "<!-- scimark: low-confidence-algorithm -->" not in rewritten
    assert "_assets/paper/algorithm-2.png" not in rewritten
    assert "**Algorithm 2:** Approximate Algorithm for Split Finding" in rewritten
    assert "![](_fallbacks/paper/page-0003.png)" in rewritten


def test_reorder_algorithm_pages_moves_right_column_algorithm_after_later_left_column_text() -> None:
    raw_markdown = """Left intro
More left intro

**Algorithm 1:** Exact Greedy Algorithm for Split Finding

![](_assets/paper/algorithm-1.png)

Left later text
Left end text

## **3.1 Next Section**
"""
    algorithm_start = raw_markdown.index("**Algorithm 1:**")
    image_start = raw_markdown.index("![](_assets/paper/algorithm-1.png)")
    left_later_start = raw_markdown.index("Left later text")
    section_start = raw_markdown.index("## **3.1 Next Section**")

    page = PageMarkdown(
        page_number=3,
        raw_markdown=raw_markdown,
        markdown=raw_markdown,
        page_boxes=[
            {"index": 0, "class": "text", "bbox": (53, 60, 293, 120), "pos": (0, algorithm_start)},
            {
                "index": 1,
                "class": "caption",
                "bbox": (321, 59, 547, 68),
                "pos": (algorithm_start, image_start),
            },
            {
                "index": 2,
                "class": "formula",
                "bbox": (325, 72, 529, 211),
                "pos": (image_start, left_later_start),
            },
            {
                "index": 3,
                "class": "text",
                "bbox": (53, 367, 253, 376),
                "pos": (left_later_start, section_start),
            },
            {
                "index": 4,
                "class": "section-header",
                "bbox": (316, 467, 514, 477),
                "pos": (section_start, len(raw_markdown)),
            },
        ],
    )

    reordered = reorder_algorithm_pages([page])[0]

    assert reordered.markdown.index("Left later text") < reordered.markdown.index("**Algorithm 1:**")
    assert reordered.markdown.index("**Algorithm 1:**") < reordered.markdown.index("## **3.1 Next Section**")
