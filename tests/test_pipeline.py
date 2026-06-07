from scimark.pipeline import PageMarkdown, PipelineOptions, run_pipeline_pages


def test_run_pipeline_pages_preserves_source_page_and_global_offsets() -> None:
    pages = [
        PageMarkdown(
            page_number=2,
            markdown="""| Metric | Value |
| --- | --- |
| Accuracy |  |
| Recall | bad<br>split<br>cell |
""",
        ),
        PageMarkdown(
            page_number=5,
            markdown="""**Algorithm 1:** Greedy Build
Input: training data
Output: fitted model
for each round do
end
""",
        ),
    ]

    result = run_pipeline_pages(pages, PipelineOptions(pdf_stem="sample"))

    assert result.stats.tables_detected == 1
    assert result.stats.algorithm_blocks_detected == 1
    assert len(result.stats.structural_candidates) == 2

    table_candidate = result.stats.structural_candidates[0]
    algorithm_candidate = result.stats.structural_candidates[1]

    assert table_candidate.kind == "table"
    assert table_candidate.source_page == 2
    assert table_candidate.start_line == 1
    assert table_candidate.needs_fallback is True

    assert algorithm_candidate.kind == "algorithm"
    assert algorithm_candidate.source_page == 5
    assert algorithm_candidate.start_line > table_candidate.end_line
    assert algorithm_candidate.needs_fallback is True
