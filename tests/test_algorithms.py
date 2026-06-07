from scimark.passes.algorithms import annotate_algorithm_blocks, find_algorithm_regions


def test_finds_algorithm_regions_from_explicit_heading_and_body() -> None:
    markdown = """Algorithm 1: Greedy Build
Input: training data
Output: fitted model
for each round do
end

Normal prose paragraph.
"""

    regions = find_algorithm_regions(markdown)

    assert len(regions) == 1
    assert regions[0].label == "Algorithm 1"
    assert (regions[0].start_line, regions[0].end_line) == (0, 4)


def test_finds_bold_algorithm_heading_with_adjacent_image() -> None:
    markdown = """**Algorithm 1:** Exact Greedy Algorithm for Split Finding

![](_assets/paper/algorithm-1.png)

Normal prose paragraph.
"""

    regions = find_algorithm_regions(markdown)

    assert len(regions) == 1
    assert regions[0].label == "Algorithm 1"
    assert (regions[0].start_line, regions[0].end_line) == (0, 2)


def test_finds_inline_bold_algorithm_body_markers() -> None:
    markdown = """**Algorithm 3:** Sparsity-aware Split Finding **Input**: _I_, instance set of current node **Input**: _Ik_ **for** _k_ = 1 **to** m **do**

**end Output**: Split and default directions with max gain
"""

    regions = find_algorithm_regions(markdown)

    assert len(regions) == 1
    assert regions[0].label == "Algorithm 3"
    assert (regions[0].start_line, regions[0].end_line) == (0, 2)


def test_annotates_algorithm_blocks_once() -> None:
    markdown = """Algorithm 2: Beam Search
Input: tokens
Output: sequence
for each step do
end
"""

    annotated, count = annotate_algorithm_blocks(markdown)
    rerun, rerun_count = annotate_algorithm_blocks(annotated)

    assert count == 1
    assert rerun_count == 0
    assert annotated.startswith("<!-- scimark: low-confidence-algorithm -->")
    assert rerun.count("<!-- scimark: low-confidence-algorithm -->") == 1
