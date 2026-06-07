from scimark.structure import collect_structural_candidates


def test_collect_structural_candidates_tracks_tables_and_algorithms() -> None:
    markdown = """| Metric | Value |
| --- | --- |
| Accuracy |  |
| Recall | bad<br>split<br>cell |

Algorithm 1: Greedy Build
Input: training data
Output: fitted model
for each round do
end
"""

    candidates = collect_structural_candidates(markdown)

    assert [candidate.kind for candidate in candidates] == ["table", "algorithm"]

    table_candidate = candidates[0]
    algorithm_candidate = candidates[1]

    assert table_candidate.block_id == "table-1"
    assert table_candidate.low_confidence is True
    assert table_candidate.needs_fallback is True
    assert "sparse-stacked-cells" in table_candidate.reasons

    assert algorithm_candidate.block_id == "algorithm-1"
    assert algorithm_candidate.label == "Algorithm 1"
    assert algorithm_candidate.low_confidence is True
    assert algorithm_candidate.needs_fallback is True
    assert "requires-screenshot" in algorithm_candidate.reasons
