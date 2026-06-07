from scimark.passes.confidence import annotate_low_confidence_math
from scimark.passes.math_cleanup import cleanup_math_text


def test_cleanup_math_text() -> None:
    markdown = "The score is 0 _._ 9 and the split is 1 _/_ 4 . Also 5 − 3."

    cleaned = cleanup_math_text(markdown)

    assert "0.9" in cleaned
    assert "1/4" in cleaned
    assert "4." in cleaned
    assert "5 - 3" in cleaned


def test_cleanup_math_text_normalizes_repeatable_parser_tokens() -> None:
    markdown = (
        "let _y_ ˆ _i_[(] _[t][-]_[1)] and _rD_[+] _[,]_[ ˜] _[r] D[-] "
        "with _wj[∗]_[of][leaf] _[j]_[by]"
    )

    cleaned = cleanup_math_text(markdown)

    assert "[(]" not in cleaned
    assert "[)]" not in cleaned
    assert "[+]" not in cleaned
    assert "_[t]" not in cleaned
    assert "_[r]" not in cleaned
    assert "_i_(t-1)" in cleaned
    assert "_rD_+" in cleaned
    assert ",˜ r D-" in cleaned
    assert "leaf" in cleaned


def test_cleanup_math_text_fixes_reciprocal_square_root_artifact() -> None:
    markdown = (
        "Dot-product attention is identical to our algorithm, except for the scaling factor "
        "of 1[Additive attention computes the compatibility function using a feed-forward "
        "network with] ~~_√_~~ _dk_[.] a single hidden layer. "
        "To counteract this effect, we scale the dot products by ~~_√_~~ 1 _dk_[.]"
    )

    cleaned = cleanup_math_text(markdown)

    assert "1[Additive attention computes" not in cleaned
    assert "~~_√_~~" not in cleaned
    assert "1 / √dk." in cleaned
    assert "Additive attention computes the compatibility function using a feed-forward network with a single hidden layer." in cleaned
    assert "we scale the dot products by 1 / √dk." in cleaned


def test_cleanup_math_text_recovers_dot_product_sum_sentence() -> None:
    markdown = (
        "Then their dot product, _q · k_ =[�] d i_ =1 k[q][i][k][i]_[, has mean][ 0][ and variance] _[ d][k]."
    )

    cleaned = cleanup_math_text(markdown)

    assert "[�]" not in cleaned
    assert "[q][i][k][i]" not in cleaned
    assert "[, has mean][ 0][ and variance]" not in cleaned
    assert "_q · k_ = sum_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k." in cleaned


def test_cleanup_math_text_recovers_dot_product_sum_sentence_after_partial_normalization() -> None:
    markdown = (
        "Then their dot product, _q · k_ =[�] d i_ =1 k[q][i][k][i]_[, has mean][ 0][ and variance] _d_k."
    )

    cleaned = cleanup_math_text(markdown)

    assert "_q · k_ = sum_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k." in cleaned


def test_cleanup_math_text_normalizes_matrix_notation() -> None:
    markdown = (
        "Where the projections are parameter matrices _Wi[Q] ∈_ R d[model] _[×][d][k]_, "
        "_Wi[K] ∈_ R d[model] _[×][d][k]_, _Wi[V] ∈_ R d[model] _[×][d][v]_ and "
        "_W[O] ∈_ R hd[v][×][d]model. For each of these we use _d_ model _/h_ = 64."
    )

    cleaned = cleanup_math_text(markdown)

    assert "Wi[Q]" not in cleaned
    assert "d[model]" not in cleaned
    assert "[×][d][k]" not in cleaned
    assert "W_i^Q ∈" in cleaned
    assert "W_i^K ∈" in cleaned
    assert "W_i^V ∈" in cleaned
    assert "W^O ∈" in cleaned
    assert "R d_model × d_k" in cleaned
    assert "R h d_v × d_model" in cleaned
    assert "d_model / h = 64" in cleaned


def test_cleanup_math_text_preserves_paragraph_breaks() -> None:
    markdown = "First paragraph.\n\nWhere the projections are parameter matrices _Wi[Q] ∈_ R d[model] _[×][d][k]_.\n\nSecond paragraph."

    cleaned = cleanup_math_text(markdown)

    assert "First paragraph.\n\n" in cleaned
    assert "\n\nSecond paragraph." in cleaned


def test_annotate_low_confidence_math() -> None:
    markdown = """This paragraph is readable.

The update uses _[ x _D[+] 1 ] with [+] and [−] terms that look broken.
"""

    annotated, count = annotate_low_confidence_math(markdown)

    assert count == 1
    assert "<!-- scimark: low-confidence-math -->" in annotated


def test_annotate_low_confidence_math_only_flags_paragraph_blocks() -> None:
    markdown = """The update uses _[ x _D[+] 1 ] with [+] and [−] terms that look broken.

|expr|value|
|---|---|
|_[x]|[+]|

```text
_[ x _D[+] 1 ] with [+] and [−] terms that look broken.
```
"""

    annotated, count = annotate_low_confidence_math(markdown)

    assert count == 1
    assert annotated.count("<!-- scimark: low-confidence-math -->") == 1
    assert "|_[x]|[+]|" in annotated
    assert "```text" in annotated


def test_annotate_low_confidence_math_does_not_duplicate_existing_comment() -> None:
    markdown = """<!-- scimark: low-confidence-math -->
The update uses _[ x _D[+] 1 ] with [+] and [−] terms that look broken.
"""

    annotated, count = annotate_low_confidence_math(markdown)

    assert count == 0
    assert annotated.count("<!-- scimark: low-confidence-math -->") == 1
