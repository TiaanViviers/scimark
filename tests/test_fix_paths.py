from scimark.passes.fix_paths import normalize_image_links


def test_normalize_image_links() -> None:
    markdown = "\n".join(
        [
            "![](parsed/_assets/XGBoost/XGBoost.pdf-0005-06.png)",
            "![](_assets/XGBoost/XGBoost.pdf-0002-11.png)",
            "![](XGBoost.pdf-0003-01.png)",
            "![remote](https://example.com/image.png)",
        ]
    )

    normalized, replacements = normalize_image_links(markdown, "XGBoost")

    assert replacements == 2
    assert "![](_assets/XGBoost/XGBoost.pdf-0005-06.png)" in normalized
    assert "![](_assets/XGBoost/XGBoost.pdf-0002-11.png)" in normalized
    assert "![](_assets/XGBoost/XGBoost.pdf-0003-01.png)" in normalized
    assert "https://example.com/image.png" in normalized
