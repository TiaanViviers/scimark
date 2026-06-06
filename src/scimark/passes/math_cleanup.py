from __future__ import annotations

import re


MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "﹣": "-",
        "－": "-",
    }
)

BRACKET_SYMBOL_MAP = {
    "(": "(",
    ")": ")",
    "+": "+",
    "-": "-",
    "−": "-",
    ",": ",",
    ".": ".",
    ":": ":",
    ";": ";",
    "=": "=",
    "≤": "≤",
    "≥": "≥",
    "→": "→",
    "∞": "∞",
    "∈": "∈",
    "˜": "˜",
    "∗": "*",
    "×": "×",
}
LEADING_UNDERSCORE_BRACKET_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z0-9])_\[\s*("
    + "|".join(re.escape(token) for token in BRACKET_SYMBOL_MAP)
    + r")\s*\](?:_(?!\[))?"
)
PLAIN_BRACKET_SYMBOL_RE = re.compile(
    r"\[\s*("
    + "|".join(re.escape(token) for token in BRACKET_SYMBOL_MAP)
    + r")\s*\](?:_(?!\[))?"
)
UNDERSCORE_WRAPPED_TOKEN_RE = re.compile(r"_\[([A-Za-z0-9]{1,6})\]_")
LEADING_UNDERSCORE_TOKEN_RE = re.compile(
    r"_\[([A-Za-z0-9]{1,6})\](?=(?:\s|_|\[|\)|\]|[+\-=/,.;:]))"
)
LEADING_UNDERSCORE_TOKEN_WITH_CLOSE_PAREN_RE = re.compile(r"_\[([A-Za-z0-9]{1,6})\)\]")
INLINE_SQRT_TOKEN_RE = re.compile(r"_\[\s*√\s*\]\s*([A-Za-z][A-Za-z0-9]*)_")
STRIKETHROUGH_SQRT_RE = re.compile(r"~~\s*_?√_?\s*~~")
SIMPLE_SQRT_ITALIC_TOKEN_RE = re.compile(r"√\s*_([A-Za-z][A-Za-z0-9]*)_")
RECIPROCAL_SQRT_WITH_INSERT_RE = re.compile(
    r"1\[(?P<insert>[A-Z][^\]]+)\]\s*√\s*_?(?P<var>[A-Za-z][A-Za-z0-9]*)_?\.\s*(?P<tail>[a-z][^.?!]*[.?!])"
)
RECIPROCAL_SQRT_RE = re.compile(
    r"(?:1\s*√|√\s*1)\s*_?(?P<var>[A-Za-z][A-Za-z0-9]*)_?\."
)
DOT_PRODUCT_SUM_RE = re.compile(
    r"(?P<lhs>_[A-Za-z]\s*·\s*[A-Za-z]_)\s*="
    r"\[�\]\s*(?P<upper_a>[A-Za-z])\s*(?P<idx>[A-Za-z])_?\s*=\s*1\s*"
    r"(?P<upper_b>[A-Za-z])\[(?P<term_a>[A-Za-z])\]\[(?P<idx_a>[A-Za-z])\]\[(?P<term_b>[A-Za-z])\]\[(?P<idx_b>[A-Za-z])\]_?"
    r"(?:\[, has mean\]\[\s*0\]\[ and variance\]\s*|,\s*has mean\s*0\s*and variance\s*)"
    r"(?:(?:_\[\s*(?P<var_a>[A-Za-z])\]\[(?P<var_b>[A-Za-z])\])|(?:_(?P<var_c>[A-Za-z])_(?P<var_d>[A-Za-z])))\."
)
MATRIX_WEIGHT_RE = re.compile(r"(?<![A-Za-z0-9])W([A-Za-z0-9]+)\[([A-Za-z0-9]+)\]")
MATRIX_SUPERSCRIPT_ONLY_RE = re.compile(r"(?<![A-Za-z0-9])W\[([A-Za-z0-9]+)\]")
DIMENSION_SUBSCRIPT_RE = re.compile(r"\bd\[(model|ff|k|v)\]")
BRACKET_DIMENSION_RE = re.compile(r"\[\s*d\s*\]\[\s*(model|ff|k|v)\s*\]")
BRACKET_D_MODEL_RE = re.compile(r"\[\s*d\s*\]model")
HD_SUBSCRIPT_RE = re.compile(r"\bhd\[(model|ff|k|v)\]")
D_MODEL_OVER_H_RE = re.compile(r"_d_\s*model\s*_/h_")


def _normalize_bracket_symbols(markdown: str) -> str:
    cleaned = LEADING_UNDERSCORE_BRACKET_SYMBOL_RE.sub(
        lambda match: BRACKET_SYMBOL_MAP[match.group(1)],
        markdown,
    )
    return PLAIN_BRACKET_SYMBOL_RE.sub(
        lambda match: BRACKET_SYMBOL_MAP[match.group(1)],
        cleaned,
    )


def _normalize_square_root_expressions(markdown: str) -> str:
    cleaned = INLINE_SQRT_TOKEN_RE.sub(lambda match: f"√{match.group(1)}", markdown)
    cleaned = STRIKETHROUGH_SQRT_RE.sub("√", cleaned)
    cleaned = SIMPLE_SQRT_ITALIC_TOKEN_RE.sub(lambda match: f"√{match.group(1)}", cleaned)
    cleaned = RECIPROCAL_SQRT_WITH_INSERT_RE.sub(
        lambda match: (
            f"1 / √{match.group('var')}. "
            f"{match.group('insert')} {match.group('tail')}"
        ),
        cleaned,
    )
    cleaned = RECIPROCAL_SQRT_RE.sub(
        lambda match: f"1 / √{match.group('var')}.",
        cleaned,
    )
    return cleaned


def _normalize_dot_product_sum_sentences(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        lhs = match.group("lhs")
        idx = match.group("idx")
        upper = f"{match.group('upper_a')}_{match.group('upper_b')}"
        term_a = f"{match.group('term_a')}_{match.group('idx_a')}"
        term_b = f"{match.group('term_b')}_{match.group('idx_b')}"
        var_left = match.group("var_a") or match.group("var_c")
        var_right = match.group("var_b") or match.group("var_d")
        variance = f"{var_left}_{var_right}"
        return (
            f"{lhs} = sum_{{{idx}=1}}^{{{upper}}} {term_a} {term_b}, "
            f"has mean 0 and variance {variance}."
        )

    return DOT_PRODUCT_SUM_RE.sub(replace, markdown)


def _normalize_matrix_notation(markdown: str) -> str:
    cleaned = MATRIX_WEIGHT_RE.sub(lambda match: f"W_{match.group(1)}^{match.group(2)}", markdown)
    cleaned = MATRIX_SUPERSCRIPT_ONLY_RE.sub(lambda match: f"W^{match.group(1)}", cleaned)
    cleaned = DIMENSION_SUBSCRIPT_RE.sub(lambda match: f"d_{match.group(1)}", cleaned)
    cleaned = BRACKET_DIMENSION_RE.sub(lambda match: f"d_{match.group(1)}", cleaned)
    cleaned = BRACKET_D_MODEL_RE.sub("d_model", cleaned)
    cleaned = HD_SUBSCRIPT_RE.sub(lambda match: f"h d_{match.group(1)}", cleaned)
    cleaned = D_MODEL_OVER_H_RE.sub("d_model / h", cleaned)
    cleaned = re.sub(r"\s*×\s*", " × ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def cleanup_math_text(markdown: str) -> str:
    cleaned = markdown.translate(MINUS_TRANSLATION)
    cleaned = re.sub(r"(\d)\s+_\._\s+(\d)", r"\1.\2", cleaned)
    cleaned = re.sub(r"(\d)\s+_/_\s+(\d)", r"\1/\2", cleaned)
    cleaned = _normalize_bracket_symbols(cleaned)
    cleaned = _normalize_square_root_expressions(cleaned)
    cleaned = _normalize_dot_product_sum_sentences(cleaned)
    cleaned = _normalize_matrix_notation(cleaned)
    cleaned = LEADING_UNDERSCORE_TOKEN_WITH_CLOSE_PAREN_RE.sub(r"\1)", cleaned)
    cleaned = UNDERSCORE_WRAPPED_TOKEN_RE.sub(r"\1", cleaned)
    cleaned = LEADING_UNDERSCORE_TOKEN_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?)])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned
