from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass


Replacement = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True, slots=True)
class RegexRule:
    name: str
    pattern: re.Pattern[str]
    replacement: Replacement

    def apply(self, text: str) -> str:
        return self.pattern.sub(self.replacement, text)


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

BRACKET_SYMBOL_CHOICES = "|".join(re.escape(token) for token in BRACKET_SYMBOL_MAP)

LEADING_UNDERSCORE_BRACKET_SYMBOL_RE = re.compile(
    rf"(?<![A-Za-z0-9])_\[\s*({BRACKET_SYMBOL_CHOICES})\s*\](?:_(?!\[))?"
)
PLAIN_BRACKET_SYMBOL_RE = re.compile(
    rf"\[\s*({BRACKET_SYMBOL_CHOICES})\s*\](?:_(?!\[))?"
)


def _replace_bracket_symbol(match: re.Match[str]) -> str:
    return BRACKET_SYMBOL_MAP[match.group(1)]


TOKEN_NORMALIZATION_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        name="decimal-fragment",
        pattern=re.compile(r"(\d)\s+_\._\s+(\d)"),
        replacement=r"\1.\2",
    ),
    RegexRule(
        name="fraction-fragment",
        pattern=re.compile(r"(\d)\s+_/_\s+(\d)"),
        replacement=r"\1/\2",
    ),
    RegexRule(
        name="leading-underscore-bracket-symbol",
        pattern=LEADING_UNDERSCORE_BRACKET_SYMBOL_RE,
        replacement=_replace_bracket_symbol,
    ),
    RegexRule(
        name="plain-bracket-symbol",
        pattern=PLAIN_BRACKET_SYMBOL_RE,
        replacement=_replace_bracket_symbol,
    ),
)

INLINE_SQRT_TOKEN_RE = re.compile(r"_\[\s*√\s*\]\s*([A-Za-z][A-Za-z0-9]*)_")
STRIKETHROUGH_SQRT_RE = re.compile(r"~~\s*_?√_?\s*~~")
SIMPLE_SQRT_ITALIC_TOKEN_RE = re.compile(r"√\s*_([A-Za-z][A-Za-z0-9]*)_")
RECIPROCAL_SQRT_WITH_INSERT_RE = re.compile(
    r"1\[(?P<insert>[A-Z][^\]]+)\]\s*√\s*_?(?P<var>[A-Za-z][A-Za-z0-9]*)_?\.\s*(?P<tail>[a-z][^.?!]*[.?!])"
)
RECIPROCAL_SQRT_RE = re.compile(
    r"(?:1\s*√|√\s*1)\s*_?(?P<var>[A-Za-z][A-Za-z0-9]*)_?\."
)
MATRIX_WEIGHT_RE = re.compile(r"(?<![A-Za-z0-9])W([A-Za-z0-9]+)\[([A-Za-z0-9]+)\]")
MATRIX_SUPERSCRIPT_ONLY_RE = re.compile(r"(?<![A-Za-z0-9])W\[([A-Za-z0-9]+)\]")
DIMENSION_SUBSCRIPT_RE = re.compile(r"\bd\[(model|ff|k|v)\]")
BRACKET_DIMENSION_RE = re.compile(r"\[\s*d\s*\]\[\s*(model|ff|k|v)\s*\]")
BRACKET_D_MODEL_RE = re.compile(r"\[\s*d\s*\]model")
HD_SUBSCRIPT_RE = re.compile(r"\bhd\[(model|ff|k|v)\]")
D_MODEL_OVER_H_RE = re.compile(r"_d_\s*model\s*_/h_")

NOTATION_NORMALIZATION_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        name="inline-sqrt-token",
        pattern=INLINE_SQRT_TOKEN_RE,
        replacement=lambda match: f"√{match.group(1)}",
    ),
    RegexRule(
        name="strikethrough-sqrt",
        pattern=STRIKETHROUGH_SQRT_RE,
        replacement="√",
    ),
    RegexRule(
        name="sqrt-italic-token",
        pattern=SIMPLE_SQRT_ITALIC_TOKEN_RE,
        replacement=lambda match: f"√{match.group(1)}",
    ),
    RegexRule(
        name="reciprocal-sqrt-with-insert",
        pattern=RECIPROCAL_SQRT_WITH_INSERT_RE,
        replacement=lambda match: (
            f"1 / √{match.group('var')}. "
            f"{match.group('insert')} {match.group('tail')}"
        ),
    ),
    RegexRule(
        name="reciprocal-sqrt",
        pattern=RECIPROCAL_SQRT_RE,
        replacement=lambda match: f"1 / √{match.group('var')}.",
    ),
    RegexRule(
        name="matrix-weight",
        pattern=MATRIX_WEIGHT_RE,
        replacement=lambda match: f"W_{match.group(1)}^{match.group(2)}",
    ),
    RegexRule(
        name="matrix-superscript-only",
        pattern=MATRIX_SUPERSCRIPT_ONLY_RE,
        replacement=lambda match: f"W^{match.group(1)}",
    ),
    RegexRule(
        name="dimension-subscript",
        pattern=DIMENSION_SUBSCRIPT_RE,
        replacement=lambda match: f"d_{match.group(1)}",
    ),
    RegexRule(
        name="bracket-dimension",
        pattern=BRACKET_DIMENSION_RE,
        replacement=lambda match: f"d_{match.group(1)}",
    ),
    RegexRule(
        name="bracket-d-model",
        pattern=BRACKET_D_MODEL_RE,
        replacement="d_model",
    ),
    RegexRule(
        name="hd-subscript",
        pattern=HD_SUBSCRIPT_RE,
        replacement=lambda match: f"h d_{match.group(1)}",
    ),
    RegexRule(
        name="d-model-over-h",
        pattern=D_MODEL_OVER_H_RE,
        replacement="d_model / h",
    ),
    RegexRule(
        name="times-spacing",
        pattern=re.compile(r"\s*×\s*"),
        replacement=" × ",
    ),
    RegexRule(
        name="notation-multi-space",
        pattern=re.compile(r"[ \t]{2,}"),
        replacement=" ",
    ),
)

DOT_PRODUCT_SUM_RE = re.compile(
    r"(?P<lhs>_[A-Za-z]\s*·\s*[A-Za-z]_)\s*="
    r"\[�\]\s*(?P<upper_a>[A-Za-z])\s*(?P<idx>[A-Za-z])_?\s*=\s*1\s*"
    r"(?P<upper_b>[A-Za-z])\[(?P<term_a>[A-Za-z])\]\[(?P<idx_a>[A-Za-z])\]\[(?P<term_b>[A-Za-z])\]\[(?P<idx_b>[A-Za-z])\]_?"
    r"(?:\[, has mean\]\[\s*0\]\[ and variance\]\s*|,\s*has mean\s*0\s*and variance\s*)"
    r"(?:(?:_\[\s*(?P<var_a>[A-Za-z])\]\[(?P<var_b>[A-Za-z])\])|(?:_(?P<var_c>[A-Za-z])_(?P<var_d>[A-Za-z])))\."
)


def _replace_dot_product_sum(match: re.Match[str]) -> str:
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


TEMPLATE_RECOVERY_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        name="dot-product-sum-sentence",
        pattern=DOT_PRODUCT_SUM_RE,
        replacement=_replace_dot_product_sum,
    ),
)

UNDERSCORE_WRAPPED_TOKEN_RE = re.compile(r"_\[([A-Za-z0-9]{1,6})\]_")
LEADING_UNDERSCORE_TOKEN_RE = re.compile(
    r"_\[([A-Za-z0-9]{1,6})\](?=(?:\s|_|\[|\)|\]|[+\-=/,.;:]))"
)
LEADING_UNDERSCORE_TOKEN_WITH_CLOSE_PAREN_RE = re.compile(r"_\[([A-Za-z0-9]{1,6})\)\]")

FINAL_SPACING_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        name="token-with-close-paren",
        pattern=LEADING_UNDERSCORE_TOKEN_WITH_CLOSE_PAREN_RE,
        replacement=r"\1)",
    ),
    RegexRule(
        name="underscore-wrapped-token",
        pattern=UNDERSCORE_WRAPPED_TOKEN_RE,
        replacement=r"\1",
    ),
    RegexRule(
        name="leading-underscore-token",
        pattern=LEADING_UNDERSCORE_TOKEN_RE,
        replacement=r"\1",
    ),
    RegexRule(
        name="space-after-open-paren",
        pattern=re.compile(r"\(\s+"),
        replacement="(",
    ),
    RegexRule(
        name="space-before-close-paren",
        pattern=re.compile(r"\s+\)"),
        replacement=")",
    ),
    RegexRule(
        name="space-before-punctuation",
        pattern=re.compile(r"[ \t]+([,.;:!?)])"),
        replacement=r"\1",
    ),
    RegexRule(
        name="final-multi-space",
        pattern=re.compile(r"[ \t]{2,}"),
        replacement=" ",
    ),
)


def _apply_rules(text: str, rules: Sequence[RegexRule]) -> str:
    current = text
    for rule in rules:
        current = rule.apply(current)
    return current


def _normalize_math_tokens(markdown: str) -> str:
    return _apply_rules(markdown, TOKEN_NORMALIZATION_RULES)


def _normalize_math_notation(markdown: str) -> str:
    return _apply_rules(markdown, NOTATION_NORMALIZATION_RULES)


def _recover_math_templates(markdown: str) -> str:
    return _apply_rules(markdown, TEMPLATE_RECOVERY_RULES)


def _finalize_math_spacing(markdown: str) -> str:
    return _apply_rules(markdown, FINAL_SPACING_RULES)


def cleanup_math_text(markdown: str) -> str:
    cleaned = markdown.translate(MINUS_TRANSLATION)
    cleaned = _normalize_math_tokens(cleaned)
    cleaned = _normalize_math_notation(cleaned)
    cleaned = _recover_math_templates(cleaned)
    cleaned = _finalize_math_spacing(cleaned)
    return cleaned
