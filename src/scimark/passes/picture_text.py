from __future__ import annotations

import re


PICTURE_TEXT_BLOCK_RE = re.compile(
    r"""
    (?:^|\n)
    [ \t]*
    (?:\*\*|__)?-----\ Start\ of\ picture\ text\ -----(?:\*\*|__)?
    [ \t]*(?:<br\s*/?>[ \t]*)?(?:\n|$)?
    .*?
    (?:<br\s*/?>[ \t]*)?
    (?:\*\*|__)?-----\ End\ of\ picture\ text\ -----(?:\*\*|__)?
    [ \t]*(?:<br\s*/?>[ \t]*)?(?=\n|$)
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)


def remove_picture_text_blocks(markdown: str) -> tuple[str, int]:
    count = len(PICTURE_TEXT_BLOCK_RE.findall(markdown))
    cleaned = PICTURE_TEXT_BLOCK_RE.sub("\n", markdown)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n", count
