from __future__ import annotations

import re
from pathlib import PurePosixPath


IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
EXTERNAL_SCHEMES = ("http://", "https://", "data:", "file://")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def normalize_image_links(markdown: str, pdf_stem: str) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements

        alt_text, raw_target = match.groups()
        target = raw_target.strip()
        if target.startswith(EXTERNAL_SCHEMES):
            return match.group(0)

        if " " in target:
            target_path, suffix = target.split(" ", 1)
            trailer = f" {suffix}"
        else:
            target_path = target
            trailer = ""

        cleaned_target = target_path.strip("<>").replace("\\", "/")
        filename = PurePosixPath(cleaned_target).name
        if not filename:
            return match.group(0)

        if PurePosixPath(filename).suffix.lower() not in IMAGE_SUFFIXES:
            return match.group(0)

        normalized_target = f"_assets/{pdf_stem}/{filename}"
        if cleaned_target != normalized_target:
            replacements += 1

        return f"![{alt_text}]({normalized_target}{trailer})"

    return IMAGE_LINK_RE.sub(replace, markdown), replacements
