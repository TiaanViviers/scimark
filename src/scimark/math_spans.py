from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from statistics import median, pstdev

from scimark.layout import (
    RawLayoutBox,
    RawLayoutDocument,
    RawLayoutLine,
    RawLayoutPage,
    RawLayoutSpan,
    reorder_page_boxes_for_reading,
)


MATH_TEXT_RE = re.compile(
    r"(?:[=+\-*/^_]|√|∈|≤|≥|≈|ω|ϵ|λ|γ|Σ|Π|\b(?:Lemma|Theorem|Proof|Definition)\b)"
)
MATH_STATEMENT_RE = re.compile(
    r"(?:"
    r"\b(?:if|then|return|Input|Output|Lemma|Theorem|Proof|Definition|Corollary|Property)\b|"
    r"(?:r|ω|˜r|Q|x∗)[A-Za-z0-9_()′∗,+\-]*|"
    r"Q\([A-Za-z0-9_′]+\)|"
    r"x∗=|"
    r"∈|≤|≥|≈|→"
    r")"
)
AFFILIATION_RE = re.compile(
    r"\b(?:university|department|school|research|institute|laboratory|lab|corporation|company|college)\b",
    re.IGNORECASE,
)
EMAIL_OR_URL_RE = re.compile(r"(?:@|https?://|www\.)", re.IGNORECASE)
CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\b")
STRONG_MATH_RELATION_RE = re.compile(
    r"(?:=|≤|≥|≈|∈|→|√|[+*/]|(?<=[A-Za-z0-9)\]}ωϵλγΣΠα-ωΑ-Ω˜])-(?=[A-Za-z0-9ωϵλγΣΠα-ωΑ-Ω˜]))"
)
BIBLIOGRAPHY_RE = re.compile(
    r"(?:\bet al\.\b|\barXiv\b|\bpp\.\b|\bIn:\b|\bProceedings\b|\bConference\b|\bJournal\b|\bISBN\b|\bDOI\b)",
    re.IGNORECASE,
)
SCRIPT_CLUSTER_RE = re.compile(r"^[A-Za-z0-9α-ωΑ-ΩϵωλγΣΠ_]+(?:[+\-][A-Za-z0-9α-ωΑ-ΩϵωλγΣΠ_]+)*$")
REGION_PROSE_BOX_CLASSES = {"text", "list-item", "section-header"}
FORMULA_PLACEHOLDER = "[[FORMULA_BOX]]"
#
# Generic replacements fix broad parser boundary failures that are likely to
# recur across papers. Corpus replacements are narrower patches observed in the
# current appendix-heavy regression corpus and should be promoted into the
# generic tier only after they show up in multiple papers.
#
GENERIC_SURFACE_REPLACEMENTS = (
    ("definedin", "defined in"),
    ("weredefinedonly", "were defined only"),
    ("satisfiesthefollowing", "satisfies the following"),
    ("wecallitis", "we call it"),
    ("aredefinedtobe", "are defined to be"),
    ("functionThe", "function\nThe"),
    ("function The", "function\nThe"),
    ("asfollows.", "as follows."),
    ("inputD", "input D"),
    ("multi-setD", "multi-set D"),
    ("summaryQ(", "summary Q("),
    ("summaryQ′(", "summary Q′("),
    ("thatQ(", "that Q("),
    ("datasetD", "dataset D"),
    ("forQ′", "for Q′"),
    ("withS", "with S"),
    ("inS", "in S"),
    ("onS", "on S"),
    ("fory", "for y"),
    ("fori", "for i"),
    ("∈X", "∈ X"),
    ("andω", "and ω"),
    ("isan", "is an"),
    ("wehave", "we have"),
    ("meansthe", "means the"),
    ("realworld", "real-world"),
    ("tob+1", "to b+1"),
)
CORPUS_SURFACE_REPLACEMENTS = (
    ("quantilesummaryoftwodataset", "quantile summary of two dataset"),
    ("estimater", "estimate r"),
    ("andr-(", "and r-("),
    ("andr+(", "and r+("),
    ("Notethattherank", "Note that the rank"),
    ("followingproperty", "following property"),
    ("copiedfromoriginalsummary", "copied from original summary"),
    ("approximatesummary", "approximate summary"),
    ("areselectedbyquerytheoriginalsummarysuchthat", "are selected by querying the original summary such that"),
    ("ω˜DtoX", "ω˜D to X"),
    ("ω˜Dare", "ω˜D are"),
    ("˜ωDto", "˜ωD to"),
    ("˜ωDare", "˜ωD are"),
    ("DinQ′", "D in Q′"),
)


@dataclass(slots=True)
class MathRegion:
    page_number: int
    start_box_index: int
    end_box_index: int
    boxes: list[RawLayoutBox] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0 = min(box.bbox[0] for box in self.boxes)
        y0 = min(box.bbox[1] for box in self.boxes)
        x1 = max(box.bbox[2] for box in self.boxes)
        y1 = max(box.bbox[3] for box in self.boxes)
        return x0, y0, x1, y1


@dataclass(slots=True)
class PageSegment:
    page_number: int
    start_box_index: int
    end_box_index: int
    kind: str
    boxes: list[RawLayoutBox] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0 = min(box.bbox[0] for box in self.boxes)
        y0 = min(box.bbox[1] for box in self.boxes)
        x1 = max(box.bbox[2] for box in self.boxes)
        y1 = max(box.bbox[3] for box in self.boxes)
        return x0, y0, x1, y1


class LineKind(str, Enum):
    PROSE = "prose"
    INLINE_MATH_PROSE = "inline_math_prose"
    DISPLAY_MATH = "display_math"
    ALGORITHM = "algorithm"
    THEOREM_PROOF = "theorem_proof"
    CAPTION = "caption"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class LineFeatures:
    text: str
    span_count: int
    word_count: int
    prose_word_count: int
    math_like_token_count: int
    average_token_length: float
    operator_count: int
    greek_count: int
    symbol_density: float
    bracket_underscore_density: float
    script_fraction: float
    centeredness: float
    relative_width: float
    indentation: float
    baseline_variance: float
    font_size_variance: float
    starts_keyword: str | None
    has_equation_number: bool


@dataclass(slots=True)
class LineClassification:
    kind: LineKind
    confidence: float
    reasons: list[str] = field(default_factory=list)
    features: LineFeatures | None = None


class RegionKind(str, Enum):
    PROSE = "prose"
    INLINE_MATH_PROSE = "inline_math_prose"
    DISPLAY_MATH_BLOCK = "display_math_block"
    THEOREM_PROOF_BLOCK = "theorem_proof_block"
    ALGORITHM_BLOCK = "algorithm_block"
    CAPTION = "caption"
    LIST_BLOCK = "list_block"
    UNKNOWN = "unknown"


class RegionSource(str, Enum):
    INITIAL_GROUPING = "initial_grouping"
    EMBEDDED_DISPLAY_RUN = "embedded_display_run"
    FORMULA_BOX_SPLIT = "formula_box_split"


class FormulaBoxKind(str, Enum):
    INLINE_NOISE = "inline_noise"
    DISPLAY_EQUATION = "display_equation"
    CASE_DEFINITION = "case_definition"
    CONSTRAINT_SET = "constraint_set"
    DERIVATION_STEP = "derivation_step"
    ALGORITHM_MATH = "algorithm_math"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class FormulaBoxAnalysis:
    kind: FormulaBoxKind
    usefulness_score: float
    standalone_score: float
    promotion_score: float
    should_standalone: bool
    should_promote: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperMathDiagnostics:
    pages_evaluated: int
    region_count: int
    display_math_regions: int
    promoted_regions: int
    fallback_regions: int
    formula_box_split_regions: int
    embedded_display_run_regions: int
    inline_noise_merged: int
    suspected_figure_label_suppressions: int
    author_affiliation_suppressions: int
    bibliography_entry_suppressions: int
    experimental_candidate_pages: int
    reference_like_pages: int
    figure_table_like_pages: int

    def to_dict(self) -> dict[str, int]:
        return {
            "pages_evaluated": self.pages_evaluated,
            "region_count": self.region_count,
            "display_math_regions": self.display_math_regions,
            "promoted_regions": self.promoted_regions,
            "fallback_regions": self.fallback_regions,
            "formula_box_split_regions": self.formula_box_split_regions,
            "embedded_display_run_regions": self.embedded_display_run_regions,
            "inline_noise_merged": self.inline_noise_merged,
            "suspected_figure_label_suppressions": self.suspected_figure_label_suppressions,
            "author_affiliation_suppressions": self.author_affiliation_suppressions,
            "bibliography_entry_suppressions": self.bibliography_entry_suppressions,
            "experimental_candidate_pages": self.experimental_candidate_pages,
            "reference_like_pages": self.reference_like_pages,
            "figure_table_like_pages": self.figure_table_like_pages,
        }


@dataclass(slots=True)
class PageMathDiagnostics:
    page_number: int
    region_count: int
    display_math_regions: int
    promoted_regions: int
    fallback_regions: int
    formula_box_split_regions: int
    embedded_display_run_regions: int
    caption_regions: int
    suspected_figure_label_suppressions: int
    author_affiliation_suppressions: int
    bibliography_entry_suppressions: int
    experimental_candidate: bool
    reference_like: bool
    figure_table_like: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MathLine:
    text: str
    classification: LineClassification
    bbox: tuple[float, float, float, float]
    source_box_index: int
    box_class: str


@dataclass(slots=True)
class SegmentAnalysis:
    segment: PageSegment
    lines: list[MathLine] = field(default_factory=list)
    base_kind: RegionKind = RegionKind.UNKNOWN
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    has_formula_boxes: bool = False


@dataclass(slots=True)
class RegionBlock:
    page_number: int
    kind: RegionKind
    segments: list[SegmentAnalysis] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    source: RegionSource = RegionSource.INITIAL_GROUPING

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        boxes = [box for segment in self.segments for box in segment.segment.boxes]
        x0 = min(box.bbox[0] for box in boxes)
        y0 = min(box.bbox[1] for box in boxes)
        x1 = max(box.bbox[2] for box in boxes)
        y1 = max(box.bbox[3] for box in boxes)
        return x0, y0, x1, y1


@dataclass(slots=True)
class LineRun:
    start: int
    end: int
    confidence: float
    reasons: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)
    lines: list[MathLine] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return not self.rejected_reasons


def _reference_baseline(spans: list[RawLayoutSpan]) -> float:
    if not spans:
        return 0.0
    max_size = max(span.size for span in spans)
    normal_candidates = [
        span.baseline
        for span in spans
        if not span.is_superscript and span.size >= max_size * 0.95
    ]
    if normal_candidates:
        return median(normal_candidates)
    return median(span.baseline for span in spans)


def _reference_size(spans: list[RawLayoutSpan]) -> float:
    return max((span.size for span in spans if span.size > 0), default=0.0)


def _union_span_bboxes(spans: list[RawLayoutSpan]) -> tuple[float, float, float, float]:
    x0 = min(span.bbox[0] for span in spans)
    y0 = min(span.bbox[1] for span in spans)
    x1 = max(span.bbox[2] for span in spans)
    y1 = max(span.bbox[3] for span in spans)
    return x0, y0, x1, y1


def split_line_by_baseline(line: RawLayoutLine) -> list[RawLayoutLine]:
    spans = sorted(line.spans, key=lambda span: (span.bbox[0], span.bbox[1]))
    if len(spans) < 3:
        return [line]

    max_size = _reference_size(spans)
    baseline_threshold = max(2.0, max_size * 0.45)
    anchors = [
        span
        for span in spans
        if not span.is_superscript and span.size >= max_size * 0.95
    ]
    if len(anchors) < 2:
        return [line]

    anchor_baselines: list[float] = []
    for span in sorted(anchors, key=lambda item: item.baseline):
        if not anchor_baselines or abs(span.baseline - anchor_baselines[-1]) > baseline_threshold:
            anchor_baselines.append(span.baseline)
    if len(anchor_baselines) < 2:
        return [line]

    grouped: dict[int, list[RawLayoutSpan]] = {index: [] for index in range(len(anchor_baselines))}
    for span in spans:
        cluster_index = min(
            range(len(anchor_baselines)),
            key=lambda index: abs(span.baseline - anchor_baselines[index]),
        )
        grouped[cluster_index].append(span)

    virtual_lines = [
        RawLayoutLine(
            bbox=_union_span_bboxes(grouped[index]),
            spans=sorted(grouped[index], key=lambda span: (span.bbox[0], span.bbox[1])),
        )
        for index in range(len(anchor_baselines))
        if grouped[index]
    ]
    if len(virtual_lines) < 2:
        return [line]
    return virtual_lines


def classify_span_role(spans: list[RawLayoutSpan], index: int) -> str:
    span = spans[index]
    if span.is_superscript:
        return "superscript"

    if len(spans) == 1:
        return "normal"

    baseline = _reference_baseline(spans)
    size = _reference_size(spans)
    baseline_delta = span.baseline - baseline
    threshold = max(0.8, size * 0.18)

    if span.size <= size * 0.92 and baseline_delta <= -threshold:
        return "superscript"
    if span.size <= size * 0.92 and baseline_delta >= threshold:
        return "subscript"
    return "normal"


def normalize_span_token(token: str) -> str:
    normalized = token.strip().replace("−", "-")

    previous = None
    while normalized != previous:
        previous = normalized
        normalized = re.sub(r"\[([^\[\]]+)\]", r"\1", normalized)

    normalized = normalized.replace("[", "").replace("]", "")
    normalized = re.sub(r"\s+([,.;:)\]}])", r"\1", normalized)
    normalized = re.sub(r"([(\[{])\s+", r"\1", normalized)
    normalized = re.sub(r"\s*([=+\-*/^_])\s*", r"\1", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _format_script(token: str, prefix: str) -> str:
    if not token:
        return ""
    if len(token) == 1 and re.fullmatch(r"[A-Za-z0-9+\-*/]", token):
        return f"{prefix}{token}"
    return f"{prefix}{{{token}}}"


def _should_insert_space(previous: str, current: str, gap: float, avg_char_width: float) -> bool:
    if not previous:
        return False
    if previous.endswith(("(", "[", "{", "/", "^", "_")):
        return False
    if current.startswith((")", "]", "}", ",", ".", ";", ":", "?", "!", "^", "_", "+", "-", "*", "/")):
        return False
    if previous.endswith(("^", "_")):
        return False
    return gap > max(0.75, avg_char_width * 0.35)


def _is_valid_script_cluster(token: str) -> bool:
    return bool(SCRIPT_CLUSTER_RE.fullmatch(token))


def _count_script_spans(line: RawLayoutLine) -> int:
    return sum(1 for index, _ in enumerate(line.spans) if classify_span_role(line.spans, index) != "normal")


def _count_math_operators(text: str) -> int:
    base = len(re.findall(r"[=+*/^_∈≤≥≈√]", text))
    minus = len(
        re.findall(
            r"(?<=[A-Za-z0-9)\]}ωϵλγΣΠα-ωΑ-Ω˜])-(?=[A-Za-z0-9ωϵλγΣΠα-ωΑ-Ω˜])",
            text,
        )
    )
    return base + minus


def extract_line_features(
    line: RawLayoutLine,
    *,
    serialized_text: str | None = None,
    box_class: str = "text",
    page_width: float | None = None,
    box_bbox: tuple[float, float, float, float] | None = None,
) -> LineFeatures:
    text = (serialized_text if serialized_text is not None else serialize_line(line)).strip()
    nonspace_length = max(len(re.sub(r"\s+", "", text)), 1)
    tokens = [token for token in re.split(r"\s+", text) if token]
    word_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    prose_word_count = sum(1 for token in word_tokens if len(token) >= 3)
    math_like_token_count = sum(
        1
        for token in tokens
        if re.search(r"[0-9=+*/^_∈≤≥≈√ωϵλγΣΠα-ωΑ-Ω]", token)
    )
    operator_count = _count_math_operators(text)
    greek_count = len(re.findall(r"[ωϵλγΣΠα-ωΑ-Ω]", text))
    bracket_underscore_count = len(re.findall(r"[_\[\]\(\)\{\}]", text))
    span_count = len(line.spans)
    script_fraction = _count_script_spans(line) / span_count if span_count else 0.0
    baselines = [span.baseline for span in line.spans]
    sizes = [span.size for span in line.spans]
    baseline_variance = pstdev(baselines) if len(baselines) >= 2 else 0.0
    font_size_variance = pstdev(sizes) if len(sizes) >= 2 else 0.0

    line_width = max(line.bbox[2] - line.bbox[0], 1.0)
    layout_width = page_width or (box_bbox[2] - box_bbox[0] if box_bbox is not None else line_width)
    layout_width = max(layout_width, line_width, 1.0)
    if box_bbox is not None:
        box_width = max(box_bbox[2] - box_bbox[0], 1.0)
        indentation = max(line.bbox[0] - box_bbox[0], 0.0) / box_width
    else:
        indentation = 0.0

    line_center = (line.bbox[0] + line.bbox[2]) / 2.0
    if box_bbox is not None:
        container_center = (box_bbox[0] + box_bbox[2]) / 2.0
        container_width = max(box_bbox[2] - box_bbox[0], 1.0)
    else:
        container_center = layout_width / 2.0
        container_width = layout_width
    centeredness = max(0.0, 1.0 - abs(line_center - container_center) / max(container_width / 2.0, 1.0))

    keyword_match = re.match(
        r"^\s*(Proof|Lemma|Theorem|Definition|Corollary|Property|Algorithm|Input|Output|Figure|Fig\.?|Table|for|if|while|return|else|end|Find)\b",
        text,
    )
    equation_match = re.search(r"\(\d+\)\s*$", text)

    return LineFeatures(
        text=text,
        span_count=span_count,
        word_count=len(word_tokens),
        prose_word_count=prose_word_count,
        math_like_token_count=math_like_token_count,
        average_token_length=(sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0,
        operator_count=operator_count,
        greek_count=greek_count,
        symbol_density=(operator_count + greek_count) / nonspace_length,
        bracket_underscore_density=bracket_underscore_count / nonspace_length,
        script_fraction=script_fraction,
        centeredness=centeredness,
        relative_width=line_width / layout_width,
        indentation=indentation,
        baseline_variance=baseline_variance,
        font_size_variance=font_size_variance,
        starts_keyword=keyword_match.group(1).lower() if keyword_match else None,
        has_equation_number=bool(equation_match),
    )


def classify_line(
    line: RawLayoutLine,
    *,
    serialized_text: str | None = None,
    box_class: str = "text",
    page_width: float | None = None,
    box_bbox: tuple[float, float, float, float] | None = None,
) -> LineClassification:
    features = extract_line_features(
        line,
        serialized_text=serialized_text,
        box_class=box_class,
        page_width=page_width,
        box_bbox=box_bbox,
    )
    reasons: list[str] = []

    if box_class == "formula":
        return LineClassification(
            kind=LineKind.DISPLAY_MATH,
            confidence=0.99,
            reasons=["formula_box"],
            features=features,
        )

    keyword = features.starts_keyword or ""
    if keyword in {"proof", "lemma", "theorem", "definition", "corollary", "property"}:
        reasons.append(f"starts_with_{keyword}")
        if features.operator_count >= 2 or features.script_fraction >= 0.1:
            reasons.append("math_in_statement")
        return LineClassification(
            kind=LineKind.THEOREM_PROOF,
            confidence=0.92,
            reasons=reasons,
            features=features,
        )

    algorithm_heading = bool(re.match(r"^\s*Algorithm(?:\s+\d+)?\b", features.text))
    algorithm_control = keyword in {"input", "output", "for", "if", "while", "return", "else", "end", "find"}
    if algorithm_heading or algorithm_control:
        reasons.append(f"starts_with_{keyword}")
        if features.indentation >= 0.03:
            reasons.append("indented")
        return LineClassification(
            kind=LineKind.ALGORITHM,
            confidence=0.9,
            reasons=reasons,
            features=features,
        )

    if keyword in {"figure", "fig", "table"}:
        return LineClassification(
            kind=LineKind.CAPTION,
            confidence=0.9,
            reasons=[f"starts_with_{keyword}"],
            features=features,
        )

    if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", features.text):
        return LineClassification(
            kind=LineKind.LIST,
            confidence=0.88,
            reasons=["list_marker"],
            features=features,
        )

    frontmatter_reason = _frontmatter_reason(features)
    if frontmatter_reason is not None:
        return LineClassification(
            kind=LineKind.PROSE,
            confidence=0.74,
            reasons=[frontmatter_reason],
            features=features,
        )

    display_score = 0.0
    if features.symbol_density >= 0.14:
        display_score += 1.1
        reasons.append("high_symbol_density")
    if features.operator_count >= 4:
        display_score += 1.0
        reasons.append("many_operators")
    if features.script_fraction >= 0.18:
        display_score += 0.9
        reasons.append("script_heavy")
    if features.prose_word_count <= 3:
        display_score += 0.7
        reasons.append("low_prose_word_count")
    if features.relative_width <= 0.72:
        display_score += 0.6
        reasons.append("narrow_line")
    if features.centeredness >= 0.72:
        display_score += 0.4
        reasons.append("centered")
    if features.baseline_variance >= 1.2:
        display_score += 0.5
        reasons.append("baseline_variance")
    if features.has_equation_number:
        display_score += 0.6
        reasons.append("equation_number")

    if display_score >= 2.3:
        if _looks_like_diagram_label_line(features):
            return LineClassification(
                kind=LineKind.PROSE,
                confidence=0.68,
                reasons=["diagram_label_like"],
                features=features,
            )
        if _has_strong_display_math_signal(features.text, features):
            confidence = min(0.98, 0.55 + display_score / 4.0)
            return LineClassification(
                kind=LineKind.DISPLAY_MATH,
                confidence=confidence,
                reasons=reasons,
                features=features,
            )

    if display_score >= 1.4 and _looks_like_diagram_label_line(features) and features.prose_word_count <= 1:
        return LineClassification(
            kind=LineKind.UNKNOWN,
            confidence=0.45,
            reasons=["diagram_label_like"],
            features=features,
        )

    if (
        features.prose_word_count >= 2
        and (
            features.operator_count >= 2
            or features.script_fraction >= 0.08
            or (features.math_like_token_count >= 2 and features.symbol_density >= 0.05)
        )
    ):
        reasons = ["mixed_prose_math"]
        if features.script_fraction >= 0.08:
            reasons.append("script_spans")
        if features.operator_count >= 2:
            reasons.append("operator_tokens")
        return LineClassification(
            kind=LineKind.INLINE_MATH_PROSE,
            confidence=0.78,
            reasons=reasons,
            features=features,
        )

    if features.prose_word_count >= 1 or features.word_count >= 1:
        return LineClassification(
            kind=LineKind.PROSE,
            confidence=0.8,
            reasons=["default_prose"],
            features=features,
        )

    return LineClassification(
        kind=LineKind.UNKNOWN,
        confidence=0.4,
        reasons=["insufficient_signal"],
        features=features,
    )


def _region_kind_from_line_kind(kind: LineKind) -> RegionKind:
    mapping = {
        LineKind.PROSE: RegionKind.PROSE,
        LineKind.INLINE_MATH_PROSE: RegionKind.INLINE_MATH_PROSE,
        LineKind.DISPLAY_MATH: RegionKind.DISPLAY_MATH_BLOCK,
        LineKind.ALGORITHM: RegionKind.ALGORITHM_BLOCK,
        LineKind.THEOREM_PROOF: RegionKind.THEOREM_PROOF_BLOCK,
        LineKind.CAPTION: RegionKind.CAPTION,
        LineKind.LIST: RegionKind.LIST_BLOCK,
        LineKind.UNKNOWN: RegionKind.UNKNOWN,
    }
    return mapping[kind]


def _line_starts_formula_context(text: str) -> bool:
    return bool(
        re.match(r"^(?:When|If|For|Given|Find)\b", text)
        or text.endswith(":")
        or "constraint" in text.lower()
        or "satisfies" in text.lower()
        or text.lower().startswith("we define")
    )


def _has_strong_display_math_signal(text: str, features: LineFeatures) -> bool:
    if features.has_equation_number:
        return True
    return bool(STRONG_MATH_RELATION_RE.search(text))


def _frontmatter_reason(features: LineFeatures) -> str | None:
    text = features.text.strip()
    if not text:
        return None
    if (
        (re.match(r"^\s*\d+\s+", text) or re.match(r"^\s*\[\d+\]", text))
        and (BIBLIOGRAPHY_RE.search(text) or len(CAPITALIZED_WORD_RE.findall(text)) >= 3)
    ):
        return "bibliography_entry_like"
    if EMAIL_OR_URL_RE.search(text):
        return "author_affiliation_like"
    if AFFILIATION_RE.search(text) and features.operator_count <= 1:
        return "author_affiliation_like"

    capitalized_words = len(CAPITALIZED_WORD_RE.findall(text))
    if (
        capitalized_words >= 2
        and features.word_count <= 8
        and features.prose_word_count >= 2
        and features.operator_count <= 1
        and not text.endswith((".", ":"))
    ):
        return "author_affiliation_like"
    return None


def _looks_like_diagram_label_line(features: LineFeatures) -> bool:
    text = features.text.strip()
    if not text:
        return False
    if _has_strong_display_math_signal(text, features):
        return False
    if text.endswith((".", ":")):
        return False
    if features.word_count == 0 or features.word_count > 4:
        return False
    if features.prose_word_count > 2:
        return False
    if features.operator_count > 1:
        return False
    if features.relative_width > 0.72:
        return False
    if features.script_fraction < 0.08 and not re.search(r"[∆_ωϵλγΣΠ]", text):
        return False
    if features.script_fraction < 0.08 and features.centeredness < 0.55:
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _analyze_segment(segment: PageSegment, *, page_width: float) -> SegmentAnalysis:
    has_formula_boxes = any(box.boxclass == "formula" for box in segment.boxes)

    lines: list[MathLine] = []
    for box_index, box in enumerate(segment.boxes, start=segment.start_box_index):
        if box.boxclass == "formula":
            continue
        for raw_line in box.textlines:
            for virtual_line in split_line_by_baseline(raw_line):
                prose_serialized = serialize_line(virtual_line)
                if not prose_serialized:
                    continue
                classification = classify_line(
                    virtual_line,
                    serialized_text=prose_serialized,
                    box_class=box.boxclass,
                    page_width=page_width,
                    box_bbox=box.bbox,
                )
                rendered = (
                    serialize_display_math_line(virtual_line)
                    if classification.kind == LineKind.DISPLAY_MATH
                    else prose_serialized
                )
                lines.append(
                    MathLine(
                        text=rendered,
                        classification=classification,
                        bbox=virtual_line.bbox,
                        source_box_index=box_index,
                        box_class=box.boxclass,
                    )
                )

    lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    base_kind, confidence, reasons = _classify_segment_lines(lines, has_formula_boxes=has_formula_boxes)

    return SegmentAnalysis(
        segment=segment,
        lines=lines,
        base_kind=base_kind,
        confidence=confidence,
        reasons=reasons,
        has_formula_boxes=has_formula_boxes,
    )


def _normalize_dense_math_prose_line(text: str) -> str:
    normalized = text
    replacements = (
        ("thenreturn", "then return"),
        ("returnx", "return x"),
        ("returnk", "return k"),
        ("returnxi", "return xi"),
        ("returnxi+1", "return xi+1"),
        ("arefunctionsin", "are functions in"),
        ("ω˜Dare", "ω˜D are"),
        ("Q′is", "Q′ is"),
        ("andω", "and ω"),
        ("andx∗", "and x∗"),
        ("Notethattherank", "Note that the rank"),
        ("wehave", "we have"),
        ("ω˜DtoX", "ω˜D to X"),
        ("ω˜Dto X", "ω˜D to X"),
        ("toX", "to X"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"(?<=[A-Za-z0-9\)])\.(?=[A-Z])", ". ", normalized)
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", normalized)
    normalized = re.sub(r"\bS=(?=[A-Za-z0-9])", "S = ", normalized)
    normalized = re.sub(r"\bS =(?=[A-Za-z0-9])", "S = ", normalized)
    normalized = re.sub(r"\bS =\{", "S = {", normalized)
    normalized = re.sub(r"\bD=(?=\{)", "D = ", normalized)
    normalized = re.sub(r"\bD =\{", "D = {", normalized)
    normalized = re.sub(r"Q\(([^)]+)\)=\{", r"Q(\1) = {", normalized)
    normalized = re.sub(r"Q′\(([^)]+)\)=\{", r"Q′(\1) = {", normalized)
    normalized = re.sub(r"\bwith S=\{", "with S = {", normalized)
    normalized = re.sub(r"\bwith S′=\{", "with S′ = {", normalized)
    normalized = re.sub(r"\bwith S=\b", "with S = ", normalized)
    normalized = re.sub(r"\bS=\{", "S = {", normalized)
    normalized = re.sub(r"\bS′=\{", "S′ = {", normalized)
    normalized = re.sub(r"Q′\bis", "Q′ is", normalized)
    normalized = re.sub(r"\bin Q′is\b", "in Q′ is", normalized)
    normalized = re.sub(r"andx∗\s*=", "and x∗=", normalized)
    normalized = re.sub(r"(?<=\bthen return )(?=xi(?:\+1)?)", "", normalized)
    normalized = re.sub(r"(?<=\breturn )(?=[xX][A-Za-z0-9_′∗+-])", "", normalized)
    normalized = re.sub(r"(?<=\breturn)(?=[xX][A-Za-z0-9_′∗+-])", " ", normalized)
    normalized = re.sub(r"x∗=x([0-9A-Za-z_′∗+-]+)", r"x∗ = x\1", normalized)
    normalized = re.sub(r"(?<=x∗ = x[0-9a-zA-Z_′∗+-])\.(?=[A-Z])", ". ", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def _is_math_statement_line(text: str) -> bool:
    if _is_dense_math_text(text):
        return True
    return bool(MATH_STATEMENT_RE.search(text))


def _should_insert_display_math_space(previous: str, current: str, gap: float, avg_char_width: float) -> bool:
    if not previous:
        return False
    if previous.endswith(("(", "[", "{", "/", "^", "_")):
        return False
    if current.startswith((")", "]", "}", ",", ".", ";", ":", "^", "_")):
        return False
    if gap > max(1.2, avg_char_width * 0.7):
        return True
    return False


def _normalize_display_math_line(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\s*([=≤≥≈])\s*", r" \1 ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9)\]}])\s*([+\-])\s*(?=[A-Za-z0-9ωϵλγΣΠα-ωΑ-Ω˜])", r" \1 ", normalized)
    normalized = re.sub(r"\s*([*/])\s*", r" \1 ", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _serialize_line_tokens(line: RawLayoutLine, *, display_math: bool) -> str:
    if not line.spans:
        return ""

    spans = sorted(line.spans, key=lambda span: (span.bbox[0], span.bbox[1]))
    parts: list[str] = []
    previous_span: RawLayoutSpan | None = None
    index = 0

    while index < len(spans):
        span = spans[index]
        token = normalize_span_token(span.text)
        if not token:
            index += 1
            continue

        role = classify_span_role(spans, index)
        gap = 0.0 if previous_span is None else span.bbox[0] - previous_span.bbox[2]
        avg_char_width = (span.bbox[2] - span.bbox[0]) / max(len(token), 1)

        if role == "normal":
            rendered = token
            previous_text = parts[-1] if parts else ""
            if (
                _should_insert_display_math_space(previous_text, rendered, gap, avg_char_width)
                if display_math
                else _should_insert_space(previous_text, rendered, gap, avg_char_width)
            ):
                parts.append(" ")
            parts.append(rendered)
            previous_span = span
            index += 1
            continue

        cluster_spans = [span]
        cluster_tokens = [token]
        next_index = index + 1
        while next_index < len(spans) and classify_span_role(spans, next_index) == role:
            normalized = normalize_span_token(spans[next_index].text)
            if normalized:
                cluster_spans.append(spans[next_index])
                cluster_tokens.append(normalized)
            next_index += 1

        cluster_text = "".join(cluster_tokens)
        if _is_valid_script_cluster(cluster_text):
            rendered = _format_script(cluster_text, "^" if role == "superscript" else "_")
        else:
            rendered = cluster_text

        previous_text = parts[-1] if parts else ""
        if (
            _should_insert_display_math_space(previous_text, rendered, gap, avg_char_width)
            if display_math
            else _should_insert_space(previous_text, rendered, gap, avg_char_width)
        ):
            parts.append(" ")
        parts.append(rendered)
        previous_span = cluster_spans[-1]
        index = next_index

    serialized = "".join(parts)
    serialized = re.sub(r"\s+([,.;:)\]}])", r"\1", serialized)
    serialized = re.sub(r"([(\[{])\s+", r"\1", serialized)
    serialized = re.sub(r"\s{2,}", " ", serialized)
    if display_math:
        serialized = _normalize_display_math_line(serialized)
    elif _is_math_statement_line(serialized):
        serialized = _normalize_dense_math_prose_line(serialized)
    return serialized.strip()


def serialize_line(line: RawLayoutLine) -> str:
    return _serialize_line_tokens(line, display_math=False)


def serialize_display_math_line(line: RawLayoutLine) -> str:
    return _serialize_line_tokens(line, display_math=True)


def serialize_box(box: RawLayoutBox, *, page_width: float | None = None) -> str:
    if box.boxclass == "formula":
        return FORMULA_PLACEHOLDER
    serialized_lines: list[str] = []
    for line in box.textlines:
        for virtual_line in split_line_by_baseline(line):
            prose_serialized = serialize_line(virtual_line)
            classification = classify_line(
                virtual_line,
                serialized_text=prose_serialized,
                box_class=box.boxclass,
                page_width=page_width,
                box_bbox=box.bbox,
            )
            serialized = (
                serialize_display_math_line(virtual_line)
                if classification.kind == LineKind.DISPLAY_MATH
                else prose_serialized
            )
            if serialized:
                serialized_lines.append(serialized)
    return "\n".join(serialized_lines).strip()


def _boxes_overlap_horizontally(left: RawLayoutBox, right: RawLayoutBox) -> bool:
    return min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]) > 0


def _box_gap(previous: RawLayoutBox, current: RawLayoutBox) -> float:
    return current.bbox[1] - previous.bbox[3]


def _is_formula_stub_text(text: str) -> bool:
    return text == FORMULA_PLACEHOLDER or text.startswith("[[DISPLAY_FORMULA")


def _is_standalone_equation_number_text(text: str) -> bool:
    return bool(re.fullmatch(r"\(\d+\)", text.strip()))


def _ends_with_numbered_formula_stub(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return bool(re.fullmatch(r"\[\[DISPLAY_FORMULA: numbered-equation \(\d+\)\]\]", lines[-1]))


def _is_dense_math_text(text: str) -> bool:
    operators = len(re.findall(r"[=+\-*/^_∈≤≥≈√]", text))
    math_words = len(re.findall(r"\b(?:sum|prod|arg|max|min|Lemma|Theorem|Proof|Definition)\b", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if _is_formula_stub_text(text):
        return True
    if operators >= 4:
        return True
    if letters and operators / letters >= 0.18:
        return True
    return math_words > 0 and operators >= 2


def _starts_heading_like(text: str) -> bool:
    return bool(re.match(r"^(?:Lemma|Theorem|Definition|Proof|Corollary|Property)\b", text))


def _choose_box_separator(previous_text: str, current_text: str, *, gap: float) -> str:
    if _is_formula_stub_text(previous_text) or _is_formula_stub_text(current_text):
        return "\n"
    if previous_text.endswith("-"):
        return "\n"
    if gap > 14:
        return "\n"
    if _is_dense_math_text(previous_text) or _is_dense_math_text(current_text):
        return "\n"
    if previous_text.endswith((".", "?", "!", ":")):
        return "\n"
    if _starts_heading_like(current_text):
        return "\n"
    return " "


def _stitch_region_text(serialized: str) -> str:
    lines = [line.strip() for line in serialized.splitlines() if line.strip()]
    if not lines:
        return ""

    stitched = [lines[0]]
    for line in lines[1:]:
        previous = stitched[-1]
        if _is_formula_stub_text(previous) or _is_formula_stub_text(line):
            stitched.append(line)
            continue
        if previous.endswith("-") and line[:1].islower():
            stitched[-1] = previous[:-1] + line
            continue
        if (
            not previous.endswith((".", "?", "!", ":"))
            and line[:1].islower()
            and not _is_dense_math_text(previous)
            and not _is_dense_math_text(line)
        ):
            stitched[-1] = previous + " " + line
            continue
        stitched.append(line)
    return "\n".join(stitched)


def _is_region_break_box(box: RawLayoutBox) -> bool:
    if box.boxclass == "section-header":
        return True
    if box.boxclass not in REGION_PROSE_BOX_CLASSES or not box.textlines:
        return False
    text = serialize_box(box)
    if not text:
        return False
    return bool(re.match(r"^(?:Lemma|Theorem|Definition|Proof|Corollary|Property)\b", text))


def _should_extend_region(previous: RawLayoutBox, current: RawLayoutBox) -> bool:
    gap = _box_gap(previous, current)
    if gap > 32:
        return False
    if _boxes_overlap_horizontally(previous, current):
        return True
    return gap <= 12


def _should_extend_prose_cluster(previous: RawLayoutBox, current: RawLayoutBox) -> bool:
    gap = _box_gap(previous, current)
    if gap > 28:
        return False
    if _boxes_overlap_horizontally(previous, current):
        return True

    previous_x0 = previous.bbox[0]
    current_x0 = current.bbox[0]
    return gap <= 8 and abs(previous_x0 - current_x0) <= 24


def find_math_regions(page: RawLayoutPage) -> list[MathRegion]:
    regions: list[MathRegion] = []
    current_boxes: list[tuple[int, RawLayoutBox]] = []

    for index, box in enumerate(page.boxes):
        if not is_math_like_box(box):
            if current_boxes and _is_region_break_box(box):
                start_index = current_boxes[0][0]
                end_index = current_boxes[-1][0]
                regions.append(
                    MathRegion(
                        page_number=page.page_number,
                        start_box_index=start_index,
                        end_box_index=end_index,
                        boxes=[candidate for _, candidate in current_boxes],
                    )
                )
                current_boxes = []
            continue

        if not current_boxes:
            current_boxes = [(index, box)]
            continue

        previous_box = current_boxes[-1][1]
        if _should_extend_region(previous_box, box):
            current_boxes.append((index, box))
            continue

        start_index = current_boxes[0][0]
        end_index = current_boxes[-1][0]
        regions.append(
            MathRegion(
                page_number=page.page_number,
                start_box_index=start_index,
                end_box_index=end_index,
                boxes=[candidate for _, candidate in current_boxes],
            )
        )
        current_boxes = [(index, box)]

    if current_boxes:
        start_index = current_boxes[0][0]
        end_index = current_boxes[-1][0]
        regions.append(
            MathRegion(
                page_number=page.page_number,
                start_box_index=start_index,
                end_box_index=end_index,
                boxes=[candidate for _, candidate in current_boxes],
            )
        )

    return regions


def serialize_region(region: MathRegion, *, page_width: float | None = None) -> str:
    parts: list[str] = []
    previous_box: RawLayoutBox | None = None
    previous_text = ""

    for box in region.boxes:
        text = serialize_box(box, page_width=page_width)
        if not text:
            continue

        if previous_box is not None:
            gap = _box_gap(previous_box, box)
            parts.append(_choose_box_separator(previous_text, text, gap=gap))

        parts.append(text)
        previous_box = box
        previous_text = text

    serialized = "".join(parts)
    serialized = re.sub(r" *\n *", "\n", serialized)
    serialized = re.sub(r"\n{3,}", "\n\n", serialized)
    serialized = re.sub(r"[ \t]{2,}", " ", serialized)
    return _stitch_region_text(serialized).strip()


def _serialize_prose_boxes(boxes: list[RawLayoutBox], *, page_width: float | None = None) -> str:
    parts: list[str] = []
    previous_box: RawLayoutBox | None = None
    previous_text = ""

    for box in boxes:
        text = serialize_box(box, page_width=page_width)
        if not text:
            continue

        if previous_box is not None:
            gap = _box_gap(previous_box, box)
            parts.append(_choose_box_separator(previous_text, text, gap=gap))

        parts.append(text)
        previous_box = box
        previous_text = text

    serialized = "".join(parts)
    serialized = re.sub(r" *\n *", "\n", serialized)
    serialized = re.sub(r"\n{3,}", "\n\n", serialized)
    serialized = re.sub(r"[ \t]{2,}", " ", serialized)
    return _stitch_region_text(serialized).strip()


def is_math_like_box(box: RawLayoutBox) -> bool:
    if box.boxclass == "formula":
        return True
    if not box.textlines:
        return False

    text = serialize_box(box)
    if MATH_TEXT_RE.search(text):
        return True

    script_spans = 0
    total_spans = 0
    for line in box.textlines:
        for index, _ in enumerate(line.spans):
            total_spans += 1
            if classify_span_role(line.spans, index) != "normal":
                script_spans += 1

    return total_spans > 0 and script_spans / total_spans >= 0.2


def _format_bbox(bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    return f"({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})"


def _normalize_surface_text(text: str) -> str:
    normalized = text
    for source, target in GENERIC_SURFACE_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    for source, target in CORPUS_SURFACE_REPLACEMENTS:
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"(?<=\))(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z]\()", " ", normalized)
    normalized = re.sub(r"(?<=[a-z0-9])∈(?=[A-Za-z0-9])", " ∈ ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9_])→(?=[A-Za-z0-9])", " → ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])=(?=[A-Za-z])", " = ", normalized)
    normalized = re.sub(r"(?<=[\w\)]),(?=\w)", ", ", normalized)
    normalized = re.sub(r"Eq\.\(", "Eq. (", normalized)
    normalized = re.sub(r"\bQ′is\b", "Q′ is", normalized)
    normalized = re.sub(r"\bin Q′is\b", "in Q′ is", normalized)
    normalized = re.sub(r"\bQ\(([^)]+)\)=\(", r"Q(\1) = (", normalized)
    normalized = re.sub(r"\bQ′\(([^)]+)\)=\(", r"Q′(\1) = (", normalized)
    normalized = re.sub(r"\bQ\(([^)]+)\)=\{", r"Q(\1) = {", normalized)
    normalized = re.sub(r"\bQ′\(([^)]+)\)=\{", r"Q′(\1) = {", normalized)
    normalized = re.sub(r"\bS =\{", "S = {", normalized)
    normalized = re.sub(r"\bD=(?=\{)", "D = ", normalized)
    normalized = re.sub(r"\bD =\{", "D = {", normalized)
    normalized = re.sub(r"\bwith S=\{", "with S = {", normalized)
    normalized = re.sub(r"\bwith S′=\{", "with S′ = {", normalized)
    normalized = re.sub(r"\bS=\{", "S = {", normalized)
    normalized = re.sub(r"\bS′=\{", "S′ = {", normalized)
    normalized = re.sub(r"ω˜Dto X", "ω˜D to X", normalized)
    normalized = re.sub(r"\bx∗=g\(", "x∗ = g(", normalized)
    normalized = re.sub(r"x∗=x([0-9A-Za-z_′∗+-]+)", r"x∗ = x\1", normalized)
    normalized = re.sub(r"andx∗\s*=", "and x∗=", normalized)
    normalized = re.sub(r"\bthenreturn", "then return", normalized)
    normalized = re.sub(r"\belse return", "else return", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9_)])(?=(?:were|satisfies|means|copied)\b)", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized.strip()


def _segment_context_text(segment: PageSegment, *, page_width: float | None = None) -> str:
    if segment.kind == "prose":
        return _serialize_prose_boxes(segment.boxes, page_width=page_width)

    prose_parts = [serialize_box(box, page_width=page_width) for box in segment.boxes if box.boxclass != "formula"]
    prose_parts = [part for part in prose_parts if part]
    return "\n".join(prose_parts).strip()


def _context_edge_line(text: str, *, last: bool) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1] if last else lines[0]


def _render_formula_group_stub(
    boxes: list[RawLayoutBox],
    *,
    before_text: str,
    after_text: str,
) -> str:
    before_line = _context_edge_line(before_text, last=True)
    after_line = _context_edge_line(after_text, last=False)
    before_lower = before_line.lower()
    after_lower = after_line.lower()

    equation_number = re.fullmatch(r"\((\d+)\)", after_line)
    if equation_number:
        kind = f"numbered-equation ({equation_number.group(1)})"
    elif before_lower.startswith("when ") or "for some i" in before_lower:
        kind = "case-definition"
    elif any(token in before_lower for token in ("input:", "output:", "return", "find i")) or any(
        token in after_lower for token in ("then return", "else return")
    ):
        kind = "algorithm-step"
    elif any(token in before_lower for token in ("constraint", "satisfies", "satisfy")) or after_lower.startswith("for all"):
        kind = "constraint"
    elif len(boxes) > 1 or max((box.bbox[3] - box.bbox[1] for box in boxes), default=0.0) >= 28.0:
        kind = "derivation"
    else:
        kind = "equation"

    if len(boxes) == 1:
        return f"[[DISPLAY_FORMULA: {kind}]]"
    return f"[[DISPLAY_FORMULA_BLOCK x{len(boxes)}: {kind}]]"


def _serialize_math_region_for_page(
    region: MathRegion,
    *,
    previous_context: str,
    next_context: str,
    page_width: float | None = None,
) -> str:
    parts: list[str] = []
    previous_box: RawLayoutBox | None = None
    previous_text = ""
    box_index = 0

    while box_index < len(region.boxes):
        box = region.boxes[box_index]
        if box.boxclass == "formula":
            group = [box]
            lookahead = box_index + 1
            while lookahead < len(region.boxes) and region.boxes[lookahead].boxclass == "formula":
                group.append(region.boxes[lookahead])
                lookahead += 1

            before_text = previous_text or previous_context
            after_text = next_context
            for later_box in region.boxes[lookahead:]:
                if later_box.boxclass != "formula":
                    after_text = serialize_box(later_box, page_width=page_width)
                    break

            rendered = _render_formula_group_stub(group, before_text=before_text, after_text=after_text)
            if previous_box is not None:
                gap = _box_gap(previous_box, group[0])
                parts.append(_choose_box_separator(previous_text, rendered, gap=gap))
            parts.append(rendered)
            previous_box = group[-1]
            previous_text = rendered
            box_index = lookahead
            continue

        rendered = serialize_box(box, page_width=page_width)
        if rendered:
            if previous_box is not None:
                gap = _box_gap(previous_box, box)
                parts.append(_choose_box_separator(previous_text, rendered, gap=gap))
            parts.append(rendered)
            previous_box = box
            previous_text = rendered
        box_index += 1

    serialized = "".join(parts)
    serialized = re.sub(r" *\n *", "\n", serialized)
    serialized = re.sub(r"\n{3,}", "\n\n", serialized)
    serialized = re.sub(r"[ \t]{2,}", " ", serialized)
    return _stitch_region_text(serialized).strip()


def build_page_segments(page: RawLayoutPage) -> list[PageSegment]:
    ordered_page = reorder_page_boxes_for_reading(page)
    segments: list[PageSegment] = []
    index = 0

    while index < len(ordered_page.boxes):
        box = ordered_page.boxes[index]
        if is_math_like_box(box):
            start = index
            current_boxes = [box]
            index += 1
            while index < len(ordered_page.boxes):
                candidate = ordered_page.boxes[index]
                if not is_math_like_box(candidate):
                    break
                if not _should_extend_region(current_boxes[-1], candidate):
                    break
                current_boxes.append(candidate)
                index += 1
            segments.append(
                PageSegment(
                    page_number=ordered_page.page_number,
                    start_box_index=start,
                    end_box_index=start + len(current_boxes) - 1,
                    kind="math_region",
                    boxes=current_boxes,
                )
            )
            continue

        start = index
        current_boxes = [box]
        index += 1
        while index < len(ordered_page.boxes):
            candidate = ordered_page.boxes[index]
            if is_math_like_box(candidate):
                break
            if current_boxes[-1].boxclass == "section-header":
                break
            if _is_region_break_box(candidate):
                break
            if not _should_extend_prose_cluster(current_boxes[-1], candidate):
                break
            current_boxes.append(candidate)
            index += 1

        segments.append(
            PageSegment(
                page_number=ordered_page.page_number,
                start_box_index=start,
                end_box_index=start + len(current_boxes) - 1,
                kind="prose",
                boxes=current_boxes,
            )
        )

    return segments


def _can_absorb_into_theorem_block(candidate: SegmentAnalysis) -> bool:
    return candidate.base_kind in {
        RegionKind.PROSE,
        RegionKind.INLINE_MATH_PROSE,
        RegionKind.DISPLAY_MATH_BLOCK,
        RegionKind.THEOREM_PROOF_BLOCK,
    }


def _merge_prose_kind(left: RegionKind, right: RegionKind) -> RegionKind:
    if RegionKind.INLINE_MATH_PROSE in {left, right}:
        return RegionKind.INLINE_MATH_PROSE
    return RegionKind.PROSE


def _segment_ends_formula_preface(segment: SegmentAnalysis) -> bool:
    if not segment.lines:
        return False
    return _line_starts_formula_context(segment.lines[-1].text)


def _build_region_block(
    kind: RegionKind,
    segments: list[SegmentAnalysis],
    *,
    page_number: int,
    source: RegionSource = RegionSource.INITIAL_GROUPING,
) -> RegionBlock:
    confidence = sum(segment.confidence for segment in segments) / len(segments) if segments else 0.0
    reasons: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        for reason in segment.reasons:
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)
    return RegionBlock(
        page_number=page_number,
        kind=kind,
        segments=segments,
        confidence=confidence,
        reasons=reasons,
        source=source,
    )


def _kind_for_line_chunk(lines: list[MathLine], fallback: RegionKind) -> RegionKind:
    if not lines:
        return fallback
    kinds = {line.classification.kind for line in lines}
    if kinds == {LineKind.DISPLAY_MATH}:
        return RegionKind.DISPLAY_MATH_BLOCK
    if LineKind.THEOREM_PROOF in kinds:
        return RegionKind.THEOREM_PROOF_BLOCK
    if LineKind.INLINE_MATH_PROSE in kinds:
        return RegionKind.INLINE_MATH_PROSE
    if kinds == {LineKind.PROSE}:
        return RegionKind.PROSE
    return fallback


def _classify_segment_lines(lines: list[MathLine], *, has_formula_boxes: bool) -> tuple[RegionKind, float, list[str]]:
    if not lines:
        if has_formula_boxes:
            return RegionKind.DISPLAY_MATH_BLOCK, 0.96, ["formula_boxes"]
        return RegionKind.UNKNOWN, 0.0, ["empty"]

    first_kind = lines[0].classification.kind
    first_text = lines[0].text
    if has_formula_boxes and (first_kind == LineKind.THEOREM_PROOF or _line_starts_formula_context(first_text)):
        base_kind = RegionKind.THEOREM_PROOF_BLOCK
    elif first_kind == LineKind.THEOREM_PROOF:
        base_kind = RegionKind.THEOREM_PROOF_BLOCK
    elif all(line.classification.kind == LineKind.ALGORITHM for line in lines):
        base_kind = RegionKind.ALGORITHM_BLOCK
    elif all(line.classification.kind == LineKind.LIST for line in lines):
        base_kind = RegionKind.LIST_BLOCK
    elif lines[0].classification.kind == LineKind.CAPTION:
        base_kind = RegionKind.CAPTION
    elif all(line.classification.kind == LineKind.DISPLAY_MATH for line in lines):
        base_kind = RegionKind.DISPLAY_MATH_BLOCK
    elif any(line.classification.kind == LineKind.INLINE_MATH_PROSE for line in lines):
        base_kind = RegionKind.INLINE_MATH_PROSE
    elif all(line.classification.kind == LineKind.PROSE for line in lines):
        base_kind = RegionKind.PROSE
    else:
        base_kind = _region_kind_from_line_kind(first_kind)

    confidence = sum(line.classification.confidence for line in lines) / len(lines)
    reasons: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for reason in line.classification.reasons:
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)
    return base_kind, confidence, reasons


def _build_child_segment_analysis(
    parent: SegmentAnalysis,
    *,
    boxes: list[RawLayoutBox],
    lines: list[MathLine],
    start_box_index: int,
    end_box_index: int,
    has_formula_boxes: bool,
) -> SegmentAnalysis:
    base_kind, confidence, reasons = _classify_segment_lines(lines, has_formula_boxes=has_formula_boxes)
    return SegmentAnalysis(
        segment=PageSegment(
            page_number=parent.segment.page_number,
            start_box_index=start_box_index,
            end_box_index=end_box_index,
            kind=parent.segment.kind,
            boxes=boxes,
        ),
        lines=lines,
        base_kind=base_kind,
        confidence=confidence,
        reasons=reasons,
        has_formula_boxes=has_formula_boxes,
    )


def _split_segment_into_box_analyses(segment: SegmentAnalysis) -> list[SegmentAnalysis]:
    if len(segment.segment.boxes) <= 1:
        return [segment]

    child_analyses: list[SegmentAnalysis] = []
    local_box_index = 0
    while local_box_index < len(segment.segment.boxes):
        box = segment.segment.boxes[local_box_index]
        if box.boxclass == "formula":
            formula_boxes = [box]
            start_local_index = local_box_index
            local_box_index += 1
            while local_box_index < len(segment.segment.boxes):
                candidate = segment.segment.boxes[local_box_index]
                if candidate.boxclass != "formula":
                    break
                formula_boxes.append(candidate)
                local_box_index += 1
            start_box_index = segment.segment.start_box_index + start_local_index
            end_box_index = start_box_index + len(formula_boxes) - 1
            child_analyses.append(
                _build_child_segment_analysis(
                    segment,
                    boxes=formula_boxes,
                    lines=[],
                    start_box_index=start_box_index,
                    end_box_index=end_box_index,
                    has_formula_boxes=True,
                )
            )
            continue

        child_lines = [line for line in segment.lines if line.source_box_index == local_box_index]
        if child_lines:
            start_box_index = segment.segment.start_box_index + local_box_index
            child_analyses.append(
                _build_child_segment_analysis(
                    segment,
                    boxes=[box],
                    lines=child_lines,
                    start_box_index=start_box_index,
                    end_box_index=start_box_index,
                    has_formula_boxes=False,
                )
            )
        local_box_index += 1
    return child_analyses or [segment]


def _is_sentence_fragment_display_line(line: MathLine) -> bool:
    features = line.classification.features
    if features is None:
        return False
    if re.search(r"\b(?:to|and|or|the|is|for|when|where|which means|we have|such)\b", line.text):
        return True
    if line.text.endswith("."):
        return True
    if features.prose_word_count >= 3:
        return True
    return False


def _evaluate_display_run(lines: list[MathLine]) -> LineRun:
    confidence = sum(line.classification.confidence for line in lines) / len(lines)
    reasons: list[str] = []
    seen: set[str] = set()
    rejected_reasons: list[str] = []

    for line in lines:
        for reason in line.classification.reasons:
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)

    if any(line.classification.kind != LineKind.DISPLAY_MATH for line in lines):
        rejected_reasons.append("non_display_math_line")
    if any(_is_sentence_fragment_display_line(line) for line in lines):
        rejected_reasons.append("sentence_fragment")
    if len(lines) == 1 and confidence < 0.92:
        rejected_reasons.append("single_line_confidence_below_threshold")
    if len(lines) > 1 and confidence < 0.82:
        rejected_reasons.append("run_confidence_below_threshold")

    return LineRun(
        start=0,
        end=len(lines) - 1,
        confidence=confidence,
        reasons=reasons,
        rejected_reasons=rejected_reasons,
        lines=lines,
    )


def _find_display_math_runs_in_analysis(segment: SegmentAnalysis) -> list[LineRun]:
    if segment.has_formula_boxes or not segment.lines:
        return []

    runs: list[LineRun] = []
    current_lines: list[MathLine] = []
    current_start = 0

    def flush(end_index: int) -> None:
        nonlocal current_lines, current_start
        if not current_lines:
            return
        run = _evaluate_display_run(current_lines)
        run.start = current_start
        run.end = end_index
        runs.append(run)
        current_lines = []

    for index, line in enumerate(segment.lines):
        is_candidate = line.classification.kind in {LineKind.DISPLAY_MATH, LineKind.INLINE_MATH_PROSE}
        if not is_candidate:
            flush(index - 1)
            continue
        if not current_lines:
            current_start = index
            current_lines = [line]
            continue
        previous = current_lines[-1]
        if index == current_start + len(current_lines) and abs(line.bbox[1] - previous.bbox[3]) <= 18:
            current_lines.append(line)
            continue
        flush(index - 1)
        current_start = index
        current_lines = [line]

    flush(len(segment.lines) - 1)
    return runs


def find_embedded_display_math_runs(segment: SegmentAnalysis) -> list[LineRun]:
    runs: list[LineRun] = []
    line_offset = 0
    for child in _split_segment_into_box_analyses(segment):
        if child.has_formula_boxes:
            line_offset += len(child.lines)
            continue
        for run in _find_display_math_runs_in_analysis(child):
            run.start += line_offset
            run.end += line_offset
            runs.append(run)
        line_offset += len(child.lines)
    return runs


def _split_segment_by_display_math_runs(
    segment: SegmentAnalysis,
    *,
    parent_kind: RegionKind,
) -> list[SegmentAnalysis]:
    if segment.has_formula_boxes or not segment.lines:
        return [segment]
    runs = [run for run in find_embedded_display_math_runs(segment) if run.accepted]
    if not runs:
        return [segment]

    split_segments: list[SegmentAnalysis] = []
    cursor = 0
    for run in runs:
        if run.start > cursor:
            left_lines = segment.lines[cursor:run.start]
            split_segments.append(
                SegmentAnalysis(
                    segment=segment.segment,
                    lines=left_lines,
                    base_kind=_kind_for_line_chunk(left_lines, parent_kind),
                    confidence=sum(line.classification.confidence for line in left_lines) / len(left_lines),
                    reasons=list({reason for line in left_lines for reason in line.classification.reasons}),
                    has_formula_boxes=False,
                )
            )
        run_lines = segment.lines[run.start : run.end + 1]
        split_segments.append(
            SegmentAnalysis(
                segment=segment.segment,
                lines=run_lines,
                base_kind=RegionKind.DISPLAY_MATH_BLOCK,
                confidence=run.confidence,
                reasons=run.reasons,
                has_formula_boxes=False,
            )
        )
        cursor = run.end + 1

    if cursor < len(segment.lines):
        right_lines = segment.lines[cursor:]
        split_segments.append(
            SegmentAnalysis(
                segment=segment.segment,
                lines=right_lines,
                base_kind=_kind_for_line_chunk(right_lines, parent_kind),
                confidence=sum(line.classification.confidence for line in right_lines) / len(right_lines),
                reasons=list({reason for line in right_lines for reason in line.classification.reasons}),
                has_formula_boxes=False,
            )
        )

    return [split for split in split_segments if split.lines] or [segment]


def split_embedded_display_math_regions(regions: list[RegionBlock]) -> list[RegionBlock]:
    split_regions: list[RegionBlock] = []

    for region in regions:
        if region.kind not in {RegionKind.THEOREM_PROOF_BLOCK, RegionKind.PROSE, RegionKind.INLINE_MATH_PROSE}:
            split_regions.append(region)
            continue

        pending: list[SegmentAnalysis] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            pending_kind = region.kind
            if pending_kind not in {RegionKind.THEOREM_PROOF_BLOCK, RegionKind.PROSE, RegionKind.INLINE_MATH_PROSE}:
                pending_kind = RegionKind.PROSE
            split_regions.append(
                _build_region_block(
                    pending_kind,
                    pending,
                    page_number=region.page_number,
                    source=region.source,
                )
            )
            pending = []

        for segment in region.segments:
            for box_analysis in _split_segment_into_box_analyses(segment):
                if any(line.classification.kind == LineKind.CAPTION for line in box_analysis.lines):
                    pending.append(box_analysis)
                    continue
                for chunk in _split_segment_by_display_math_runs(box_analysis, parent_kind=region.kind):
                    if chunk.base_kind == RegionKind.DISPLAY_MATH_BLOCK:
                        flush_pending()
                        source = (
                            RegionSource.FORMULA_BOX_SPLIT
                            if chunk.has_formula_boxes
                            else RegionSource.EMBEDDED_DISPLAY_RUN
                        )
                        split_regions.append(
                            _build_region_block(
                                RegionKind.DISPLAY_MATH_BLOCK,
                                [chunk],
                                page_number=region.page_number,
                                source=source,
                            )
                        )
                    else:
                        pending.append(chunk)

        flush_pending()

    return split_regions


def _formula_box_stub_kind(
    region: RegionBlock,
    *,
    previous_context: str,
    next_context: str,
) -> str | None:
    if not all(segment.has_formula_boxes for segment in region.segments):
        return None
    boxes = [box for segment in region.segments for box in segment.segment.boxes if box.boxclass == "formula"]
    if not boxes:
        return None
    rendered = _render_formula_group_stub(boxes, before_text=previous_context, after_text=next_context)
    match = re.fullmatch(r"\[\[DISPLAY_FORMULA(?:_BLOCK x\d+)?: ([^\]]+)\]\]", rendered)
    if not match:
        return None
    return match.group(1)


def analyze_formula_box_region(
    region: RegionBlock,
    *,
    page_width: float,
    previous_context: str,
    next_context: str,
    previous_region_kind: RegionKind | None = None,
    next_region_kind: RegionKind | None = None,
    previous_region_source: RegionSource | None = None,
    next_region_source: RegionSource | None = None,
) -> FormulaBoxAnalysis:
    if region.source != RegionSource.FORMULA_BOX_SPLIT:
        return FormulaBoxAnalysis(
            FormulaBoxKind.UNKNOWN,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ["not_formula_box_split"],
        )
    if region.kind != RegionKind.DISPLAY_MATH_BLOCK:
        return FormulaBoxAnalysis(
            FormulaBoxKind.UNKNOWN,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ["not_display_math_block"],
        )
    if not region.segments or not all(segment.has_formula_boxes for segment in region.segments):
        return FormulaBoxAnalysis(
            FormulaBoxKind.UNKNOWN,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ["non_formula_segments"],
        )

    boxes = [box for segment in region.segments for box in segment.segment.boxes if box.boxclass == "formula"]
    if not boxes:
        return FormulaBoxAnalysis(
            FormulaBoxKind.UNKNOWN,
            0.0,
            0.0,
            0.0,
            True,
            False,
            ["empty_formula_boxes"],
        )

    stub_kind = _formula_box_stub_kind(region, previous_context=previous_context, next_context=next_context)
    previous_lower = previous_context.lower()
    next_lower = next_context.lower()
    max_width = max((box.bbox[2] - box.bbox[0] for box in boxes), default=0.0)
    max_height = max((box.bbox[3] - box.bbox[1] for box in boxes), default=0.0)
    width_ratio = max_width / page_width if page_width > 0 else 0.0
    large_display = len(boxes) > 1 or max_height >= 24.0 or width_ratio >= 0.45
    tiny_inline = len(boxes) == 1 and max_height < 24.0 and width_ratio < 0.42
    prose_like_neighbors = {
        RegionKind.PROSE,
        RegionKind.INLINE_MATH_PROSE,
        RegionKind.THEOREM_PROOF_BLOCK,
    }
    math_like_neighbors = {RegionKind.DISPLAY_MATH_BLOCK}
    between_prose_regions = (
        previous_region_kind in prose_like_neighbors and next_region_kind in prose_like_neighbors
    )
    between_display_regions = (
        previous_region_kind in math_like_neighbors and next_region_kind in math_like_neighbors
    )
    inside_theorem_proof_context = (
        previous_region_kind == RegionKind.THEOREM_PROOF_BLOCK
        or next_region_kind == RegionKind.THEOREM_PROOF_BLOCK
    )
    after_formula_intro = any(
        marker in previous_lower
        for marker in ("as follows", "satisfies", "defined as", "let us define", "we define", "constraints")
    )
    before_formula_tail = next_lower.startswith("for all") or next_lower.startswith("where ")
    reasons: list[str] = []
    standalone_score = 0.65 if large_display else 0.25 if tiny_inline else 0.5
    promotion_score = 0.25 if large_display else 0.1 if tiny_inline else 0.2

    if large_display:
        reasons.append("display_sized")
    if tiny_inline:
        reasons.append("tiny_box")
    if len(boxes) > 1:
        reasons.append("formula_box_run")
    if between_prose_regions:
        reasons.append("between_prose_regions")
        standalone_score -= 0.1
        promotion_score -= 0.1
    if between_display_regions:
        reasons.append("between_display_regions")
        standalone_score += 0.1
    if inside_theorem_proof_context:
        reasons.append("inside_theorem_proof_context")
        promotion_score -= 0.05
    if after_formula_intro:
        reasons.append("formula_intro_context")
        standalone_score += 0.12
        promotion_score += 0.06
    if before_formula_tail:
        reasons.append("formula_tail_context")
        standalone_score += 0.08
        promotion_score += 0.04
    if previous_region_source == RegionSource.EMBEDDED_DISPLAY_RUN or next_region_source == RegionSource.EMBEDDED_DISPLAY_RUN:
        reasons.append("adjacent_embedded_display")
        standalone_score += 0.08

    if stub_kind == "case-definition":
        reasons.append("case_definition_stub")
        return FormulaBoxAnalysis(
            FormulaBoxKind.CASE_DEFINITION,
            0.95,
            max(standalone_score, 0.94),
            max(promotion_score, 0.93),
            True,
            True,
            reasons,
        )
    if stub_kind == "constraint":
        reasons.append("constraint_stub")
        return FormulaBoxAnalysis(
            FormulaBoxKind.CONSTRAINT_SET,
            0.93,
            max(standalone_score, 0.92),
            max(promotion_score, 0.9),
            True,
            True,
            reasons,
        )
    if stub_kind == "algorithm-step":
        reasons.append("algorithm_stub")
        standalone_score = max(standalone_score, 0.72 if large_display else 0.58)
        promotion_score = min(max(promotion_score, 0.36), 0.58)
        return FormulaBoxAnalysis(
            FormulaBoxKind.ALGORITHM_MATH,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            standalone_score >= 0.55,
            promotion_score >= 0.75,
            reasons,
        )
    if stub_kind and stub_kind.startswith("numbered-equation"):
        reasons.append("numbered_equation")
        standalone_score = max(standalone_score, 0.88)
        promotion_score = max(promotion_score, 0.9)
        return FormulaBoxAnalysis(
            FormulaBoxKind.DISPLAY_EQUATION,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            True,
            True,
            reasons,
        )
    if stub_kind == "derivation":
        reasons.append("derivation_stub")
        standalone_score = max(standalone_score, 0.82 if large_display else 0.54)
        promotion_score = max(promotion_score, 0.42 if large_display else 0.22)
        return FormulaBoxAnalysis(
            FormulaBoxKind.DERIVATION_STEP,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            standalone_score >= 0.55,
            promotion_score >= 0.8,
            reasons,
        )
    if stub_kind == "equation" and large_display:
        reasons.append("equation_stub")
        standalone_score = max(standalone_score, 0.8)
        promotion_score = max(promotion_score, 0.45)
        return FormulaBoxAnalysis(
            FormulaBoxKind.DISPLAY_EQUATION,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            standalone_score >= 0.55,
            promotion_score >= 0.82,
            reasons,
        )
    if stub_kind == "equation" and tiny_inline:
        reasons.extend(["equation_stub", "prose_neighbors"])
        standalone_score = min(standalone_score, 0.25)
        promotion_score = min(promotion_score, 0.1)
        return FormulaBoxAnalysis(
            FormulaBoxKind.INLINE_NOISE,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            False,
            False,
            reasons,
        )
    if tiny_inline:
        reasons.append("prose_neighbors")
        standalone_score = min(standalone_score, 0.3)
        promotion_score = min(promotion_score, 0.12)
        return FormulaBoxAnalysis(
            FormulaBoxKind.INLINE_NOISE,
            max(standalone_score, promotion_score),
            standalone_score,
            promotion_score,
            False,
            False,
            reasons,
        )
    return FormulaBoxAnalysis(
        FormulaBoxKind.UNKNOWN,
        max(standalone_score, promotion_score),
        standalone_score,
        promotion_score,
        standalone_score >= 0.55,
        promotion_score >= 0.82,
        reasons or ["context_dependent_formula"],
    )


def merge_low_value_formula_box_regions(regions: list[RegionBlock], *, page_width: float) -> list[RegionBlock]:
    if not regions:
        return []

    merged: list[RegionBlock] = []
    prose_like_kinds = {
        RegionKind.THEOREM_PROOF_BLOCK,
        RegionKind.PROSE,
        RegionKind.INLINE_MATH_PROSE,
        RegionKind.UNKNOWN,
    }

    def fallback_text(region: RegionBlock) -> str:
        return _serialize_region_block_stable(region, page_width=page_width)

    def find_forward_target(start: int) -> int | None:
        skipped_display = 0
        for candidate_index in range(start + 1, len(regions)):
            candidate = regions[candidate_index]
            if candidate.kind in prose_like_kinds:
                return candidate_index
            if candidate.kind == RegionKind.DISPLAY_MATH_BLOCK:
                skipped_display += 1
                if skipped_display <= 2:
                    continue
            break
        return None

    def find_backward_target() -> int | None:
        skipped_display = 0
        for candidate_index in range(len(merged) - 1, -1, -1):
            candidate = merged[candidate_index]
            if candidate.kind in prose_like_kinds:
                return candidate_index
            if candidate.kind == RegionKind.DISPLAY_MATH_BLOCK:
                skipped_display += 1
                if skipped_display <= 2:
                    continue
            break
        return None

    index = 0
    while index < len(regions):
        region = regions[index]
        previous_region = merged[-1] if merged else None
        next_region = regions[index + 1] if index + 1 < len(regions) else None
        previous_context = fallback_text(previous_region) if previous_region is not None else ""
        next_context = fallback_text(next_region) if next_region is not None else ""
        formula_analysis = analyze_formula_box_region(
            region,
            page_width=page_width,
            previous_context=previous_context,
            next_context=next_context,
            previous_region_kind=previous_region.kind if previous_region is not None else None,
            next_region_kind=next_region.kind if next_region is not None else None,
            previous_region_source=previous_region.source if previous_region is not None else None,
            next_region_source=next_region.source if next_region is not None else None,
        )

        if formula_analysis.should_standalone:
            merged.append(region)
            index += 1
            continue

        merged_into_previous = False
        backward_target = find_backward_target()
        if backward_target is not None:
            previous_region = merged[backward_target]
            merged[backward_target] = _build_region_block(
                previous_region.kind,
                previous_region.segments + region.segments,
                page_number=previous_region.page_number,
                source=previous_region.source,
            )
            merged_into_previous = True

        if merged_into_previous:
            index += 1
            continue

        forward_target = find_forward_target(index)
        if forward_target is not None:
            next_region = regions[forward_target]
            regions[forward_target] = _build_region_block(
                next_region.kind,
                region.segments + next_region.segments,
                page_number=next_region.page_number,
                source=next_region.source,
            )
        else:
            merged.append(region)

        index += 1

    return merged


def build_page_regions(page: RawLayoutPage) -> list[RegionBlock]:
    analyses = [_analyze_segment(segment, page_width=page.width) for segment in build_page_segments(page)]
    split_regions = split_embedded_display_math_regions(_build_initial_page_regions(page, analyses))
    return merge_low_value_formula_box_regions(split_regions, page_width=page.width)


def _serialize_lines_as_paragraph(lines: list[MathLine]) -> str:
    if not lines:
        return ""
    text = "\n".join(line.text for line in lines if line.text)
    text = re.sub(r" *\n *", "\n", text)
    return _stitch_region_text(text).strip()


def _serialize_prose_region(region: RegionBlock) -> str:
    paragraphs = []
    for segment in region.segments:
        paragraph = _serialize_lines_as_paragraph(segment.lines)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def _serialize_segment_flow(
    segment: SegmentAnalysis,
    *,
    previous_context: str,
    next_context: str,
) -> str:
    items: list[tuple[str, tuple[float, float, float, float], object]] = []
    for line in segment.lines:
        items.append(("line", line.bbox, line))
    for box in segment.segment.boxes:
        if box.boxclass == "formula":
            items.append(("formula", box.bbox, box))
    items.sort(key=lambda item: (item[1][1], item[1][0], 1 if item[0] == "formula" else 0))

    parts: list[str] = []
    index = 0
    while index < len(items):
        item_kind, _, payload = items[index]
        if item_kind == "line":
            line = payload
            if line.text:
                parts.append(line.text)
            index += 1
            continue

        group = [payload]
        lookahead = index + 1
        while lookahead < len(items) and items[lookahead][0] == "formula":
            group.append(items[lookahead][2])
            lookahead += 1

        before_text = parts[-1] if parts else previous_context
        after_text = next_context
        for later_kind, _, later_payload in items[lookahead:]:
            if later_kind == "line" and later_payload.text:
                after_text = later_payload.text
                break
        rendered = _render_formula_group_stub(group, before_text=before_text, after_text=after_text)
        parts.append(rendered)
        index = lookahead

    compacted: list[str] = []
    for part in parts:
        if not part:
            continue
        if compacted and _is_standalone_equation_number_text(part) and _ends_with_numbered_formula_stub(compacted[-1]):
            continue
        compacted.append(part)
    return "\n\n".join(compacted).strip()


def _serialize_display_math_block(
    region: RegionBlock,
    *,
    page_width: float,
    previous_context: str = "",
    next_context: str = "",
) -> str:
    parts: list[str] = []
    for index, segment in enumerate(region.segments):
        segment_previous_context = parts[-1] if parts else previous_context
        segment_next_context = next_context
        for later_segment in region.segments[index + 1 :]:
            later_lines = [line.text for line in later_segment.lines if line.text]
            if later_lines:
                segment_next_context = "\n".join(later_lines)
                break
        if segment.has_formula_boxes:
            parts.append(
                _serialize_segment_flow(
                    segment,
                    previous_context=segment_previous_context,
                    next_context=segment_next_context,
                )
            )
        else:
            parts.append("\n".join(line.text for line in segment.lines if line.text))
    compacted: list[str] = []
    for part in (part for part in parts if part):
        if compacted and _is_standalone_equation_number_text(part) and _ends_with_numbered_formula_stub(compacted[-1]):
            continue
        compacted.append(part)
    return "\n\n".join(compacted).strip()


def _serialize_theorem_proof_block(region: RegionBlock, *, page_width: float) -> str:
    parts: list[str] = []
    for index, segment in enumerate(region.segments):
        if segment.has_formula_boxes or segment.base_kind == RegionKind.DISPLAY_MATH_BLOCK:
            previous_context = ""
            next_context = ""
            if parts:
                previous_context = parts[-1]
            for later_segment in region.segments[index + 1 :]:
                later_lines = [line for line in later_segment.lines if line.text]
                if later_lines:
                    next_context = "\n".join(line.text for line in later_lines)
                    break
            rendered = _serialize_segment_flow(
                segment,
                previous_context=previous_context,
                next_context=next_context,
            )
        else:
            rendered = _serialize_lines_as_paragraph(segment.lines)
        if rendered:
            if parts and _is_standalone_equation_number_text(rendered) and _ends_with_numbered_formula_stub(parts[-1]):
                continue
            parts.append(rendered)
    return "\n\n".join(parts).strip()


def should_use_region_serializer(
    region: RegionBlock,
    *,
    page_width: float | None = None,
    previous_context: str = "",
    next_context: str = "",
    previous_region_kind: RegionKind | None = None,
    next_region_kind: RegionKind | None = None,
    previous_region_source: RegionSource | None = None,
    next_region_source: RegionSource | None = None,
) -> bool:
    if region.kind != RegionKind.DISPLAY_MATH_BLOCK:
        return False
    if region.confidence < 0.85:
        return False
    if previous_region_kind == RegionKind.CAPTION or next_region_kind == RegionKind.CAPTION:
        return False
    if re.search(r"\b(?:Figure|Fig\.|Table)\b", previous_context) or re.search(r"\b(?:Figure|Fig\.|Table)\b", next_context):
        return False
    if any(segment.has_formula_boxes for segment in region.segments):
        if region.source != RegionSource.FORMULA_BOX_SPLIT or page_width is None:
            return False
        analysis = analyze_formula_box_region(
            region,
            page_width=page_width,
            previous_context=previous_context,
            next_context=next_context,
            previous_region_kind=previous_region_kind,
            next_region_kind=next_region_kind,
            previous_region_source=previous_region_source,
            next_region_source=next_region_source,
        )
        return analysis.should_promote
    if region.source == RegionSource.FORMULA_BOX_SPLIT:
        if page_width is None:
            return False
        analysis = analyze_formula_box_region(
            region,
            page_width=page_width,
            previous_context=previous_context,
            next_context=next_context,
            previous_region_kind=previous_region_kind,
            next_region_kind=next_region_kind,
            previous_region_source=previous_region_source,
            next_region_source=next_region_source,
        )
        return analysis.should_promote

    for segment in region.segments:
        if segment.has_formula_boxes:
            continue
        if segment.base_kind != RegionKind.DISPLAY_MATH_BLOCK:
            return False
        for line in segment.lines:
            if line.classification.kind in {
                LineKind.THEOREM_PROOF,
                LineKind.ALGORITHM,
                LineKind.CAPTION,
                LineKind.LIST,
                LineKind.PROSE,
                LineKind.INLINE_MATH_PROSE,
            }:
                return False
            if re.search(r"\b(?:to|and|or|the|is|for|when|where|such)\b", line.text):
                return False
            if line.text.endswith("."):
                return False

    return True


def _serialize_region_block_stable(region: RegionBlock, *, page_width: float) -> str:
    paragraphs: list[str] = []
    for index, segment in enumerate(region.segments):
        if segment.has_formula_boxes:
            previous_context = ""
            next_context = ""
            if paragraphs:
                previous_context = paragraphs[-1]
            for later_segment in region.segments[index + 1 :]:
                later_lines = [line.text for line in later_segment.lines if line.text]
                if later_lines:
                    next_context = "\n".join(later_lines)
                    break
            rendered = _serialize_math_region_for_page(
                MathRegion(
                    page_number=segment.segment.page_number,
                    start_box_index=segment.segment.start_box_index,
                    end_box_index=segment.segment.end_box_index,
                    boxes=segment.segment.boxes,
                ),
                previous_context=previous_context,
                next_context=next_context,
                page_width=page_width,
            )
        else:
            rendered = _serialize_lines_as_paragraph(segment.lines)
        if rendered:
            if paragraphs and _is_standalone_equation_number_text(rendered) and _ends_with_numbered_formula_stub(paragraphs[-1]):
                continue
            paragraphs.append(rendered)
    return "\n\n".join(paragraphs).strip()


def serialize_region_block(
    region: RegionBlock,
    *,
    page_width: float,
    previous_context: str = "",
    next_context: str = "",
) -> str:
    if region.kind == RegionKind.DISPLAY_MATH_BLOCK:
        return _serialize_display_math_block(
            region,
            page_width=page_width,
            previous_context=previous_context,
            next_context=next_context,
        )
    if region.kind == RegionKind.THEOREM_PROOF_BLOCK:
        return _serialize_theorem_proof_block(region, page_width=page_width)
    if region.kind in {RegionKind.PROSE, RegionKind.INLINE_MATH_PROSE}:
        return _serialize_prose_region(region)

    lines = [line for segment in region.segments for line in segment.lines]
    if region.kind in {RegionKind.CAPTION, RegionKind.LIST_BLOCK, RegionKind.ALGORITHM_BLOCK}:
        return _serialize_lines_as_paragraph(lines)
    return _serialize_lines_as_paragraph(lines)


def build_region_review_report(document: RawLayoutDocument) -> str:
    lines = [
        "scimark region review",
        f"source: {document.source_pdf}",
        f"page_count: {document.page_count}",
        "",
    ]

    for page in document.pages:
        lines.append(f"page {page.page_number}")
        regions = build_page_regions(page)
        if not regions:
            lines.append("  no regions")
            lines.append("")
            continue
        stable_texts = [_serialize_region_block_stable(region, page_width=page.width) for region in regions]

        for index, region in enumerate(regions):
            previous_context = ""
            next_context = ""
            for earlier in reversed(stable_texts[:index]):
                if earlier:
                    previous_context = earlier
                    break
            for later in stable_texts[index + 1 :]:
                if later:
                    next_context = later
                    break
            formula_analysis = None
            if region.source == RegionSource.FORMULA_BOX_SPLIT:
                formula_analysis = analyze_formula_box_region(
                    region,
                    page_width=page.width,
                    previous_context=previous_context,
                    next_context=next_context,
                    previous_region_kind=regions[index - 1].kind if index > 0 else None,
                    next_region_kind=regions[index + 1].kind if index + 1 < len(regions) else None,
                    previous_region_source=regions[index - 1].source if index > 0 else None,
                    next_region_source=regions[index + 1].source if index + 1 < len(regions) else None,
                )
            use_specialized = should_use_region_serializer(
                region,
                page_width=page.width,
                previous_context=previous_context,
                next_context=next_context,
                previous_region_kind=regions[index - 1].kind if index > 0 else None,
                next_region_kind=regions[index + 1].kind if index + 1 < len(regions) else None,
                previous_region_source=regions[index - 1].source if index > 0 else None,
                next_region_source=regions[index + 1].source if index + 1 < len(regions) else None,
            )
            lines.append(
                "  "
                f"REGION {index} | {region.kind.value} | source={region.source.value} | confidence={region.confidence:.2f} | "
                f"{'specialized=yes' if use_specialized else 'fallback=stable'}"
            )
            if region.reasons:
                lines.append(f"    reasons: {', '.join(region.reasons)}")
            line_kinds: list[str] = []
            for segment in region.segments:
                if segment.has_formula_boxes and not segment.lines:
                    line_kinds.append("formula_box")
                else:
                    line_kinds.extend(line.classification.kind.value for line in segment.lines)
            if line_kinds:
                lines.append(f"    lines: {', '.join(line_kinds)}")
            if formula_analysis is not None:
                lines.append(
                    "    "
                    f"formula_kind={formula_analysis.kind.value} usefulness={formula_analysis.usefulness_score:.2f} "
                    f"standalone_score={formula_analysis.standalone_score:.2f} "
                    f"promotion_score={formula_analysis.promotion_score:.2f} "
                    f"standalone={'yes' if formula_analysis.should_standalone else 'no'} "
                    f"promote={'yes' if formula_analysis.should_promote else 'no'}"
                )
                if formula_analysis.reasons:
                    lines.append(f"      formula_reasons: {', '.join(formula_analysis.reasons)}")
            for segment in region.segments:
                runs = find_embedded_display_math_runs(segment)
                for run_index, run in enumerate(runs):
                    status = "accepted" if run.accepted else "rejected"
                    details = ", ".join(run.rejected_reasons) if run.rejected_reasons else ", ".join(run.reasons)
                    lines.append(
                        "    "
                        f"run {run_index}: lines={run.start}-{run.end} conf={run.confidence:.2f} {status}"
                    )
                    if details:
                        lines.append(f"      reasons: {details}")
            stable = stable_texts[index]
            lines.append(f"    stable: {stable}" if stable else "    stable: <empty>")
            if use_specialized:
                specialized = serialize_region_block(
                    region,
                    page_width=page.width,
                    previous_context=previous_context,
                    next_context=next_context,
                )
                lines.append(f"    specialized: {specialized}" if specialized else "    specialized: <empty>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def serialize_page(page: RawLayoutPage) -> str:
    segments = build_page_segments(page)
    rendered: list[str] = []

    for index, segment in enumerate(segments):
        if segment.kind == "math_region":
            previous_context = ""
            next_context = ""
            if index > 0:
                previous_context = _segment_context_text(segments[index - 1], page_width=page.width)
            if index + 1 < len(segments):
                next_context = _segment_context_text(segments[index + 1], page_width=page.width)
            text = _serialize_math_region_for_page(
                MathRegion(
                    page_number=segment.page_number,
                    start_box_index=segment.start_box_index,
                    end_box_index=segment.end_box_index,
                    boxes=segment.boxes,
                ),
                previous_context=previous_context,
                next_context=next_context,
                page_width=page.width,
            )
        else:
            text = _serialize_prose_boxes(segment.boxes, page_width=page.width)
        if not text:
            continue
        if rendered and _is_standalone_equation_number_text(text) and _ends_with_numbered_formula_stub(rendered[-1]):
            continue
        if rendered:
            rendered.append("\n\n")
        rendered.append(text)

    return _normalize_surface_text("".join(rendered).strip())


def serialize_page_with_region_promotion(page: RawLayoutPage) -> str:
    regions = build_page_regions(page)
    stable_texts = [_serialize_region_block_stable(region, page_width=page.width) for region in regions]
    rendered: list[str] = []

    for index, region in enumerate(regions):
        previous_context = rendered[-1] if rendered else ""
        next_context = ""
        for later_text in stable_texts[index + 1 :]:
            if later_text:
                next_context = later_text
                break
        previous_region = regions[index - 1] if index > 0 else None
        next_region = regions[index + 1] if index + 1 < len(regions) else None
        if should_use_region_serializer(
            region,
            page_width=page.width,
            previous_context=previous_context,
            next_context=next_context,
            previous_region_kind=previous_region.kind if previous_region is not None else None,
            next_region_kind=next_region.kind if next_region is not None else None,
            previous_region_source=previous_region.source if previous_region is not None else None,
            next_region_source=next_region.source if next_region is not None else None,
        ):
            text = serialize_region_block(
                region,
                page_width=page.width,
                previous_context=previous_context,
                next_context=next_context,
            )
        else:
            text = stable_texts[index]
        if not text:
            continue
        if rendered and _is_standalone_equation_number_text(text) and _ends_with_numbered_formula_stub(rendered[-1]):
            continue
        if rendered:
            rendered.append("\n\n")
        rendered.append(text)

    return _normalize_surface_text("".join(rendered).strip())


def build_page_math_diagnostics(page: RawLayoutPage) -> PageMathDiagnostics:
    analyses = [_analyze_segment(segment, page_width=page.width) for segment in build_page_segments(page)]
    pre_merge_regions = split_embedded_display_math_regions(_build_initial_page_regions(page, analyses))
    final_regions = merge_low_value_formula_box_regions(pre_merge_regions, page_width=page.width)

    display_math_regions = 0
    promoted_regions = 0
    formula_box_split_regions = 0
    embedded_display_run_regions = 0
    caption_regions = 0
    suspected_figure_label_suppressions = 0
    author_affiliation_suppressions = 0
    bibliography_entry_suppressions = 0

    for analysis in analyses:
        for line in analysis.lines:
            if "diagram_label_like" in line.classification.reasons:
                suspected_figure_label_suppressions += 1
            if "author_affiliation_like" in line.classification.reasons:
                author_affiliation_suppressions += 1
            if "bibliography_entry_like" in line.classification.reasons:
                bibliography_entry_suppressions += 1

    stable_texts = [_serialize_region_block_stable(region, page_width=page.width) for region in final_regions]
    for index, region in enumerate(final_regions):
        if region.kind == RegionKind.DISPLAY_MATH_BLOCK:
            display_math_regions += 1
        if region.kind == RegionKind.CAPTION:
            caption_regions += 1
        if region.source == RegionSource.FORMULA_BOX_SPLIT:
            formula_box_split_regions += 1
        if region.source == RegionSource.EMBEDDED_DISPLAY_RUN:
            embedded_display_run_regions += 1

        previous_context = ""
        next_context = ""
        for earlier in reversed(stable_texts[:index]):
            if earlier:
                previous_context = earlier
                break
        for later in stable_texts[index + 1 :]:
            if later:
                next_context = later
                break

        previous_region = final_regions[index - 1] if index > 0 else None
        next_region = final_regions[index + 1] if index + 1 < len(final_regions) else None
        if should_use_region_serializer(
            region,
            page_width=page.width,
            previous_context=previous_context,
            next_context=next_context,
            previous_region_kind=previous_region.kind if previous_region is not None else None,
            next_region_kind=next_region.kind if next_region is not None else None,
            previous_region_source=previous_region.source if previous_region is not None else None,
            next_region_source=next_region.source if next_region is not None else None,
        ):
            promoted_regions += 1

    fallback_regions = max(display_math_regions - promoted_regions, 0)
    reference_like = bibliography_entry_suppressions >= 5 and display_math_regions <= 1
    figure_table_like = caption_regions >= 2 and promoted_regions == 0 and display_math_regions <= 2
    experimental_candidate = (
        not reference_like
        and not figure_table_like
        and (
            promoted_regions > 0
            or embedded_display_run_regions > 0
            or formula_box_split_regions >= 2
            or display_math_regions >= 3
        )
    )
    reasons: list[str] = []
    if reference_like:
        reasons.append("reference_like_page")
    if figure_table_like:
        reasons.append("figure_table_like_page")
    if experimental_candidate:
        reasons.append("experimental_candidate")

    return PageMathDiagnostics(
        page_number=page.page_number,
        region_count=len(final_regions),
        display_math_regions=display_math_regions,
        promoted_regions=promoted_regions,
        fallback_regions=fallback_regions,
        formula_box_split_regions=formula_box_split_regions,
        embedded_display_run_regions=embedded_display_run_regions,
        caption_regions=caption_regions,
        suspected_figure_label_suppressions=suspected_figure_label_suppressions,
        author_affiliation_suppressions=author_affiliation_suppressions,
        bibliography_entry_suppressions=bibliography_entry_suppressions,
        experimental_candidate=experimental_candidate,
        reference_like=reference_like,
        figure_table_like=figure_table_like,
        reasons=reasons,
    )


def should_use_experimental_math_on_page(page: RawLayoutPage) -> bool:
    return build_page_math_diagnostics(page).experimental_candidate


def build_paper_math_diagnostics(document: RawLayoutDocument) -> PaperMathDiagnostics:
    region_count = 0
    display_math_regions = 0
    promoted_regions = 0
    formula_box_split_regions = 0
    embedded_display_run_regions = 0
    suspected_figure_label_suppressions = 0
    author_affiliation_suppressions = 0
    bibliography_entry_suppressions = 0
    inline_noise_merged = 0
    experimental_candidate_pages = 0
    reference_like_pages = 0
    figure_table_like_pages = 0

    for page in document.pages:
        analyses = [_analyze_segment(segment, page_width=page.width) for segment in build_page_segments(page)]
        pre_merge_regions = split_embedded_display_math_regions(_build_initial_page_regions(page, analyses))
        final_regions = merge_low_value_formula_box_regions(pre_merge_regions, page_width=page.width)
        page_diagnostics = build_page_math_diagnostics(page)

        region_count += page_diagnostics.region_count
        display_math_regions += page_diagnostics.display_math_regions
        promoted_regions += page_diagnostics.promoted_regions
        formula_box_split_regions += page_diagnostics.formula_box_split_regions
        embedded_display_run_regions += page_diagnostics.embedded_display_run_regions
        suspected_figure_label_suppressions += page_diagnostics.suspected_figure_label_suppressions
        author_affiliation_suppressions += page_diagnostics.author_affiliation_suppressions
        bibliography_entry_suppressions += page_diagnostics.bibliography_entry_suppressions
        experimental_candidate_pages += int(page_diagnostics.experimental_candidate)
        reference_like_pages += int(page_diagnostics.reference_like)
        figure_table_like_pages += int(page_diagnostics.figure_table_like)

        for index, region in enumerate(pre_merge_regions):
            if region.source != RegionSource.FORMULA_BOX_SPLIT:
                continue
            previous_region = pre_merge_regions[index - 1] if index > 0 else None
            next_region = pre_merge_regions[index + 1] if index + 1 < len(pre_merge_regions) else None
            previous_context = _serialize_region_block_stable(previous_region, page_width=page.width) if previous_region is not None else ""
            next_context = _serialize_region_block_stable(next_region, page_width=page.width) if next_region is not None else ""
            analysis = analyze_formula_box_region(
                region,
                page_width=page.width,
                previous_context=previous_context,
                next_context=next_context,
                previous_region_kind=previous_region.kind if previous_region is not None else None,
                next_region_kind=next_region.kind if next_region is not None else None,
                previous_region_source=previous_region.source if previous_region is not None else None,
                next_region_source=next_region.source if next_region is not None else None,
            )
            if analysis.kind == FormulaBoxKind.INLINE_NOISE and not analysis.should_standalone:
                inline_noise_merged += 1

    return PaperMathDiagnostics(
        pages_evaluated=document.page_count,
        region_count=region_count,
        display_math_regions=display_math_regions,
        promoted_regions=promoted_regions,
        fallback_regions=max(display_math_regions - promoted_regions, 0),
        formula_box_split_regions=formula_box_split_regions,
        embedded_display_run_regions=embedded_display_run_regions,
        inline_noise_merged=inline_noise_merged,
        suspected_figure_label_suppressions=suspected_figure_label_suppressions,
        author_affiliation_suppressions=author_affiliation_suppressions,
        bibliography_entry_suppressions=bibliography_entry_suppressions,
        experimental_candidate_pages=experimental_candidate_pages,
        reference_like_pages=reference_like_pages,
        figure_table_like_pages=figure_table_like_pages,
    )


def _build_initial_page_regions(page: RawLayoutPage, analyses: list[SegmentAnalysis]) -> list[RegionBlock]:
    regions: list[RegionBlock] = []
    index = 0

    while index < len(analyses):
        current = analyses[index]
        kind = current.base_kind
        grouped = [current]
        index += 1

        if kind == RegionKind.THEOREM_PROOF_BLOCK:
            while index < len(analyses) and _can_absorb_into_theorem_block(analyses[index]):
                if analyses[index].base_kind == RegionKind.THEOREM_PROOF_BLOCK and grouped:
                    break
                grouped.append(analyses[index])
                index += 1
        elif kind == RegionKind.DISPLAY_MATH_BLOCK:
            while index < len(analyses) and analyses[index].base_kind == RegionKind.DISPLAY_MATH_BLOCK:
                grouped.append(analyses[index])
                index += 1
        elif kind in {RegionKind.PROSE, RegionKind.INLINE_MATH_PROSE}:
            if index < len(analyses) and analyses[index].base_kind == RegionKind.DISPLAY_MATH_BLOCK and _segment_ends_formula_preface(current):
                kind = RegionKind.THEOREM_PROOF_BLOCK
                grouped.append(analyses[index])
                index += 1
                while index < len(analyses) and analyses[index].base_kind in {
                    RegionKind.PROSE,
                    RegionKind.INLINE_MATH_PROSE,
                    RegionKind.DISPLAY_MATH_BLOCK,
                }:
                    if analyses[index].base_kind == RegionKind.THEOREM_PROOF_BLOCK:
                        break
                    grouped.append(analyses[index])
                    index += 1
            else:
                while index < len(analyses) and analyses[index].base_kind in {RegionKind.PROSE, RegionKind.INLINE_MATH_PROSE}:
                    kind = _merge_prose_kind(kind, analyses[index].base_kind)
                    grouped.append(analyses[index])
                    index += 1
        elif kind == RegionKind.ALGORITHM_BLOCK:
            while index < len(analyses) and analyses[index].base_kind == RegionKind.ALGORITHM_BLOCK:
                grouped.append(analyses[index])
                index += 1
        elif kind == RegionKind.LIST_BLOCK:
            while index < len(analyses) and analyses[index].base_kind == RegionKind.LIST_BLOCK:
                grouped.append(analyses[index])
                index += 1

        regions.append(
            _build_region_block(
                kind,
                grouped,
                page_number=page.page_number,
                source=RegionSource.INITIAL_GROUPING,
            )
        )

    return regions


def build_math_debug_report(
    document: RawLayoutDocument,
    *,
    include_all_boxes: bool = False,
) -> str:
    lines = [
        "scimark math debug",
        f"source: {document.source_pdf}",
        f"page_count: {document.page_count}",
        "",
    ]

    for page in document.pages:
        lines.extend(_format_page_report(page, include_all_boxes=include_all_boxes))

    return "\n".join(lines).rstrip() + "\n"


def _format_page_report(page: RawLayoutPage, *, include_all_boxes: bool) -> list[str]:
    ordered_page = reorder_page_boxes_for_reading(page)
    reordered = [box.source_index for box in ordered_page.boxes] != [box.source_index for box in page.boxes]
    page_prototype = serialize_page(page)
    page_regions = build_page_regions(page)
    lines = [
        f"page {page.page_number} [{page.width:.1f} x {page.height:.1f}]",
        f"flags: full_ocred={page.full_ocred} text_ocred={page.text_ocred}",
        f"reading_order: {'column-reordered' if reordered else 'original'}",
    ]
    if page_prototype:
        lines.append(f"page-prototype: {page_prototype}")
    if page_regions:
        for region_index, region in enumerate(page_regions):
            lines.append(
                "  "
                f"page-region {region_index} [{region.kind.value}] conf={region.confidence:.2f} "
                f"bbox={_format_bbox(region.bbox)}"
            )
            if region.reasons:
                lines.append(f"    reasons: {', '.join(region.reasons)}")
            region_text = serialize_region_block(region, page_width=page.width)
            if region_text:
                lines.append(f"    region-prototype: {region_text}")

    if include_all_boxes:
        boxes_to_render = [
            MathRegion(
                page_number=ordered_page.page_number,
                start_box_index=index,
                end_box_index=index,
                boxes=[box],
            )
            for index, box in enumerate(ordered_page.boxes)
        ]
    else:
        boxes_to_render = find_math_regions(ordered_page)

    for region_index, region in enumerate(boxes_to_render):
        lines.append(
            "  "
            f"region {region_index} boxes={region.start_box_index}-{region.end_box_index} "
            f"bbox={_format_bbox(region.bbox)}"
        )
        prototype = serialize_region(region, page_width=ordered_page.width)
        if prototype:
            lines.append(f"    prototype: {prototype}")

        for box_offset, box in enumerate(region.boxes, start=region.start_box_index):
            lines.append(
                "    "
                f"box {box_offset} src={box.source_index} [{box.boxclass}] "
                f"bbox={_format_bbox(box.bbox)} image={box.has_image}"
            )
            box_prototype = serialize_box(box, page_width=ordered_page.width)
            if box_prototype:
                lines.append(f"      box-prototype: {box_prototype}")

            for line_index, line in enumerate(box.textlines):
                virtual_lines = split_line_by_baseline(line)
                if len(virtual_lines) > 1:
                    lines.append(f"      line {line_index}: split into {len(virtual_lines)} virtual lines")
                    for virtual_index, virtual_line in enumerate(virtual_lines):
                        serialized = serialize_line(virtual_line)
                        classification = classify_line(
                            virtual_line,
                            serialized_text=serialized,
                            box_class=box.boxclass,
                            page_width=ordered_page.width,
                            box_bbox=box.bbox,
                        )
                        if not serialized and not include_all_boxes:
                            continue
                        lines.append(
                            f"        vline {virtual_index}: {serialized} "
                            f"[{classification.kind.value} conf={classification.confidence:.2f}]"
                        )
                        if classification.reasons:
                            lines.append(f"          reasons: {', '.join(classification.reasons)}")
                        if classification.features is not None:
                            lines.append(
                                "          "
                                f"features: sym={classification.features.symbol_density:.2f} "
                                f"ops={classification.features.operator_count} "
                                f"words={classification.features.word_count} "
                                f"prose={classification.features.prose_word_count} "
                                f"scripts={classification.features.script_fraction:.2f} "
                                f"center={classification.features.centeredness:.2f} "
                                f"width={classification.features.relative_width:.2f}"
                            )
                        for span_index, span in enumerate(virtual_line.spans):
                            role = classify_span_role(virtual_line.spans, span_index)
                            lines.append(
                                "          "
                                f"span {span_index}: text={span.text!r} role={role} "
                                f"base={span.baseline:.2f} size={span.size:.2f} bbox={_format_bbox(span.bbox)}"
                            )
                    continue

                serialized = serialize_line(line)
                classification = classify_line(
                    line,
                    serialized_text=serialized,
                    box_class=box.boxclass,
                    page_width=ordered_page.width,
                    box_bbox=box.bbox,
                )
                if not serialized and not include_all_boxes:
                    continue
                lines.append(
                    f"      line {line_index}: {serialized} "
                    f"[{classification.kind.value} conf={classification.confidence:.2f}]"
                )
                if classification.reasons:
                    lines.append(f"        reasons: {', '.join(classification.reasons)}")
                if classification.features is not None:
                    lines.append(
                        "        "
                        f"features: sym={classification.features.symbol_density:.2f} "
                        f"ops={classification.features.operator_count} "
                        f"words={classification.features.word_count} "
                        f"prose={classification.features.prose_word_count} "
                        f"scripts={classification.features.script_fraction:.2f} "
                        f"center={classification.features.centeredness:.2f} "
                        f"width={classification.features.relative_width:.2f}"
                    )
                for span_index, span in enumerate(line.spans):
                    role = classify_span_role(line.spans, span_index)
                    lines.append(
                        "        "
                        f"span {span_index}: text={span.text!r} role={role} "
                        f"base={span.baseline:.2f} size={span.size:.2f} bbox={_format_bbox(span.bbox)}"
                    )

        lines.append("")

    lines.append("")
    return lines
