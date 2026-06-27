from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Callable, Iterable, Mapping, Sequence

# Sentence boundary pattern
SentenceBoundaryPattern = re.compile(r"(?<=[.!?])\s+")
TagPattern = re.compile(r"[<\[{]\s*([A-Za-z0-9_]+)\s*[>\]}]")


@dataclass(frozen=True)
class SentenceSpan:
    """A sentence and its character offsets in the source text."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class RedactionEvent:
    """Replacement span in original text plus normalized tag entities."""

    start: int
    end: int
    tag_entities: frozenset[str]


@dataclass(frozen=True)
class GroundTruthOccurrence:
    """Finalized ground-truth occurrence used for scoring."""

    start: int
    end: int
    entity: str
    value: str


SentenceSegmenter = Callable[[str], Sequence[SentenceSpan]]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim surrounding whitespace and return adjusted boundaries."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def regex_sentence_segmenter(text: str) -> list[SentenceSpan]:
    """Segment text into sentence spans using punctuation boundaries."""

    segments: list[SentenceSpan] = []
    start = 0

    for match in SentenceBoundaryPattern.finditer(text):
        end = match.start()
        bounds = _trim_span(text=text, start=start, end=end)
        if bounds is not None:
            seg_start, seg_end = bounds
            segments.append(
                SentenceSpan(text=text[seg_start:seg_end], start=seg_start, end=seg_end)
            )
        start = match.end()

    bounds = _trim_span(text=text, start=start, end=len(text))
    if bounds is not None:
        seg_start, seg_end = bounds
        segments.append(
            SentenceSpan(text=text[seg_start:seg_end], start=seg_start, end=seg_end)
        )

    return segments


class RedactionEvaluator:
    """Evaluate redaction quality from tabular rows."""

    def __init__(
        self,
        original_text_column: str,
        entity_columns: Iterable[str],
        entity_aliases: Mapping[str, Iterable[str]] | None = None,
        iou_threshold: float = 0.8,
        strict_entity_matching: bool = True,
        segmenter: SentenceSegmenter | None = None,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        self.original_text_column = original_text_column
        self.entity_columns = list(entity_columns)
        self._entity_order = {
            entity: idx for idx, entity in enumerate(self.entity_columns)
        }
        self.iou_threshold = iou_threshold
        self.strict_entity_matching = strict_entity_matching
        self._segmenter = segmenter or regex_sentence_segmenter
        self._entity_aliases = self._build_alias_map(entity_aliases=entity_aliases)
        self._alias_to_entity = self._build_reverse_alias_map()

    def _segment_sentences(self, text: str) -> list[SentenceSpan]:
        """Call configured segmenter and validate its output."""

        segments = list(self._segmenter(text))
        for segment in segments:
            if not isinstance(segment, SentenceSpan):
                raise TypeError(
                    "Segmenter must return Sequence[SentenceSpan]. "
                    f"Found type: {type(segment)!r}"
                )
        return segments

    def segment_row(
        self, *, original_text: str, redacted_text: str
    ) -> tuple[list[SentenceSpan], list[SentenceSpan]]:
        """Return sentence spans for one row's original and redacted text."""

        return (
            self._segment_sentences(original_text),
            self._segment_sentences(redacted_text),
        )

    def evaluate(
        self, data: object, *, redacted_text_column: str, beta: float = 2.0
    ) -> EvaluationResults:
        if beta <= 0:
            raise ValueError("beta must be > 0.")

        per_entity_counts = {entity: Counts() for entity in self.entity_columns}
        warnings: list[str] = []
        evaluated_rows = 0
        skipped_rows = 0
        extra_fp = 0

        for row_idx, row in self._iter_rows(data):
            original_text = self._as_text(
                self._get_row_value(row=row, key=self.original_text_column)
            )
            redacted_text = self._as_text(
                self._get_row_value(row=row, key=redacted_text_column)
            )

            original_segments, redacted_segments = self.segment_row(
                original_text=original_text, redacted_text=redacted_text
            )
            if len(original_segments) != len(redacted_segments):
                skipped_rows += 1
                warnings.append(
                    f"Row {row_idx} skipped due to sentence mismatch: "
                    f"{len(original_segments)} original vs {len(redacted_segments)} redacted."
                )
                continue

            evaluated_rows += 1
            redaction_events = self._extract_redaction_events(
                original_segments=original_segments,
                redacted_segments=redacted_segments,
            )
            gt_occurrences = self._build_gt_occurrences(row=row, original_text=original_text)
            matched_event_indices: set[int] = set()

            for occurrence in gt_occurrences:
                entity_counts = per_entity_counts[occurrence.entity]
                event_idx, best_iou = self._best_event_match(
                    original_text=original_text,
                    target_start=occurrence.start,
                    target_end=occurrence.end,
                    redaction_events=redaction_events,
                )
                if event_idx is None or best_iou < self.iou_threshold:
                    entity_counts.fn += 1
                    continue

                matched_event_indices.add(event_idx)
                event = redaction_events[event_idx]
                if self.strict_entity_matching:
                    if occurrence.entity in event.tag_entities:
                        entity_counts.tp += 1
                    elif event.tag_entities:
                        # Mislabeled tag under strict mode.
                        entity_counts.fn += 1
                        entity_counts.fp += 1
                    else:
                        entity_counts.fn += 1
                else:
                    if event.tag_entities:
                        entity_counts.tp += 1
                    else:
                        entity_counts.fn += 1

            for idx, event in enumerate(redaction_events):
                if idx in matched_event_indices:
                    continue
                if not event.tag_entities:
                    continue
                for tagged_entity in event.tag_entities:
                    if tagged_entity in per_entity_counts:
                        per_entity_counts[tagged_entity].fp += 1
                    else:
                        extra_fp += 1

        per_entity = {
            entity: CountsSnapshot.from_counts(per_entity_counts[entity], beta=beta)
            for entity in self.entity_columns
        }
        overall_counts = Counts(
            tp=sum(entity_counts.tp for entity_counts in per_entity_counts.values()),
            fp=sum(entity_counts.fp for entity_counts in per_entity_counts.values())
            + extra_fp,
            fn=sum(entity_counts.fn for entity_counts in per_entity_counts.values()),
        )
        overall = CountsSnapshot.from_counts(overall_counts, beta=beta)
        return EvaluationResults(
            overall=overall,
            per_entity=per_entity,
            warnings=warnings,
            evaluated_rows=evaluated_rows,
            skipped_rows=skipped_rows,
        )

    def _extract_redaction_events(
        self,
        *,
        original_segments: Sequence[SentenceSpan],
        redacted_segments: Sequence[SentenceSpan],
    ) -> list[RedactionEvent]:
        events: list[RedactionEvent] = []
        for orig_segment, red_segment in zip(original_segments, redacted_segments):
            tag_matches = list(TagPattern.finditer(red_segment.text))
            if not tag_matches:
                continue

            matcher = SequenceMatcher(a=orig_segment.text, b=red_segment.text, autojunk=False)
            opcodes = matcher.get_opcodes()

            for tag_match in tag_matches:
                tag_start, tag_end = tag_match.span()
                i_bounds: list[int] = []
                for _, i1, i2, j1, j2 in opcodes:
                    if not _opcode_touches_tag(j1=j1, j2=j2, tag_start=tag_start, tag_end=tag_end):
                        continue
                    i_bounds.extend([i1, i2])

                if not i_bounds:
                    continue

                red_fragment = red_segment.text[tag_start:tag_end]
                entities = self._extract_tag_entities(red_fragment)
                if not entities:
                    continue
                events.append(
                    RedactionEvent(
                        start=orig_segment.start + min(i_bounds),
                        end=orig_segment.start + max(i_bounds),
                        tag_entities=frozenset(entities),
                    )
                )
        return events

    def _extract_tag_entities(self, text: str) -> set[str]:
        entities: set[str] = set()
        for match in TagPattern.finditer(text):
            alias = match.group(1).strip().lower()
            entity = self._alias_to_entity.get(alias)
            if entity is not None:
                entities.add(entity)
        return entities

    def _build_gt_occurrences(
        self, *, row: object, original_text: str
    ) -> list[GroundTruthOccurrence]:
        candidates: list[GroundTruthOccurrence] = []
        for entity in self.entity_columns:
            entity_values = self._coerce_entity_values(self._get_row_value(row=row, key=entity))
            for value in entity_values:
                for start, end in self._find_value_spans(text=original_text, value=value):
                    candidates.append(
                        GroundTruthOccurrence(
                            start=start,
                            end=end,
                            entity=entity,
                            value=value,
                        )
                    )
        return self._resolve_gt_overlaps(candidates)

    def _resolve_gt_overlaps(
        self, candidates: Sequence[GroundTruthOccurrence]
    ) -> list[GroundTruthOccurrence]:
        selected: list[GroundTruthOccurrence] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -(item.end - item.start),
                self._entity_order.get(item.entity, 10**6),
                item.start,
            ),
        ):
            if any(_span_contains(existing, candidate) for existing in selected):
                continue
            selected = [
                existing
                for existing in selected
                if not _span_contains(candidate, existing)
            ]
            selected.append(candidate)
        return sorted(selected, key=lambda item: (item.start, item.end, item.entity))

    @staticmethod
    def _find_value_spans(*, text: str, value: str) -> list[tuple[int, int]]:
        if not value:
            return []
        spans: list[tuple[int, int]] = []
        for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            if _is_boundary_aware_match(text=text, start=start, end=end):
                spans.append((start, end))
        return spans

    @staticmethod
    def _best_event_match(
        *,
        original_text: str,
        target_start: int,
        target_end: int,
        redaction_events: Sequence[RedactionEvent],
    ) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_score = 0.0
        for idx, event in enumerate(redaction_events):
            iou = _span_iou(
                text=original_text,
                a_start=target_start,
                a_end=target_end,
                b_start=event.start,
                b_end=event.end,
            )
            # Allow one broader mask (e.g. "<PERSON>" over "Sam Wilson")
            # to satisfy multiple atomic GT values ("Sam", "Wilson").
            coverage = _span_coverage_on_target(
                text=original_text,
                target_start=target_start,
                target_end=target_end,
                event_start=event.start,
                event_end=event.end,
            )
            score = max(iou, coverage)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def _build_alias_map(
        self, entity_aliases: Mapping[str, Iterable[str]] | None
    ) -> dict[str, set[str]]:
        alias_map: dict[str, set[str]] = {}
        for entity in self.entity_columns:
            aliases = {entity}
            if entity_aliases is not None and entity in entity_aliases:
                aliases.update(entity_aliases[entity])
            alias_map[entity] = {alias.strip().lower() for alias in aliases if alias}
        return alias_map

    def _build_reverse_alias_map(self) -> dict[str, str]:
        reverse: dict[str, str] = {}
        for entity, aliases in self._entity_aliases.items():
            for alias in aliases:
                reverse[alias] = entity
        return reverse

    @staticmethod
    def _iter_rows(data: object) -> Iterable[tuple[object, object]]:
        if hasattr(data, "iterrows") and callable(getattr(data, "iterrows")):
            yield from data.iterrows()  # type: ignore[misc]
            return

        if isinstance(data, Iterable):
            for idx, row in enumerate(data):
                yield idx, row
            return

        raise TypeError(
            "data must be an iterable of row mappings or an object exposing iterrows()."
        )

    @staticmethod
    def _get_row_value(*, row: object, key: str) -> object:
        if hasattr(row, "__getitem__"):
            try:
                return row[key]  # type: ignore[index]
            except Exception:
                pass
        if isinstance(row, Mapping):
            return row.get(key)
        raise KeyError(f"Could not read column {key!r} from row of type {type(row)!r}.")

    @staticmethod
    def _as_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _coerce_entity_values(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, Iterable):
            values: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    values.append(text)
            return values
        text = str(value).strip()
        return [text] if text else []


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass(frozen=True)
class CountsSnapshot:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    fbeta_score: float

    @classmethod
    def from_counts(cls, counts: Counts, *, beta: float) -> CountsSnapshot:
        precision = _safe_div(counts.tp, counts.tp + counts.fp)
        recall = _safe_div(counts.tp, counts.tp + counts.fn)
        beta_sq = beta * beta
        fbeta = _safe_div(
            (1 + beta_sq) * precision * recall, (beta_sq * precision) + recall
        )
        return cls(
            tp=counts.tp,
            fp=counts.fp,
            fn=counts.fn,
            precision=precision,
            recall=recall,
            fbeta_score=fbeta,
        )


@dataclass(frozen=True)
class EvaluationResults:
    overall: CountsSnapshot
    per_entity: dict[str, CountsSnapshot]
    warnings: list[str]
    evaluated_rows: int
    skipped_rows: int

    def summary(self) -> dict[str, object]:
        return {
            "overall": self.overall.__dict__,
            "per_entity": {
                entity: metrics.__dict__ for entity, metrics in self.per_entity.items()
            },
            "evaluated_rows": self.evaluated_rows,
            "skipped_rows": self.skipped_rows,
        }

    def get_warnings(self) -> list[str]:
        return list(self.warnings)


def _count_non_space_chars(*, text: str, start: int, end: int) -> int:
    return sum(1 for char in text[start:end] if not char.isspace())


def _span_iou(
    *,
    text: str,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = _count_non_space_chars(text=text, start=inter_start, end=inter_end)
    a_size = _count_non_space_chars(text=text, start=a_start, end=a_end)
    b_size = _count_non_space_chars(text=text, start=b_start, end=b_end)
    union = a_size + b_size - intersection
    return _safe_div(intersection, union)


def _span_coverage_on_target(
    *,
    text: str,
    target_start: int,
    target_end: int,
    event_start: int,
    event_end: int,
) -> float:
    inter_start = max(target_start, event_start)
    inter_end = min(target_end, event_end)
    intersection = _count_non_space_chars(text=text, start=inter_start, end=inter_end)
    target_size = _count_non_space_chars(text=text, start=target_start, end=target_end)
    return _safe_div(intersection, target_size)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _opcode_touches_tag(*, j1: int, j2: int, tag_start: int, tag_end: int) -> bool:
    # Delete opcodes have zero-width redacted span; treat their cursor position as a point.
    if j1 == j2:
        return tag_start <= j1 <= tag_end
    return not (j2 <= tag_start or j1 >= tag_end)


def _span_contains(
    outer: GroundTruthOccurrence,
    inner: GroundTruthOccurrence,
) -> bool:
    return outer.start <= inner.start and outer.end >= inner.end


def _is_boundary_aware_match(*, text: str, start: int, end: int) -> bool:
    return _is_left_boundary(text=text, start=start) and _is_right_boundary(text=text, end=end)

_JOINERS = {".", "_", "-", "@", "%", "+"}

def _is_left_boundary(*, text: str, start: int) -> bool:
    if start == 0:
        return True
    prev_char = text[start - 1]
    if _is_word_char(prev_char):
        return False
    # Internal joiner pattern (e.g. foo.bar, foo_bar, foo-bar, foo@bar)
    if prev_char in _JOINERS and start - 2 >= 0 and _is_word_char(text[start - 2]):
        return False
    return True


def _is_right_boundary(*, text: str, end: int) -> bool:
    if end >= len(text):
        return True
    next_char = text[end]
    if _is_word_char(next_char):
        return False
    # Internal joiner pattern (e.g. foo.bar, foo_bar, foo-bar, foo@bar)
    if next_char in _JOINERS and end + 1 < len(text) and _is_word_char(text[end + 1]):
        return False
    return True


def _is_word_char(char: str) -> bool:
    return char.isalnum()
