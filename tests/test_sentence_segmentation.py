import pytest

from redacteval import RedactionEvaluator, SentenceSpan, regex_sentence_segmenter


def test_regex_sentence_segmenter_returns_sentence_spans() -> None:
    text = "Alice lives in Melbourne. She works at ACME! Is that right?"

    spans = regex_sentence_segmenter(text)

    assert [span.text for span in spans] == [
        "Alice lives in Melbourne.",
        "She works at ACME!",
        "Is that right?",
    ]
    assert [(span.start, span.end) for span in spans] == [(0, 25), (26, 44), (45, 59)]


def test_regex_sentence_segmenter_trims_surrounding_whitespace() -> None:
    text = "  First sentence.   Second sentence.  "

    spans = regex_sentence_segmenter(text)

    assert [span.text for span in spans] == ["First sentence.", "Second sentence."]
    assert [(span.start, span.end) for span in spans] == [(2, 17), (20, 36)]


def test_redaction_evaluator_uses_custom_segmenter() -> None:
    def custom_segmenter(text: str) -> list[SentenceSpan]:
        midpoint = text.index(";")
        return [
            SentenceSpan(text=text[:midpoint], start=0, end=midpoint),
            SentenceSpan(text=text[midpoint + 1 :], start=midpoint + 1, end=len(text)),
        ]

    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["name"],
        segmenter=custom_segmenter,
    )

    original_segments, redacted_segments = evaluator.segment_row(
        original_text="alpha;beta", redacted_text="one;two"
    )

    assert [segment.text for segment in original_segments] == ["alpha", "beta"]
    assert [segment.text for segment in redacted_segments] == ["one", "two"]


def test_redaction_evaluator_validates_segmenter_output_type() -> None:
    def invalid_segmenter(_: str) -> list[str]:
        return ["not-a-span"]

    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["name"],
        segmenter=invalid_segmenter,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="Sequence\\[SentenceSpan\\]"):
        evaluator.segment_row(original_text="alpha", redacted_text="beta")
