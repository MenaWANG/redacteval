from redacteval import RedactionEvaluator, format_report


def _sample_results(*, beta: float = 2.0):
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        entity_aliases={"person": ["name"], "email": ["email_address"]},
        coverage_threshold=0.7,
        strict_entity_matching=True,
    )
    data = [
        {
            "original_text": "Alice email is xavier@example.com.",
            "redacted_framework_a": "Alice email is <EMAIL>.",
            "person": "Alice",
            "email": "xavier@example.com",
        }
    ]
    return evaluator.evaluate(data, redacted_text_column="redacted_framework_a", beta=beta)


def test_format_report_contains_core_sections() -> None:
    report = format_report(_sample_results())
    assert "=== Redaction Evaluation Report ===" in report
    assert "Overall metrics" in report
    assert "Per-entity metrics" in report
    assert "Warnings: none" in report
    assert "person" in report
    assert "email" in report


def test_format_report_respects_precision() -> None:
    report = format_report(_sample_results(), precision=2)
    assert "Precision: 1.00" in report
    assert "Recall:    0.50" in report


def test_format_report_shows_beta_in_fbeta_label() -> None:
    report = format_report(_sample_results(beta=2.0))
    assert "F2:" in report

    report_half = format_report(_sample_results(beta=0.5))
    assert "F0.5:" in report_half
