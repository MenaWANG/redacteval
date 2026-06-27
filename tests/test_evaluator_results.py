from redacteval import RedactionEvaluator


def test_evaluate_returns_summary_and_warnings_api() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        entity_aliases={"person": ["name"], "email": ["email_address"]},
        iou_threshold=0.7,
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

    results = evaluator.evaluate(data, redacted_text_column="redacted_framework_a", beta=2)
    summary = results.summary()

    assert summary["evaluated_rows"] == 1
    assert summary["skipped_rows"] == 0
    assert summary["overall"]["tp"] == 1
    assert summary["overall"]["fn"] == 1
    assert summary["per_entity"]["email"]["tp"] == 1
    assert summary["per_entity"]["person"]["fn"] == 1
    assert results.get_warnings() == []


def test_evaluate_records_warning_for_sentence_mismatch() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person"],
    )
    data = [
        {
            "original_text": "Alice went home. She slept.",
            "redacted_framework_a": "Alice went home she slept",
            "person": "Alice",
        }
    ]

    results = evaluator.evaluate(data, redacted_text_column="redacted_framework_a")
    summary = results.summary()
    warnings = results.get_warnings()

    assert summary["evaluated_rows"] == 0
    assert summary["skipped_rows"] == 1
    assert len(warnings) == 1
    assert "sentence mismatch" in warnings[0]
