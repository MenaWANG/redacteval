import numpy as np
import pandas as pd

from redacteval import RedactionEvaluator


def test_coerce_entity_values_treats_missing_markers_as_empty() -> None:
    # None and every pandas/NumPy missing marker must coerce to "no value",
    # not to a literal phantom string like "nan"/"<NA>"/"NaT".
    assert RedactionEvaluator._coerce_entity_values(None) == []
    assert RedactionEvaluator._coerce_entity_values(np.nan) == []
    assert RedactionEvaluator._coerce_entity_values(float("nan")) == []
    assert RedactionEvaluator._coerce_entity_values(pd.NA) == []
    assert RedactionEvaluator._coerce_entity_values(pd.NaT) == []
    # Missing markers inside a list are skipped, real values kept.
    assert RedactionEvaluator._coerce_entity_values(
        ["Sam", np.nan, None, "Wilson"]
    ) == [
        "Sam",
        "Wilson",
    ]


def test_coerce_entity_values_renders_integral_numbers_without_trailing_zero() -> None:
    # A numeric column promoted to float64 by a missing cell must still match the
    # digits in the text (no ".0" suffix).
    assert RedactionEvaluator._coerce_entity_values(412345678.0) == ["412345678"]
    assert RedactionEvaluator._coerce_entity_values(np.int64(412345678)) == [
        "412345678"
    ]


def test_as_text_treats_missing_text_as_empty_string() -> None:
    assert RedactionEvaluator._as_text(np.nan) == ""
    assert RedactionEvaluator._as_text(None) == ""
    assert RedactionEvaluator._as_text("real text") == "real text"


def test_missing_entity_value_does_not_create_phantom_ground_truth() -> None:
    # A missing entity in a row whose text happens to contain the substring
    # "nan" must not become a phantom 'nan' ground-truth occurrence.
    df = pd.DataFrame(
        {
            "original_text": ["The plan covers nanotech topics."],
            "redacted_framework_a": ["The plan covers nanotech topics."],
            "person": pd.array([pd.NA], dtype="string"),
        }
    )
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person"],
        strict_entity_matching=False,
    )
    summary = evaluator.evaluate(
        df, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 0
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0


def test_numeric_entity_column_is_matched_not_flagged_as_over_redaction() -> None:
    # phone_number stored as float64 (a missing cell elsewhere promoted the
    # column) must still be found in the text and scored as a true positive,
    # not as a false-positive over-redaction.
    df = pd.DataFrame(
        {
            "original_text": ["Call 412345678 now.", "Nothing here."],
            "redacted_framework_a": ["Call <PHONE_NUMBER> now.", "Nothing here."],
            "phone_number": [412345678, np.nan],
        }
    )
    assert df["phone_number"].dtype == np.float64  # missing cell promoted to float
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["phone_number"],
        strict_entity_matching=False,
    )
    summary = evaluator.evaluate(
        df, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["phone_number"]["tp"] == 1
    assert summary["per_entity"]["phone_number"]["fp"] == 0
    assert summary["per_entity"]["phone_number"]["fn"] == 0
