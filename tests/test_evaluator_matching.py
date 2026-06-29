import pytest

from redacteval import RedactionEvaluator, load_demo_data


def test_evaluate_counts_mislabeled_tag_as_fn_only_when_strict() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "address"],
        strict_entity_matching=True,
    )
    data = [
        {
            "original_text": "John moved yesterday.",
            "redacted_framework_a": "<ADDRESS> moved yesterday.",
            "person": "John",
            "address": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 1
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["overall"]["tp"] == 0
    assert summary["overall"]["fn"] == 1
    assert summary["overall"]["fp"] == 0


def test_evaluate_counts_mislabeled_tag_as_tp_when_not_strict() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "address"],
        strict_entity_matching=False,
    )
    data = [
        {
            "original_text": "John moved yesterday.",
            "redacted_framework_a": "<ADDRESS> moved yesterday.",
            "person": "John",
            "address": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 1
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["overall"]["tp"] == 1
    assert summary["overall"]["fn"] == 0
    assert summary["overall"]["fp"] == 0


def test_evaluate_uses_coverage_threshold_for_partial_overlap() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["address", "person"],
        coverage_threshold=0.8,
        strict_entity_matching=True,
    )
    data = [
        {
            "original_text": "Lives at 33 Mont Albert Road.",
            "redacted_framework_a": "Lives at 33 <PERSON> Road.",
            "address": "33 Mont Albert Road",
            "person": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["address"]["tp"] == 0
    assert summary["per_entity"]["address"]["fn"] == 1
    assert summary["per_entity"]["person"]["fp"] == 1
    assert summary["overall"]["tp"] == 0
    assert summary["overall"]["fn"] == 1
    assert summary["overall"]["fp"] == 1


def test_evaluate_uses_coverage_threshold_for_partial_overlap_when_not_strict() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["address", "person"],
        coverage_threshold=0.5,
        strict_entity_matching=False,
    )
    data = [
        {
            "original_text": "Lives at 33 Mont Albert Road.",
            "redacted_framework_a": "Lives at 33 <PERSON> Road.",
            "address": "33 Mont Albert Road",
            "person": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["address"]["tp"] == 1
    assert summary["per_entity"]["address"]["fn"] == 0
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["overall"]["tp"] == 1
    assert summary["overall"]["fn"] == 0
    assert summary["overall"]["fp"] == 0


def test_evaluate_uses_coverage_threshold_for_partial_overlap_when_strict() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["address", "person"],
        coverage_threshold=0.5,
        strict_entity_matching=True,
    )
    data = [
        {
            "original_text": "Lives at 33 Mont Albert Road.",
            "redacted_framework_a": "Lives at 33 <PERSON> Road.",
            "address": "33 Mont Albert Road",
            "person": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["address"]["tp"] == 0
    assert summary["per_entity"]["address"]["fn"] == 1
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["overall"]["tp"] == 0
    assert summary["overall"]["fn"] == 1
    assert summary["overall"]["fp"] == 0


def test_evaluate_counts_unmatched_redaction_tags_as_fp() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        strict_entity_matching=True,
    )
    data = [
        {
            "original_text": "John works.",
            "redacted_framework_a": "<PERSON> works at <EMAIL>.",
            "person": "John",
            "email": None,
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 1
    assert summary["per_entity"]["email"]["fp"] == 1
    assert summary["overall"]["tp"] == 1
    assert summary["overall"]["fp"] == 1


def test_adjacent_name_tags_score_no_spurious_false_positive() -> None:
    # Regression test for the case-sensitive alignment fix: with all-uppercase
    # tags, "Liam" used to collapse onto the 'L' in "LAST_NAME", projecting
    # <FIRST_NAME> to a zero-width span that was then scored as a spurious FP.
    # Masking tag spans before diffing yields exactly two TP and no FP.
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person"],
        entity_aliases={"person": ["person", "first_name", "last_name", "name"]},
        strict_entity_matching=False,
        coverage_threshold=0.8,
    )
    data = [
        {
            "original_text": "Meet Liam Sam now.",
            "redacted_framework_a": "Meet <FIRST_NAME> <LAST_NAME> now.",
            "person": ["Liam", "Sam"],
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 2
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0


def test_evaluate_overlapping_entities_with_strict_coverage_threshold() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        strict_entity_matching=False,
        coverage_threshold=0.8,
    )
    data = [
        {
            "original_text": "Hi team. Please reach out to John regarding the account update. You can email him at john.doe@example.com to coordinate the kickoff meeting.",
            "redacted_framework_a": "Hi team. Please reach out to <PERSON> regarding the account update. You can email him at <PERSON>.doe@example.com to coordinate the kickoff meeting.",
            "person": "John",
            "email": "john.doe@example.com",
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 1
    assert summary["per_entity"]["person"]["fp"] == 1
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["email"]["tp"] == 0
    assert summary["per_entity"]["email"]["fp"] == 0
    assert summary["per_entity"]["email"]["fn"] == 1
    assert summary["overall"]["tp"] == 1
    assert summary["overall"]["fp"] == 1
    assert summary["overall"]["fn"] == 1


def test_evaluate_overlapping_entities_with_non_strict_coverage_threshold() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        strict_entity_matching=False,
        coverage_threshold=0.1,
    )
    data = [
        {
            "original_text": "Hi team. Please reach out to John regarding the account update. You can email him at john.doe@example.com to coordinate the kickoff meeting.",
            "redacted_framework_a": "Hi team. Please reach out to <PERSON> regarding the account update. You can email him at <PERSON>.doe@example.com to coordinate the kickoff meeting.",
            "person": "John",
            "email": "john.doe@example.com",
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 1
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["email"]["tp"] == 1
    assert summary["per_entity"]["email"]["fp"] == 0
    assert summary["per_entity"]["email"]["fn"] == 0
    assert summary["overall"]["tp"] == 2
    assert summary["overall"]["fp"] == 0
    assert summary["overall"]["fn"] == 0


def test_evaluate_first_and_last_name_entities_seperate() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        entity_aliases={"person": ["first_name", "last_name", "person"]},
        strict_entity_matching=False,
    )
    data = [
        {
            "original_text": "Thank you for your inquiry. I have forwarded your request to sam_wilson@techcorp.com. Sam Wilson from our engineering team will review it shortly. ",
            "redacted_framework_a": "Thank you for your inquiry. I have forwarded your request to <email>. <FIRST_NAME> <LAST_NAME> from our engineering team will review it shortly. ",
            "person": ["Sam", "Wilson", "Jane"],
            "email": "sam_wilson@techcorp.com",
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 2
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["email"]["tp"] == 1
    assert summary["per_entity"]["email"]["fp"] == 0
    assert summary["per_entity"]["email"]["fn"] == 0
    assert summary["overall"]["tp"] == 3
    assert summary["overall"]["fp"] == 0
    assert summary["overall"]["fn"] == 0


@pytest.mark.parametrize(
    "first_tag, last_tag",
    [
        ("<FIRST_NAME>", "<LAST_NAME>"),
        ("<first_name>", "<last_name>"),
        ("<FIRST_NAME>", "<last_name>"),
        ("<first_name>", "<LAST_NAME>"),
    ],
)
def test_evaluate_adjacent_name_tags_are_case_insensitive(first_tag, last_tag) -> None:
    # Regression test: alias matching is case-insensitive by design, so tag
    # casing must not change scoring. Previously lowercase/mixed-case tags
    # corrupted the difflib span projection (only the all-uppercase form
    # happened to score correctly).
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person"],
        entity_aliases={"person": ["first_name", "last_name", "person", "name"]},
        strict_entity_matching=False,
    )
    data = [
        {
            "original_text": "Our primary contact for the Sydney office is Jane Smith and she is friendly.",
            "redacted_framework_a": (
                "Our primary contact for the Sydney office is "
                f"{first_tag} {last_tag} and she is friendly."
            ),
            "person": ["Jane", "Smith"],
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 2
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["overall"]["tp"] == 2
    assert summary["overall"]["fp"] == 0
    assert summary["overall"]["fn"] == 0


def test_evaluate_first_and_last_name_entities_combined() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        strict_entity_matching=False,
    )
    data = [
        {
            "original_text": "Thank you for your inquiry. I have forwarded your request to sam_wilson@techcorp.com. Sam Wilson from our engineering team or Jane will review it shortly. ",
            "redacted_framework_a": "Thank you for your inquiry. I have forwarded your request to <email>. <PERSON> from our engineering team or <PERSON> will review it shortly. ",
            "person": ["Sam", "Wilson", "Jane"],
            "email": "sam_wilson@techcorp.com",
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 3
    assert summary["per_entity"]["person"]["fp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["email"]["tp"] == 1
    assert summary["per_entity"]["email"]["fp"] == 0
    assert summary["per_entity"]["email"]["fn"] == 0
    assert summary["overall"]["tp"] == 4
    assert summary["overall"]["fp"] == 0
    assert summary["overall"]["fn"] == 0


def test_name_inside_email_is_not_counted_as_standalone_name() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person", "email"],
        strict_entity_matching=True,
        coverage_threshold=0.5,
    )
    data = [
        {
            "original_text": "Please email john.doe@example.com for updates.",
            "redacted_framework_a": "Please email <EMAIL> for updates.",
            "person": "John",
            "email": "john.doe@example.com",
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    # "John" only appears as part of email local-part, not as a standalone word.
    assert summary["per_entity"]["person"]["tp"] == 0
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["email"]["tp"] == 1
    assert summary["per_entity"]["email"]["fn"] == 0


def test_single_person_mask_can_match_multiple_atomic_name_values() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["person"],
        strict_entity_matching=False,
        coverage_threshold=0.8,
    )
    data = [
        {
            "original_text": "Sam Wilson joined the meeting.",
            "redacted_framework_a": "<PERSON> joined the meeting.",
            "person": ["Sam", "Wilson"],
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["person"]["tp"] == 2
    assert summary["per_entity"]["person"]["fn"] == 0
    assert summary["per_entity"]["person"]["fp"] == 0


def test_single_mask_can_match_multiple_atomic_values_for_other_entities() -> None:
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["address"],
        strict_entity_matching=False,
        coverage_threshold=0.8,
    )
    data = [
        {
            "original_text": "Ship to 33 Mont Albert Road today.",
            "redacted_framework_a": "Ship to <ADDRESS> today.",
            "address": ["Mont", "Albert", "Road"],
        }
    ]

    summary = evaluator.evaluate(
        data, redacted_text_column="redacted_framework_a"
    ).summary()
    assert summary["per_entity"]["address"]["tp"] == 3
    assert summary["per_entity"]["address"]["fn"] == 0
    assert summary["per_entity"]["address"]["fp"] == 0


def test_evaluate_multi_sentence_demo_row_detects_all_framework_a_entities() -> None:
    records = load_demo_data(as_pandas=False)
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["name", "email", "phone_number"],
        entity_aliases={
            "name": ["name", "person", "first_name", "last_name"],
            "email": ["email", "email_address"],
            "phone_number": ["phone_number", "mobile_number"],
        },
        strict_entity_matching=True,
    )

    summary = evaluator.evaluate(
        [records[0]], redacted_text_column="redacted_framework_a"
    ).summary()

    assert summary["overall"]["fn"] == 0
    assert summary["per_entity"]["name"]["tp"] == 2
    assert summary["per_entity"]["email"]["tp"] == 2
    assert summary["per_entity"]["phone_number"]["tp"] == 0


def test_evaluate_multi_sentence_demo_dataset_framework_a_counts() -> None:
    # Full demo dataset (all rows). framework_a leaves "Andrew" unredacted in
    # row 1 (the original has "her assistant Andrew", the redaction keeps it),
    # so a single name false negative is the *correct* result -- not a bug. The
    # previously disabled version asserted name.fn == 0, which is wrong; email
    # and phone are fully redacted (fn == 0).
    df = load_demo_data()
    evaluator = RedactionEvaluator(
        original_text_column="original_text",
        entity_columns=["name", "email", "phone_number"],
        entity_aliases={
            "name": ["name", "person", "first_name", "last_name"],
            "email": ["email", "email_address"],
            "phone_number": ["phone_number", "mobile_number"],
        },
        strict_entity_matching=True,
    )

    summary = evaluator.evaluate(
        df, redacted_text_column="redacted_framework_a"
    ).summary()

    assert summary["evaluated_rows"] == 3
    assert summary["skipped_rows"] == 0
    assert summary["per_entity"]["name"]["fn"] == 1
    assert summary["per_entity"]["email"]["fn"] == 0
    assert summary["per_entity"]["phone_number"]["fn"] == 0
    assert summary["overall"]["fn"] == 1
