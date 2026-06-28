## Project Overview: Agnostic Redaction Evaluator

RedactEval is a Python package designed to evaluate redaction frameworks directly from comparing the original and redacted text based on the ground-truth entity values that should have been redacted. It is designed to be agnostic of the return format of the redaction framework, enabling users to comapre different redaction frameworks consistently, without complext requirement such as character-span annotations in the input dataset. 

---

## 1. Input Data and Configuration

### Input Data Contract

The evaluator accepts either:

- a Pandas DataFrame (`iterrows()` supported), or
- any iterable of row mappings (for example a list of dicts).

Each row should contain:

- `original_text`: unredacted source text.
- a chosen redacted text column (passed via `redacted_text_column` at evaluation time).
- one column per entity in `entity_columns` (values may be a string or list of strings).

### Key Configuration Parameters

- `iou_threshold` (`float`, default `0.8`): minimum overlap score for a match.
- `strict_entity_matching` (`bool`, default `True`):
  - `True`: tag entity type must match the ground-truth entity.
  - `False`: any valid recognized tag masking the span counts as redacted.
- `segmenter` (optional): custom function returning `Sequence[SentenceSpan]`.
- `entity_aliases` (optional): aliases used to normalize tag labels to canonical entities.

### Evaluation-Time Parameter

- `beta` (`float`, default `2.0`, must be `> 0`): controls F-beta weighting (`F_beta`).

---

## 2. Core Architecture and Processing Pipeline

Row-level execution:

```text
[Row Input]
   -> [Sentence segmentation for original + redacted]
   -> [Segment count equality check]
      -> mismatch: warning + row skipped
      -> match: continue
   -> [Extract redaction events from tags via difflib alignment]
   -> [Build/resolve ground-truth occurrences]
   -> [Best-match scoring (IoU/coverage) and TP/FP/FN updates]
   -> [Aggregate overall and per-entity counts]
```

### Step 1: Sentence Segmentation

By default, both texts are segmented by punctuation boundaries (`.`, `?`, `!`) using `regex_sentence_segmenter`. A custom segmenter can be injected.

### Step 2: Structural Validation

If original and redacted sentence counts differ for a row:

- the row is skipped,
- a warning is recorded,
- scoring for that row does not proceed.

### Step 3: Predicted-Event and Ground-Truth Construction

**Predicted side:**
--------------------------------
- [P1] extract tag tokens from redacted sentences (supported wrappers include `<...>`, `[...]`, `{...}`),
- [P2] align each tag back to an original-text span using `SequenceMatcher` opcodes,
- [P3] merge fragmented opcodes touching the same tag token,
- [P4] build predicted redaction events: `(start, end, tag_entities)`.

**Ground-truth side:**
--------------------------------

- [G1] find literal, case-insensitive occurrences for each entity value in `original_text`,
- [G2] enforce boundary-aware matching to avoid partial token noise,
- [G3] resolve overlaps so larger/more specific spans can own nested spans (e.g., John.Doe@example.com belongs to EMAIL, not PERSON),
- [G4] produce finalized scoring targets.

**Join and score:**
--------------------------------

- [J1] each GT occurrence is matched to the best predicted event by score (`max(IoU, target_coverage)`),
- [J2] below-threshold or missing match -> FN,
- [J3] If matched:
     - strict=True: TP only if entity type matches, else FN (on the GT entity)
     - strict=False: TP if any valid redaction tag exists
- [J4] Unmatched predicted events -> FP (over-redaction)
- [J5] Aggregate TP/FP/FN globally and per entity

**Known issue to fix:**
--------------------------------
- There is a reproducible edge case with adjacent person tags where mixed casing can change matching behavior unexpectedly.
- Reference test: `tests/test_evaluator_matching.py` (`test_evaluate_first_and_last_name_entities_small_letter_fail`).
- Symptom:
  - `<FIRST_NAME> <LAST_NAME>` may pass
  - `<FIRST_NAME> <last_name>` may fail in some sentence contexts
- This is not intended behavior because entity alias matching is case-insensitive by design.
- Suspected root cause is context-sensitive difflib span projection around adjacent tags (Predicted Side [P3]).
- Action: keep this documented and address in a dedicated matching-stability fix.

### Step 4: Metrics

Metrics are computed per entity and overall from aggregated counts:

- Precision = `TP / (TP + FP)`
- Recall = `TP / (TP + FN)`
- F-beta = `((1 + beta^2) * P * R) / ((beta^2 * P) + R)`
   - Highly configurable, defaulting to $F_2$ to heavily penalize leaked sensitive values over false alarms.

Report output includes the explicit beta label (for example `F2` or `F0.5`).

### Strict Matching Logic Matrix

Matching: the string matching for the entity_mapping values format-agnostic internally.For example, if a user specifies "first_name", your underlying engine could automatically match <first_name>, <FIRST_NAME>, [FIRST_NAME], or {FIRST_NAME}. This saves the user from having to explicitly write out the brackets for every single vendor variation.

| Scenario | IoU/Coverage >= Threshold | Tag matches canonical entity? | `strict_entity_matching=True` | `strict_entity_matching=False` |
| :--- | :--- | :--- | :--- | :--- |
| **Perfect Match** | Yes | Yes | **True Point (TP)** | **True Point (TP)** |
| **Mislabeled Tag** | Yes | No | **False Negative (FN)** | **True Point (TP)** |
| **Insufficient Overlap** | No | Don't Care | **False Negative (FN)** | **False Negative (FN)** |

### Scoring Unit and Example

Scoring unit is **per ground-truth occurrence**.

- Original: `John Doe is a student of La Trobe University. John loves basketball.`
- Redacted: `<PERSON> is a student of <PERSON> University. John loves basketball.`
- If `person = ["John", "Doe"]`, this can produce:
  - TP for the 1st masked person occurrence,
  - FP for the 2nd masked person occurrence (over-redacted non-PII span),
  - FN for missed person occurrence in the 2nd sentence.

**IoU threshold & entity matching example:**

- Ground truth: `33 Mont Albert Road` which is an address entity in the ground-truth data.
- Redacted text: `33 <PERSON> Road` replacing only `Mont Albert` with the inconsistent tag <PERSON>
- IoU score depends on non-whitespace overlap ratio which is (4+6)/(2+4+6+4) = 0.625
- **Scenario 1**: When IoU threshold > 0.625 (i.e., the redaction didn't meet the IoU threshold)
    - This instance is a FN for the `address` entity, because the original `address` entity is not adequately redacted based on the IoU threshold.
- **Scenario 2**: When IoU threshold <= 0.625 (i.e., the redaction met the IoU threshold)
    - **Scenario 2.1**: When strict_entity_matching = False, this instance is a TP for the `address` entity.
    - **Scenario 2.2**: When strict_entity_matching = True
        - this instance is still a FN for the `address` entity, becasue although it is covered, it is not recognized as the correct entity type.

---

## 3. Current API Snapshot

```python
from redacteval import RedactionEvaluator, format_report, load_demo_data

data = load_demo_data()  # pandas DataFrame by default

evaluator = RedactionEvaluator(
    original_text_column="original_text",
    entity_columns=["name", "email", "phone_number"],
    entity_aliases={
        "name": ["name", "person", "first_name", "last_name"],
        "email": ["email", "email_address"],
        "phone_number": ["phone_number", "mobile_number"],
    },
    iou_threshold=0.8,
    strict_entity_matching=True,
)

results = evaluator.evaluate(
    data,
    redacted_text_column="redacted_framework_a",
    beta=2.0,
)

print(results.summary())        # overall + per-entity metrics + beta
print(results.get_warnings())   # row-level structural warnings
print(format_report(results))   # human-readable report
```

---

## 4. Contributor Notes

- Keep behavior changes covered by tests in `tests/`.
- Add focused cases for overlap handling, entity aliasing, and segmentation mismatch.
- For API-facing changes, update `README.md` and this design doc in the same PR.
