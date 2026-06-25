## Project Overview: Agnostic Redaction Evaluator

A Python package designed to evaluate PII redaction frameworks directly from a Pandas DataFrame. By shifting the evaluation from complex character-span metadata to raw text comparison, this package allows teams to seamlessly compare redaction quality across multiple redaction frameworks, and monitor the performance of production redaction systems.

---

## 1. Input Data Structure

The package accepts a standard Pandas DataFrame containing a minimum of $2 + n$ columns:

* **`original_text` (Column 1):** The raw, unredacted text.
* **`redacted_text` (Column 2):** The text processed by the redaction system using labeled tags (e.g., `<PERSON>`, `<ADDRESS>`).
* **PII Ground-Truth Columns ($n$ columns):** Columns containing the expected target values that should have been redacted (e.g., lists or strings of names, locations, dates).

---

## 2. Core Architecture & Processing Pipeline

The execution loop processes the DataFrame row-by-row using a localized, sentence-level alignment strategy to ensure maximum accuracy and performance.

```
[Row Input] ──> [Sentence Segmentation (. ! ?)]
                       │
                       ▼
         [Check: Segment Counts Equal?]
           ├── NO  ──> [Log Warning & Skip Row]
           └── YES ──> [Loop through Sentence Pairs]
                             │
                             ▼
                     [difflib Alignment] ──> [Local Metrics Calculation]
                                                       │
                                                       ▼
                                            [Sum Metrics Globally]

```

### Step 1: Sentence-Level Segmentation

To prevent global alignment drift and coordinate shifting over long texts, both the `original_text` and `redacted_text` are split into sentences using strict structural punctuation thresholds (`.`, `?`, `!`).

* **Implementation:** `re.split(r'(?<=[.!?])\s+', text)`

### Step 2: Structural Validation (The Safety Valve)

Before running the alignment, the engine verifies that the structural layout matches.

* **Condition:** `if len(orig_sentences) != len(red_sentences):`
* **Action:** Skip evaluation for this row and append a structured warning to the execution log alerting the user that the redaction engine corrupted sentence boundaries (e.g., swallowed punctuation or hallucinated text).

### Step 3: Localized Sequence Alignment

For validated rows, the engine iterates through matched sentence pairs. It uses Python's `difflib.SequenceMatcher` to analyze changes locally.

* The `get_opcodes()` method flags exactly which segments of the original sentence were replaced by a redaction tag in the redacted sentence.

### Step 4: Ground-Truth Mapping & Metric Evaluation

* The engine checks if the replaced text matches the values present in the $n$ ground-truth columns for that row.
* Matches are evaluated using a classification matrix to determine **True Positives (TP)**, **False Positives (FP)**, and **False Negatives (FN)**.
* **Configurable Overlap Threshold:** Introduces an `iou_threshold` parameter to determine if a partial redaction (e.g., redacting only the last name) counts as a success or failure.

---

## 3. Metrics & Outputs

Instead of scoring rows in isolation, the package aggregates local counts to output standard data science performance metrics globally or broken down by PII entity type:

* **Precision:** $\frac{TP}{TP + FP}$ (Measures over-redaction / false alarms)
* **Recall:** $\frac{TP}{TP + FN}$ (Measures under-redaction / leaked PII)
* **F-beta ($F_\beta$):** Highly configurable, defaulting to $F_2$ to heavily penalize leaked PII over false alarms.

---

## 4. Initial API Design Draft

```python
from redaction_evaluator import RedactionEvaluator

# 1. Initialize the evaluator with configuration
evaluator = RedactionEvaluator(
    df=df,
    original_col="original_text",
    redacted_col="redacted_text",
    pii_cols=["Names", "Addresses", "Phone_Numbers"],
    iou_threshold=0.8  # Require 80% token overlap for partial matches
)

# 2. Run the evaluation pipeline
results = evaluator.evaluate(beta=2)

# 3. Access global metrics
print(results.summary()) 
# Outputs: Precision, Recall, F2 overall and per column

# 4. Check for structural anomalies in production data
warnings = results.get_warnings()
if warnings:
    print(f"Skipped {len(warnings)} rows due to sentence mismatch.")

```

---

## 5. Development Milestones

1. **Phase 1:** Build the core utility class for regex sentence segmentation and structural validation checks.
2. **Phase 2:** Implement the localized `difflib` opcode extraction loop and map string replaces back to ground-truth arrays.
3. **Phase 3:** Create the metrics aggregation engine ($F_\beta$ math) and integrate the `iou_threshold` configuration.
4. **Phase 4:** Package the code, write unit tests for the edge cases discussed (e.g., model swallowing punctuation), and prep for open-source distribution.