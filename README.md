# RedactEval

An evaluation and diagnostics tool for text redaction quality.

## Installation
```bash
pip install redacteval
```

## Usage
- Create a `RedactionEvaluator` instance with the following parameters:
  - `original_text_column`: the column containing original text
  - `entity_columns`: canonical ground-truth entity columns (e.g. `person`, `email`)
  - `entity_aliases`: optional aliases per canonical entity (to match tag variants)
  - `coverage_threshold`: minimum GT-coverage threshold for a valid match
  - `strict_entity_matching`: whether redaction tag type must match entity type
  - `segmenter`: optional custom sentence segmenter

- Evaluate one or more redacted outputs with the `evaluate` method:  
```python
evaluator = RedactionEvaluator(
    original_text_column="original_text",
    entity_columns=["person", "address", "email", "phone_number"],
    entity_aliases={
        "person": ["person", "first_name", "last_name", "name"],
        "address": ["address", "location", "place"],
        "email": ["email", "email_address"],
        "phone_number": ["phone_number", "mobile_number"],
    },
    coverage_threshold=0.8,
    strict_entity_matching=True,
)

results_a = evaluator.evaluate(df, redacted_text_column="redacted_framework_a", beta=2)
results_b = evaluator.evaluate(df, redacted_text_column="redacted_framework_b", beta=2)

print(results_a.summary())
warnings = results_a.get_warnings()
```
- `evaluate` returns an `EvaluationResults` object:
  - `summary()` gives global and per-entity metrics (`precision`, `recall`, `fbeta_score`, `tp`, `fp`, `fn`)
  - `get_warnings()` returns row-level warnings (for example sentence-structure mismatches)

- Prepare the data for evaluation:
  - Entity columns can contain a string, a list of strings, or `None`.
  - Example: `email` can hold `["xavier@gmail.com", "amanda_c@hotmail.com"]` for a single row.

### Ground-Truth Name Input Requirement
For person names, provide **atomic name elements** separately in ground-truth data.

- Recommended:
  - `person = ["Sam", "Wilson"]`
  - `person = ["Jane", "Mary", "Smith"]` (first/middle/last separately)
- Avoid providing only one combined full-name string when you want per-token scoring:
  - `person = "Sam Wilson"` (this is treated as one occurrence, not two)

Example row:

```python
{
    "original_text": "Sam Wilson or Jane Smith can help you with the issue.",
    "person": ["Sam", "Wilson", "Jane", "Smith"],
    "redacted_framework_a": "<PERSON> or <FIRST_NAME> <LAST_NAME> can help you with the issue.",
}
```

Notes:
- A combined mask over multiple atomic values (for example `<PERSON>` over `"Sam Wilson"`) is considered correct when overlap requirements are met.
- Name elements are matched as standalone occurrences in context (so substrings inside larger tokens such as emails are not counted as separate person occurrences).

## Demo Data
You can start with bundled sample data:

```python
from redacteval import RedactionEvaluator, load_demo_data, print_report

df = load_demo_data()
evaluator = RedactionEvaluator(
    original_text_column="original_text",
    entity_columns=["name", "email", "phone_number"],
    entity_aliases={
        "name": ["name", "person", "first_name", "last_name"],
        "email": ["email", "email_address"],
        "phone_number": ["phone_number", "mobile_number"],
    },
)

results = evaluator.evaluate(df, redacted_text_column="redacted_framework_a")
print_report(results)
```

A runnable notebook is available at `examples/demo.ipynb`.

## License
This project is licensed under the Apache License 2.0.
See `LICENSE` for details.

## Contributing
Contributions are welcome.
Please read `CONTRIBUTING.md` before opening a pull request.