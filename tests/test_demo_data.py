from redacteval import load_demo_data


def test_load_demo_data_returns_records_when_requested() -> None:
    records = load_demo_data(as_pandas=False)
    assert isinstance(records, list)
    assert len(records) == 3
    assert records[0]["name"] == ["John", "Doe"]
    assert "redacted_framework_a" in records[0]


def test_load_demo_data_returns_dataframe_by_default() -> None:
    df = load_demo_data()
    assert list(df.columns) == [
        "original_text",
        "name",
        "email",
        "phone_number",
        "redacted_framework_a",
        "redacted_framework_b",
    ]
    assert len(df) == 3
