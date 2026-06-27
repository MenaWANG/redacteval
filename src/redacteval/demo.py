from __future__ import annotations

from typing import Any


def _demo_data_dict() -> dict[str, list[Any]]:
    return {
        "original_text": [
            (
                "Hello team. Please reach out to John Doe regarding the account update. "
                "You can email him at john.doe@example.com or info@example.com to coordinate "
                "the kickoff meeting."
            ),
            (
                "Our primary contact for the Sydney office is Jane Smith. Her direct line is "
                "0412345678 if you need urgent assistance. Or you can also contact her assistant "
                "Andrew in her absence."
            ),
            (
                "Thank you for your inquiry. I have forwarded your request to "
                "sam_wilson@techcorp.com. Sam Wilson from our engineering team will review it "
                "shortly. You can also contact him via his number 0491570156 or his Emily "
                "assistant via 0423999888."
            ),
        ],
        "name": [
            ["John", "Doe"],
            ["Jane", "Smith", "Andrew"],
            ["Sam", "Wilson", "Emily"],
        ],
        "email": [
            ["john.doe@example.com", "info@example.com"],
            None,
            "sam_wilson@techcorp.com",
        ],
        "phone_number": [
            None,
            "0412345678",
            ["0491570156", "0423999888"],
        ],
        "redacted_framework_a": [
            (
                "Hello team. Please reach out to <person> regarding the account update. You can "
                "email him at <email> or <email_address> to coordinate the kickoff meeting."
            ),
            (
                "Our primary contact for the Sydney office is <FIRST_NAME> <LAST_NAME>. Her "
                "direct line is <phone_number> if you need urgent assistance. Or you can also "
                "contact her assistant Andrew in her absence."
            ),
            (
                "Thank you for your inquiry. I have forwarded your request to <email>. <name> "
                "from our engineering team will review it shortly. You can also contact him via "
                "his number <mobile_number> or his <person> assistant via <phone_number>."
            ),
        ],
        "redacted_framework_b": [
            (
                "Hello team. Please reach out to [REDACTED] regarding the account update. You can "
                "email him at [REDACTED] or [REDACTED] to coordinate the kickoff meeting."
            ),
            (
                "Our primary contact for the Sydney office is [REDACTED]. Her direct line is "
                "[REDACTED] if you need urgent assistance. Or you can also contact her assistant "
                "[REDACTED] in her absence."
            ),
            (
                "Thank you for your inquiry. I have forwarded your request to [REDACTED]. "
                "[REDACTED] from our engineering team will review it shortly. You can also contact "
                "him via his number [REDACTED] or his [REDACTED] assistant via [REDACTED]."
            ),
        ],
    }


def load_demo_data(*, as_pandas: bool = True) -> Any:
    """Load bundled demo data for quick evaluator experiments.

    Args:
        as_pandas: When True, return a pandas DataFrame. When False, return
            a list of row dictionaries.
    """

    data = _demo_data_dict()
    if as_pandas:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas is required for as_pandas=True. "
                "Install pandas or call load_demo_data(as_pandas=False)."
            ) from exc
        return pd.DataFrame(data)

    row_count = len(data["original_text"])
    return [{key: values[idx] for key, values in data.items()} for idx in range(row_count)]
