from __future__ import annotations

from .evaluator import EvaluationResults


def format_report(results: EvaluationResults, *, precision: int = 3) -> str:
    """Return a human-friendly text report for evaluation results."""

    summary = results.summary()
    overall = summary["overall"]
    per_entity = summary["per_entity"]
    beta_label = _format_beta_label(beta=float(summary["beta"]))

    lines = [
        "=== Redaction Evaluation Report ===",
        f"Evaluated rows: {summary['evaluated_rows']}",
        f"Skipped rows:   {summary['skipped_rows']}",
        "",
        "Overall metrics",
        "---------------",
        f"TP: {overall['tp']} | FP: {overall['fp']} | FN: {overall['fn']}",
        f"Precision: {overall['precision']:.{precision}f}",
        f"Recall:    {overall['recall']:.{precision}f}",
        f"{beta_label}:    {overall['fbeta_score']:.{precision}f}",
        "",
        "Per-entity metrics",
        "------------------",
    ]

    for entity, metrics in per_entity.items():
        lines.append(entity)
        lines.append(
            f"  TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}"
        )
        lines.append(
            f"  Precision: {metrics['precision']:.{precision}f} | "
            f"Recall: {metrics['recall']:.{precision}f} | "
            f"{beta_label}: {metrics['fbeta_score']:.{precision}f}"
        )

    warnings = results.get_warnings()
    lines.append("")
    if warnings:
        lines.extend(["Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none")

    return "\n".join(lines)


def print_report(results: EvaluationResults, *, precision: int = 3) -> None:
    """Print a human-friendly report for evaluation results."""

    print(format_report(results, precision=precision))


def _format_beta_label(*, beta: float) -> str:
    if beta.is_integer():
        return f"F{int(beta)}"
    return f"F{beta:.1f}"
