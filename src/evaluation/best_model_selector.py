from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BestModelSelectionResult:
    best_model_name: str
    selection_strategy: str
    selection_metric_priority: list[str]
    metrics: dict[str, Any]


def select_best_model(
    comparison_results: list[dict[str, Any]],
) -> BestModelSelectionResult:
    if not comparison_results:
        raise ValueError("comparison_results cannot be empty")

    selection_metric_priority = [
        "selected_false_negative",
        "selected_recall",
        "selected_f1",
        "selected_precision",
        "selected_roc_auc",
    ]

    best_result = sorted(
        comparison_results,
        key=lambda result: (
            result["selected_false_negative"],
            -result["selected_recall"],
            -result["selected_f1"],
            -result["selected_precision"],
            -result["selected_roc_auc"],
        ),
    )[0]

    return BestModelSelectionResult(
        best_model_name=best_result["model_name"],
        selection_strategy=(
            "minimize selected_false_negative, then maximize selected_recall, "
            "selected_f1, selected_precision and selected_roc_auc"
        ),
        selection_metric_priority=selection_metric_priority,
        metrics=best_result,
    )