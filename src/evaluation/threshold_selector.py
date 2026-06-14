from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.report import ConfusionMatrixResult
from src.evaluation.threshold_analysis import ThresholdAnalysisResult

ThresholdSelectionStrategy = Literal["maximize_f1"]


@dataclass(frozen=True)
class SelectedThresholdResult:
    selected_threshold: float
    selection_strategy: ThresholdSelectionStrategy
    metrics: ClassificationMetrics
    confusion_matrix_result: ConfusionMatrixResult


def select_threshold(
    threshold_analysis_results: list[ThresholdAnalysisResult],
    strategy: ThresholdSelectionStrategy = "maximize_f1",
) -> SelectedThresholdResult:
    if not threshold_analysis_results:
        raise ValueError("threshold_analysis_results cannot be empty")

    if strategy == "maximize_f1":
        best_result = max(
            threshold_analysis_results,
            key=lambda result: result.metrics.f1,
        )

        return SelectedThresholdResult(
            selected_threshold=best_result.threshold,
            selection_strategy=strategy,
            metrics=best_result.metrics,
            confusion_matrix_result=best_result.confusion_matrix_result,
        )

    raise ValueError(f"Unsupported threshold selection strategy: {strategy}")
