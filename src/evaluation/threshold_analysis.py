from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from src.evaluation.report import (
    ConfusionMatrixResult,
    calculate_confusion_matrix_result,
)


@dataclass
class ThresholdAnalysisResult:
    threshold: float
    metrics: ClassificationMetrics
    confusion_matrix_result: ConfusionMatrixResult


def predictions_from_threshold(
    prediction_probabilities,
    threshold: float,
):
    return (np.asarray(prediction_probabilities) >= threshold).astype(int)


def analyze_threshold(
    y_true,
    prediction_probabilities,
    threshold: float,
) -> ThresholdAnalysisResult:
    y_pred = predictions_from_threshold(
        prediction_probabilities=prediction_probabilities,
        threshold=threshold,
    )

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=prediction_probabilities,
    )

    confusion_matrix_result = calculate_confusion_matrix_result(
        y_true=y_true,
        y_pred=y_pred,
    )

    return ThresholdAnalysisResult(
        threshold=threshold,
        metrics=metrics,
        confusion_matrix_result=confusion_matrix_result,
    )


def analyze_thresholds(
    y_true,
    prediction_probabilities,
    thresholds: list[float],
) -> list[ThresholdAnalysisResult]:
    return [
        analyze_threshold(
            y_true=y_true,
            prediction_probabilities=prediction_probabilities,
            threshold=threshold,
        )
        for threshold in thresholds
    ]