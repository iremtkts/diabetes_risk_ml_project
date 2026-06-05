from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    metrics: ClassificationMetrics
    predictions: Any
    prediction_probabilities: Any


class ModelEvaluator:
    def evaluate(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> EvaluationResult:
        logger.info("Starting model evaluation")

        predictions = model.predict(X_test)

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Model must implement predict_proba to calculate roc_auc"
            )

        prediction_probabilities = model.predict_proba(X_test)[:, 1]

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=predictions,
            y_pred_proba=prediction_probabilities,
        )

        logger.info(
            "Model evaluation completed | accuracy=%.4f | f1=%.4f | roc_auc=%.4f",
            metrics.accuracy,
            metrics.f1,
            metrics.roc_auc,
        )

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )