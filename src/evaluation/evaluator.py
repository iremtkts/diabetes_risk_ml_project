from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from src.evaluation.report import (
    ConfusionMatrixResult,
    calculate_confusion_matrix_result,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    metrics: ClassificationMetrics
    confusion_matrix_result: ConfusionMatrixResult
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

        confusion_matrix_result = calculate_confusion_matrix_result(
            y_true=y_test,
            y_pred=predictions,
        )

        logger.info(
            (
                "Model evaluation completed | "
                "accuracy=%.4f | f1=%.4f | roc_auc=%.4f | "
                "tn=%s | fp=%s | fn=%s | tp=%s"
            ),
            metrics.accuracy,
            metrics.f1,
            metrics.roc_auc,
            confusion_matrix_result.true_negative,
            confusion_matrix_result.false_positive,
            confusion_matrix_result.false_negative,
            confusion_matrix_result.true_positive,
        )

        return EvaluationResult(
            metrics=metrics,
            confusion_matrix_result=confusion_matrix_result,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )