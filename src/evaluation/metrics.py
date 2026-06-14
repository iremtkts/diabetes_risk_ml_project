from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_pred_proba: Any,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_pred_proba)),
    )
