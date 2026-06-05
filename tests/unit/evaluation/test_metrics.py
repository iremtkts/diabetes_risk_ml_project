import pytest

from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)


def test_calculate_classification_metrics_returns_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_pred_proba = [0.1, 0.8, 0.4, 0.2]

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
    )

    assert isinstance(metrics, ClassificationMetrics)


def test_calculate_classification_metrics_values():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_pred_proba = [0.1, 0.8, 0.4, 0.2]

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.roc_auc == pytest.approx(1.0)