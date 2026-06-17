from typing import Any

import pytest

from src.monitoring.monitoring_summary import build_monitoring_summary


def test_build_monitoring_summary_calculates_prediction_distribution() -> None:
    summary = build_monitoring_summary(prediction_logs=_sample_prediction_logs())

    assert summary.total_predictions == 3
    assert summary.prediction_distribution == {
        "0": 2,
        "1": 1,
    }


def test_build_monitoring_summary_calculates_risk_probability_stats() -> None:
    summary = build_monitoring_summary(prediction_logs=_sample_prediction_logs())

    assert summary.risk_probability.mean == pytest.approx(0.5)
    assert summary.risk_probability.min == 0.2
    assert summary.risk_probability.max == 0.9


def test_build_monitoring_summary_calculates_model_distribution() -> None:
    summary = build_monitoring_summary(prediction_logs=_sample_prediction_logs())

    assert summary.models == {
        "xgboost": 2,
        "random_forest": 1,
    }


def test_build_monitoring_summary_calculates_feature_stats() -> None:
    summary = build_monitoring_summary(prediction_logs=_sample_prediction_logs())

    assert summary.features["Glucose"].mean == pytest.approx(120.0)
    assert summary.features["Glucose"].min == 90.0
    assert summary.features["Glucose"].max == 150.0
    assert summary.features["BMI"].mean == pytest.approx(30.0)
    assert summary.features["BMI"].min == 25.0
    assert summary.features["BMI"].max == 35.0


def test_build_monitoring_summary_rejects_empty_logs() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_monitoring_summary(prediction_logs=[])


def _sample_prediction_logs() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2026-06-17T00:00:00+00:00",
            "model_name": "xgboost",
            "risk_probability": 0.2,
            "prediction": 0,
            "threshold": 0.3,
            "input_features": {
                "Glucose": 90.0,
                "BMI": 25.0,
            },
        },
        {
            "timestamp": "2026-06-17T00:01:00+00:00",
            "model_name": "xgboost",
            "risk_probability": 0.9,
            "prediction": 1,
            "threshold": 0.3,
            "input_features": {
                "Glucose": 150.0,
                "BMI": 35.0,
            },
        },
        {
            "timestamp": "2026-06-17T00:02:00+00:00",
            "model_name": "random_forest",
            "risk_probability": 0.4,
            "prediction": 0,
            "threshold": 0.3,
            "input_features": {
                "Glucose": 120.0,
                "BMI": 30.0,
            },
        },
    ]
