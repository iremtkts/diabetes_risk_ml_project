from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NumericSummary:
    mean: float
    min: float
    max: float


@dataclass(frozen=True)
class PredictionMonitoringSummary:
    total_predictions: int
    prediction_distribution: dict[str, int]
    risk_probability: NumericSummary
    models: dict[str, int]
    features: dict[str, NumericSummary]


def build_monitoring_summary(
    prediction_logs: list[dict[str, Any]],
) -> PredictionMonitoringSummary:
    if not prediction_logs:
        raise ValueError("prediction_logs cannot be empty")

    predictions = [
        int(record["prediction"])
        for record in prediction_logs
    ]
    risk_probabilities = [
        float(record["risk_probability"])
        for record in prediction_logs
    ]
    model_names = [
        str(record["model_name"])
        for record in prediction_logs
    ]

    feature_values: dict[str, list[float]] = defaultdict(list)

    for record in prediction_logs:
        input_features = record["input_features"]

        if not isinstance(input_features, dict):
            raise ValueError("input_features must be a JSON object")

        for feature_name, feature_value in input_features.items():
            if isinstance(feature_value, int | float):
                feature_values[str(feature_name)].append(float(feature_value))

    return PredictionMonitoringSummary(
        total_predictions=len(prediction_logs),
        prediction_distribution=_count_as_string_keys(predictions),
        risk_probability=_summarize_numeric_values(risk_probabilities),
        models=dict(Counter(model_names)),
        features={
            feature_name: _summarize_numeric_values(values)
            for feature_name, values in feature_values.items()
        },
    )


def _count_as_string_keys(values: list[int]) -> dict[str, int]:
    counter = Counter(values)

    return {
        str(key): counter[key]
        for key in sorted(counter)
    }


def _summarize_numeric_values(values: list[float]) -> NumericSummary:
    if not values:
        raise ValueError("Cannot summarize empty numeric values")

    return NumericSummary(
        mean=sum(values) / len(values),
        min=min(values),
        max=max(values),
    )
