from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    risk_probability: float
    prediction: int
    threshold: float


class DiabetesRiskPredictor:
    def __init__(
        self,
        model: Any,
        preprocessing_pipeline: Any,
        threshold: float,
        input_features: list[str],
    ) -> None:
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.threshold = threshold
        self.input_features = input_features

    def predict_one(
        self,
        input_data: dict[str, Any],
    ) -> PredictionResult:
        input_dataframe = pd.DataFrame([input_data])

        self._validate_input_features(input_dataframe)

        input_dataframe = input_dataframe[self.input_features]

        processed_input = self.preprocessing_pipeline.transform(input_dataframe)

        risk_probability = float(
            self.model.predict_proba(processed_input)[:, 1][0]
        )

        prediction = int(risk_probability >= self.threshold)

        return PredictionResult(
            risk_probability=risk_probability,
            prediction=prediction,
            threshold=self.threshold,
        )

    def _validate_input_features(
        self,
        input_dataframe: pd.DataFrame,
    ) -> None:
        missing_features = [
            feature
            for feature in self.input_features
            if feature not in input_dataframe.columns
        ]

        if missing_features:
            raise ValueError(f"Missing input features: {missing_features}")