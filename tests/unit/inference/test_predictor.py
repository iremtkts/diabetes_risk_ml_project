from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.inference.predictor import DiabetesRiskPredictor


class DummyModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.array(
            [
                [
                    1 - self.positive_probability,
                    self.positive_probability,
                ]
            ]
        )


class RecordingPreprocessingPipeline:
    def __init__(self) -> None:
        self.received_columns: list[str] | None = None

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        self.received_columns = list(dataframe.columns)

        return dataframe.to_numpy()


def test_predictor_returns_prediction_using_selected_threshold() -> None:
    model = DummyModel(positive_probability=0.75)
    preprocessing_pipeline = RecordingPreprocessingPipeline()

    input_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    predictor = DiabetesRiskPredictor(
        model=model,
        preprocessing_pipeline=preprocessing_pipeline,
        threshold=0.3,
        input_features=input_features,
    )

    input_data = {
        "Age": 35,
        "BMI": 28.5,
        "Glucose": 120,
        "Pregnancies": 2,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "DiabetesPedigreeFunction": 0.5,
    }

    prediction_result = predictor.predict_one(input_data=input_data)

    assert prediction_result.risk_probability == 0.75
    assert prediction_result.prediction == 1
    assert prediction_result.threshold == 0.3
    assert preprocessing_pipeline.received_columns == input_features


def test_predictor_raises_error_when_required_feature_is_missing() -> None:
    model = DummyModel(positive_probability=0.75)
    preprocessing_pipeline = RecordingPreprocessingPipeline()

    predictor = DiabetesRiskPredictor(
        model=model,
        preprocessing_pipeline=preprocessing_pipeline,
        threshold=0.3,
        input_features=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
        ],
    )

    input_data = {
        "Pregnancies": 2,
        "Glucose": 120,
    }

    with pytest.raises(ValueError, match="Missing input features"):
        predictor.predict_one(input_data=input_data)