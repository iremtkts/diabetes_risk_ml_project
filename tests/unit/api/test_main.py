from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_prediction_logger,
    get_prediction_pipeline,
)
from src.api.main import app

VALID_PREDICTION_PAYLOAD = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 33,
}


class FakePredictionPipeline:
    model_name = "test_model"

    def __init__(self) -> None:
        self.received_input: dict[str, Any] | None = None

    def predict_one(self, input_data: dict[str, Any]) -> SimpleNamespace:
        self.received_input = input_data

        return SimpleNamespace(
            risk_probability=0.42,
            prediction=1,
            threshold=0.3,
        )


class FakePredictionLogger:
    def __init__(self) -> None:
        self.received_logs: list[dict[str, Any]] = []

    def log_prediction(
        self,
        model_name: str,
        risk_probability: float,
        prediction: int,
        threshold: float,
        input_features: dict[str, Any],
    ) -> None:
        self.received_logs.append(
            {
                "model_name": model_name,
                "risk_probability": risk_probability,
                "prediction": prediction,
                "threshold": threshold,
                "input_features": input_features,
            }
        )


@pytest.fixture()
def fake_prediction_pipeline() -> FakePredictionPipeline:
    return FakePredictionPipeline()


@pytest.fixture()
def fake_prediction_logger() -> FakePredictionLogger:
    return FakePredictionLogger()


@pytest.fixture()
def client(
    fake_prediction_pipeline: FakePredictionPipeline,
    fake_prediction_logger: FakePredictionLogger,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_prediction_pipeline] = (
        lambda: fake_prediction_pipeline
    )
    app.dependency_overrides[get_prediction_logger] = (
        lambda: fake_prediction_logger
    )

    test_client = TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_response(
    client: TestClient,
) -> None:
    response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {
        "risk_probability": 0.42,
        "prediction": 1,
        "threshold": 0.3,
        "model_name": "test_model",
    }


def test_predict_passes_valid_payload_to_prediction_pipeline(
    client: TestClient,
    fake_prediction_pipeline: FakePredictionPipeline,
) -> None:
    response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)

    assert response.status_code == 200
    assert fake_prediction_pipeline.received_input == VALID_PREDICTION_PAYLOAD


def test_predict_logs_successful_prediction(
    client: TestClient,
    fake_prediction_logger: FakePredictionLogger,
) -> None:
    response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)

    assert response.status_code == 200
    assert fake_prediction_logger.received_logs == [
        {
            "model_name": "test_model",
            "risk_probability": 0.42,
            "prediction": 1,
            "threshold": 0.3,
            "input_features": VALID_PREDICTION_PAYLOAD,
        }
    ]


def test_predict_returns_422_when_required_feature_is_missing(
    client: TestClient,
) -> None:
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_422_when_unexpected_feature_is_provided(
    client: TestClient,
) -> None:
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33,
        "UnexpectedFeature": 999,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
