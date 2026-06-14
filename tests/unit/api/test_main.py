from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_prediction_pipeline
from src.api.main import app


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


@pytest.fixture()
def fake_prediction_pipeline() -> FakePredictionPipeline:
    return FakePredictionPipeline()




@pytest.fixture()
def client(
    fake_prediction_pipeline: FakePredictionPipeline,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_prediction_pipeline] = (
        lambda: fake_prediction_pipeline
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
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33,
    }

    response = client.post("/predict", json=payload)

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
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert fake_prediction_pipeline.received_input == payload


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