from typing import Any

import pandas as pd
import pytest

from src.config.path import ROOT_DIR
from src.monitoring.drift_detector import DriftDetector


def test_drift_detector_detects_no_schema_drift(tmp_path) -> None:
    report = DriftDetector(expected_features=["Glucose", "BMI"]).detect(
        reference_data=_reference_data(),
        current_logs=_current_logs(),
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    assert report.schema_drift.missing_features == []
    assert report.schema_drift.unexpected_features == []
    assert report.schema_drift.schema_drift_detected is False


def test_drift_detector_reports_paths_relative_to_root_dir() -> None:
    report = DriftDetector(expected_features=["Glucose"]).detect(
        reference_data=_reference_data(),
        current_logs=[
            {
                "input_features": {
                    "Glucose": 100.0,
                }
            }
        ],
        reference_data_path=ROOT_DIR / "data" / "raw" / "diabetes.csv",
        current_data_path=(
            ROOT_DIR / "logs" / "predictions" / "prediction_logs.jsonl"
        ),
    )

    assert report.reference_data_path == "data/raw/diabetes.csv"
    assert report.current_data_path == "logs/predictions/prediction_logs.jsonl"


def test_drift_detector_detects_missing_feature(tmp_path) -> None:
    current_logs = [
        {
            "input_features": {
                "Glucose": 100.0,
            }
        }
    ]

    report = DriftDetector(expected_features=["Glucose", "BMI"]).detect(
        reference_data=_reference_data(),
        current_logs=current_logs,
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    assert report.schema_drift.missing_features == ["BMI"]
    assert report.schema_drift.schema_drift_detected is True
    assert report.data_drift.features[1].calculation_available is False
    assert report.data_drift.features[1].reason == "missing_current_feature"


def test_drift_detector_detects_unexpected_feature(tmp_path) -> None:
    current_logs = [
        {
            "input_features": {
                "Glucose": 100.0,
                "BMI": 25.0,
                "UnexpectedFeature": 1.0,
            }
        }
    ]

    report = DriftDetector(expected_features=["Glucose", "BMI"]).detect(
        reference_data=_reference_data(),
        current_logs=current_logs,
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    assert report.schema_drift.unexpected_features == ["UnexpectedFeature"]
    assert report.schema_drift.schema_drift_detected is True


def test_drift_detector_does_not_flag_data_drift_below_threshold(
    tmp_path,
) -> None:
    report = DriftDetector(
        expected_features=["Glucose"],
        threshold=0.2,
    ).detect(
        reference_data=_reference_data(),
        current_logs=[
            {
                "input_features": {
                    "Glucose": 110.0,
                }
            }
        ],
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    feature_result = report.data_drift.features[0]

    assert report.data_drift.data_drift_detected is False
    assert feature_result.reference_mean == pytest.approx(100.0)
    assert feature_result.current_mean == pytest.approx(110.0)
    assert feature_result.relative_difference == pytest.approx(0.1)
    assert feature_result.drift_detected is False


def test_drift_detector_flags_data_drift_above_threshold(tmp_path) -> None:
    report = DriftDetector(
        expected_features=["Glucose"],
        threshold=0.2,
    ).detect(
        reference_data=_reference_data(),
        current_logs=[
            {
                "input_features": {
                    "Glucose": 150.0,
                }
            }
        ],
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    feature_result = report.data_drift.features[0]

    assert report.data_drift.data_drift_detected is True
    assert feature_result.absolute_difference == pytest.approx(50.0)
    assert feature_result.relative_difference == pytest.approx(0.5)
    assert feature_result.drift_detected is True


def test_drift_detector_handles_reference_mean_zero(tmp_path) -> None:
    report = DriftDetector(
        expected_features=["Glucose"],
        threshold=0.2,
    ).detect(
        reference_data=pd.DataFrame({"Glucose": [0.0, 0.0]}),
        current_logs=[
            {
                "input_features": {
                    "Glucose": 1.0,
                }
            }
        ],
        reference_data_path=tmp_path / "reference.csv",
        current_data_path=tmp_path / "current.jsonl",
    )

    feature_result = report.data_drift.features[0]

    assert feature_result.reference_mean == 0.0
    assert feature_result.relative_difference == 1.0
    assert feature_result.drift_detected is True


def test_drift_detector_rejects_missing_input_features(tmp_path) -> None:
    with pytest.raises(ValueError, match="input_features"):
        DriftDetector(expected_features=["Glucose"]).detect(
            reference_data=_reference_data(),
            current_logs=[{"model_name": "xgboost"}],
            reference_data_path=tmp_path / "reference.csv",
            current_data_path=tmp_path / "current.jsonl",
        )


def _reference_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Glucose": [90.0, 110.0],
            "BMI": [20.0, 30.0],
        }
    )


def _current_logs() -> list[dict[str, Any]]:
    return [
        {
            "input_features": {
                "Glucose": 95.0,
                "BMI": 22.0,
            }
        },
        {
            "input_features": {
                "Glucose": 105.0,
                "BMI": 28.0,
            }
        },
    ]
