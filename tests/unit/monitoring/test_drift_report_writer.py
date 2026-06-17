import json

from src.monitoring.drift_detector import (
    DataDriftResult,
    DriftReport,
    FeatureDriftResult,
    SchemaDriftResult,
)
from src.monitoring.drift_report_writer import DriftReportWriter


def test_drift_report_writer_creates_output_directory(tmp_path) -> None:
    output_path = tmp_path / "reports" / "drift" / "drift_report.json"
    writer = DriftReportWriter()

    writer.write(
        drift_report=_sample_drift_report(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_drift_report_writer_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "drift_report.json"
    writer = DriftReportWriter()

    writer.write(
        drift_report=_sample_drift_report(),
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload == {
        "reference_data_path": "data/raw/diabetes.csv",
        "current_data_path": "logs/predictions/prediction_logs.jsonl",
        "total_current_records": 2,
        "schema_drift": {
            "missing_features": [],
            "unexpected_features": [],
            "schema_drift_detected": False,
        },
        "data_drift": {
            "threshold": 0.2,
            "data_drift_detected": True,
            "features": [
                {
                    "feature_name": "Glucose",
                    "reference_mean": 100.0,
                    "current_mean": 150.0,
                    "absolute_difference": 50.0,
                    "relative_difference": 0.5,
                    "drift_detected": True,
                    "calculation_available": True,
                    "reason": None,
                }
            ],
        },
    }


def _sample_drift_report() -> DriftReport:
    return DriftReport(
        reference_data_path="data/raw/diabetes.csv",
        current_data_path="logs/predictions/prediction_logs.jsonl",
        total_current_records=2,
        schema_drift=SchemaDriftResult(
            missing_features=[],
            unexpected_features=[],
            schema_drift_detected=False,
        ),
        data_drift=DataDriftResult(
            threshold=0.2,
            data_drift_detected=True,
            features=[
                FeatureDriftResult(
                    feature_name="Glucose",
                    reference_mean=100.0,
                    current_mean=150.0,
                    absolute_difference=50.0,
                    relative_difference=0.5,
                    drift_detected=True,
                    calculation_available=True,
                )
            ],
        ),
    )
