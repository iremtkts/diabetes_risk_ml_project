import json

from src.monitoring.monitoring_report_writer import MonitoringReportWriter
from src.monitoring.monitoring_summary import (
    NumericSummary,
    PredictionMonitoringSummary,
)


def test_monitoring_report_writer_creates_output_directory(tmp_path) -> None:
    output_path = tmp_path / "reports" / "monitoring" / "summary.json"
    writer = MonitoringReportWriter()

    writer.write(
        monitoring_summary=_sample_monitoring_summary(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_monitoring_report_writer_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "prediction_monitoring_summary.json"
    writer = MonitoringReportWriter()

    writer.write(
        monitoring_summary=_sample_monitoring_summary(),
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload == {
        "total_predictions": 2,
        "prediction_distribution": {
            "0": 1,
            "1": 1,
        },
        "risk_probability": {
            "mean": 0.5,
            "min": 0.2,
            "max": 0.8,
        },
        "models": {
            "xgboost": 2,
        },
        "features": {
            "Glucose": {
                "mean": 120.0,
                "min": 90.0,
                "max": 150.0,
            }
        },
    }


def _sample_monitoring_summary() -> PredictionMonitoringSummary:
    return PredictionMonitoringSummary(
        total_predictions=2,
        prediction_distribution={
            "0": 1,
            "1": 1,
        },
        risk_probability=NumericSummary(
            mean=0.5,
            min=0.2,
            max=0.8,
        ),
        models={
            "xgboost": 2,
        },
        features={
            "Glucose": NumericSummary(
                mean=120.0,
                min=90.0,
                max=150.0,
            )
        },
    )
