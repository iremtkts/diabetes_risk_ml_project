import json
from typing import Any

import pytest

from src.monitoring.prediction_log_reader import PredictionLogReader


def _write_jsonl(
    log_path,
    records: list[dict[str, Any]],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")


def test_prediction_log_reader_reads_valid_jsonl(tmp_path) -> None:
    log_path = tmp_path / "prediction_logs.jsonl"
    records = [
        {
            "timestamp": "2026-06-17T00:00:00+00:00",
            "model_name": "xgboost",
            "risk_probability": 0.42,
            "prediction": 1,
            "threshold": 0.3,
            "input_features": {"Glucose": 120},
        }
    ]
    _write_jsonl(log_path=log_path, records=records)

    reader = PredictionLogReader(log_path=log_path)

    assert reader.read() == records


def test_prediction_log_reader_raises_error_when_file_is_missing(tmp_path) -> None:
    log_path = tmp_path / "missing.jsonl"
    reader = PredictionLogReader(log_path=log_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        reader.read()


def test_prediction_log_reader_raises_error_when_file_is_empty(tmp_path) -> None:
    log_path = tmp_path / "prediction_logs.jsonl"
    log_path.write_text("", encoding="utf-8")
    reader = PredictionLogReader(log_path=log_path)

    with pytest.raises(ValueError, match="empty"):
        reader.read()


def test_prediction_log_reader_raises_error_for_malformed_json(tmp_path) -> None:
    log_path = tmp_path / "prediction_logs.jsonl"
    log_path.write_text("{not valid json}\n", encoding="utf-8")
    reader = PredictionLogReader(log_path=log_path)

    with pytest.raises(ValueError, match="Malformed JSON"):
        reader.read()
