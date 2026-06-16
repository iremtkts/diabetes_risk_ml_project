import json

from src.monitoring.prediction_logger import PredictionLogger


def test_prediction_logger_creates_output_directory(tmp_path) -> None:
    log_path = tmp_path / "logs" / "predictions" / "prediction_logs.jsonl"
    prediction_logger = PredictionLogger(log_path=log_path)

    prediction_logger.log_prediction(
        model_name="xgboost",
        risk_probability=0.42,
        prediction=1,
        threshold=0.3,
        input_features={"Glucose": 120},
    )

    assert log_path.exists()


def test_prediction_logger_writes_single_valid_json_line(tmp_path) -> None:
    log_path = tmp_path / "prediction_logs.jsonl"
    prediction_logger = PredictionLogger(log_path=log_path)

    prediction_logger.log_prediction(
        model_name="xgboost",
        risk_probability=0.42,
        prediction=1,
        threshold=0.3,
        input_features={"Glucose": 120},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert len(lines) == 1
    assert payload["model_name"] == "xgboost"
    assert payload["risk_probability"] == 0.42
    assert payload["prediction"] == 1
    assert payload["threshold"] == 0.3
    assert payload["input_features"] == {"Glucose": 120}
    assert "timestamp" in payload


def test_prediction_logger_appends_multiple_logs(tmp_path) -> None:
    log_path = tmp_path / "prediction_logs.jsonl"
    prediction_logger = PredictionLogger(log_path=log_path)

    prediction_logger.log_prediction(
        model_name="xgboost",
        risk_probability=0.42,
        prediction=1,
        threshold=0.3,
        input_features={"Glucose": 120},
    )
    prediction_logger.log_prediction(
        model_name="xgboost",
        risk_probability=0.12,
        prediction=0,
        threshold=0.3,
        input_features={"Glucose": 90},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["risk_probability"] == 0.42
    assert json.loads(lines[1])["risk_probability"] == 0.12
