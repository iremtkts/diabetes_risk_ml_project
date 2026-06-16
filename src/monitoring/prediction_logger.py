from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.path import ROOT_DIR

DEFAULT_PREDICTION_LOG_PATH = (
    ROOT_DIR / "logs" / "predictions" / "prediction_logs.jsonl"
)


class PredictionLogger:
    def __init__(
        self,
        log_path: Path | None = None,
    ) -> None:
        self.log_path = log_path or DEFAULT_PREDICTION_LOG_PATH

    def log_prediction(
        self,
        model_name: str,
        risk_probability: float,
        prediction: int,
        threshold: float,
        input_features: dict[str, Any],
    ) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "risk_probability": risk_probability,
            "prediction": prediction,
            "threshold": threshold,
            "input_features": input_features,
        }

        with self.log_path.open("a", encoding="utf-8") as file:
            json.dump(payload, file)
            file.write("\n")

        return self.log_path
