from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, cast

from src.monitoring.prediction_logger import DEFAULT_PREDICTION_LOG_PATH


class PredictionLogReader:
    def __init__(
        self,
        log_path: Path | None = None,
    ) -> None:
        self.log_path = log_path or DEFAULT_PREDICTION_LOG_PATH

    def read(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            raise FileNotFoundError(
                f"Prediction log file does not exist: {self.log_path}"
            )

        records: list[dict[str, Any]] = []

        with self.log_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except JSONDecodeError as error:
                    raise ValueError(
                        "Malformed JSON in prediction log file "
                        f"at line {line_number}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        "Prediction log records must be JSON objects "
                        f"at line {line_number}"
                    )

                records.append(cast(dict[str, Any], record))

        if not records:
            raise ValueError("Prediction log file is empty")

        return records
