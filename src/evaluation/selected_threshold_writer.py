from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.config.path import REPORTS_DIR
from src.evaluation.threshold_selector import SelectedThresholdResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SelectedThresholdWriter:
    def write(
        self,
        selected_threshold_result: SelectedThresholdResult,
        output_path: Path | None = None,
    ) -> None:
        output_path = output_path or REPORTS_DIR / "selected_threshold.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "selected_threshold": selected_threshold_result.selected_threshold,
            "selection_strategy": selected_threshold_result.selection_strategy,
            "metrics": asdict(selected_threshold_result.metrics),
            "confusion_matrix": asdict(
                selected_threshold_result.confusion_matrix_result
            ),
        }

        logger.info("Writing selected threshold artifact to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

        logger.info("Selected threshold artifact written successfully")