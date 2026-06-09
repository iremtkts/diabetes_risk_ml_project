from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.config.path import REPORTS_DIR
from src.evaluation.best_model_selector import BestModelSelectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BestModelReportWriter:
    def write(
        self,
        best_model_selection_result: BestModelSelectionResult,
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or REPORTS_DIR / "best_model_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = asdict(best_model_selection_result)

        logger.info("Writing best model report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

        logger.info("Best model report written successfully")

        return output_path