from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config.path import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelComparisonReportWriter:
    def write_json(
        self,
        comparison_results: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or REPORTS_DIR / "model_comparison_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing model comparison JSON report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(comparison_results, file, indent=4)

        logger.info("Model comparison JSON report written successfully")

        return output_path

    def write_csv(
        self,
        comparison_results: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or REPORTS_DIR / "model_comparison_report.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing model comparison CSV report to: %s", output_path)

        if not comparison_results:
            raise ValueError("comparison_results cannot be empty")

        fieldnames = list(comparison_results[0].keys())

        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_results)

        logger.info("Model comparison CSV report written successfully")

        return output_path