from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.config.path import REPORTS_DIR
from src.monitoring.drift_detector import DriftReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftReportWriter:
    def write(
        self,
        drift_report: DriftReport,
        output_path: Path | None = None,
    ) -> Path:
        output_path = (
            output_path
            or REPORTS_DIR
            / "drift"
            / "drift_report.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing drift report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(drift_report), file, indent=4)

        logger.info("Drift report written successfully")

        return output_path
