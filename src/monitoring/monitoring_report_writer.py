from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.config.path import REPORTS_DIR
from src.monitoring.monitoring_summary import PredictionMonitoringSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonitoringReportWriter:
    def write(
        self,
        monitoring_summary: PredictionMonitoringSummary,
        output_path: Path | None = None,
    ) -> Path:
        output_path = (
            output_path
            or REPORTS_DIR
            / "monitoring"
            / "prediction_monitoring_summary.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing prediction monitoring summary to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(monitoring_summary), file, indent=4)

        logger.info("Prediction monitoring summary written successfully")

        return output_path
