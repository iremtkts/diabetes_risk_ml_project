from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.threshold_analysis import ThresholdAnalysisResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdAnalysisReportWriter:
    def write(
        self,
        threshold_analysis_results: list[ThresholdAnalysisResult],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = [
            {
                "threshold": result.threshold,
                "metrics": asdict(result.metrics),
                "confusion_matrix": asdict(result.confusion_matrix_result),
            }
            for result in threshold_analysis_results
        ]

        logger.info("Writing threshold analysis report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        logger.info("Threshold analysis report written successfully")