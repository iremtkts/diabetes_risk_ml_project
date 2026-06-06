from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.evaluator import EvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReportWriter:
    def write(
        self,
        evaluation_result: EvaluationResult,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metrics": asdict(evaluation_result.metrics),
            "confusion_matrix": asdict(
                evaluation_result.confusion_matrix_result
            ),
        }

        logger.info("Writing evaluation report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        logger.info("Evaluation report written successfully")