from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.metrics import ClassificationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsWriter:
    def write(
        self,
        metrics: ClassificationMetrics,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_dict = asdict(metrics)

        logger.info("Writing metrics artifact to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_dict, file, indent=4)

        logger.info("Metrics artifact written successfully")