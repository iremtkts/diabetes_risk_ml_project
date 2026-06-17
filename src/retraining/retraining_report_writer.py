from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config.path import REPORTS_DIR
from src.retraining.promotion_gate import PromotionGateResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSnapshot:
    model_name: str
    metrics: dict[str, float]
    confusion_matrix: dict[str, int]


@dataclass(frozen=True)
class RetrainingReport:
    retraining_status: str
    candidate_model: ModelSnapshot
    production_model: ModelSnapshot
    promotion_gate: PromotionGateResult


class RetrainingReportWriter:
    def write(
        self,
        retraining_report: RetrainingReport,
        output_path: Path | None = None,
    ) -> Path:
        output_path = (
            output_path
            or REPORTS_DIR
            / "retraining"
            / "retraining_report.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing retraining report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(retraining_report), file, indent=4)

        logger.info("Retraining report written successfully")

        return output_path
