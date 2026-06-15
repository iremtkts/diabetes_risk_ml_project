from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray

from src.config.path import REPORTS_DIR
from src.explainability.shap_analyzer import ShapFeatureImportance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapReportWriter:
    def write_feature_importance(
        self,
        model_name: str,
        n_samples: int,
        feature_importance: list[ShapFeatureImportance],
        output_path: Path | None = None,
    ) -> Path:
        output_path = (
            output_path
            or REPORTS_DIR
            / "explainability"
            / "shap_feature_importance.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": model_name,
            "n_samples": n_samples,
            "features": [
                asdict(feature_result)
                for feature_result in feature_importance
            ],
        }

        logger.info("Writing SHAP feature importance report to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

        logger.info("SHAP feature importance report written successfully")

        return output_path

    def write_summary_plot(
        self,
        shap_values: NDArray[np.float64],
        features: pd.DataFrame,
        output_path: Path | None = None,
    ) -> Path:
        output_path = (
            output_path
            or REPORTS_DIR / "explainability" / "shap_summary.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            shap_values,
            features,
            show=False,
        )

        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        logger.info("SHAP summary plot written to: %s", output_path)

        return output_path
