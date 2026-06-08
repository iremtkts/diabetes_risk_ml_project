from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.config.path import MODELS_DIR, ROOT_DIR
from src.preprocessing.column_config import MODEL_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelArtifactWriter:
    
    @staticmethod
    def _to_relative_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT_DIR))
        except ValueError:
            return str(path)
    
    def save_model(
        self,
        model: Any,
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or MODELS_DIR / "model.joblib"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model artifact to: %s", output_path)
        joblib.dump(model, output_path)
        logger.info("Model artifact saved successfully")

        return output_path

    def save_preprocessing_pipeline(
        self,
        preprocessing_pipeline: Any,
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or MODELS_DIR / "preprocessing_pipeline.joblib"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving preprocessing pipeline artifact to: %s", output_path)
        joblib.dump(preprocessing_pipeline, output_path)
        logger.info("Preprocessing pipeline artifact saved successfully")

        return output_path

    def save_metadata(
        self,
        model_name: str,
        model_path: Path,
        preprocessing_pipeline_path: Path,
        selected_threshold: float,
        processed_features: list[str],
        output_path: Path | None = None,
    ) -> Path:
        output_path = output_path or MODELS_DIR / "model_metadata.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": model_name,
            "model_path": self._to_relative_path(model_path),
            "preprocessing_pipeline_path": self._to_relative_path(preprocessing_pipeline_path),
            "selected_threshold": selected_threshold,
            "input_features": MODEL_FEATURES,
            "processed_features": processed_features,
        }

        logger.info("Saving model metadata artifact to: %s", output_path)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

        logger.info("Model metadata artifact saved successfully")

        return output_path