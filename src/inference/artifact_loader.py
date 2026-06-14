from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib

from src.config.path import PRODUCTION_MODEL_DIR, ROOT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceArtifactLoader:
    def __init__(
        self,
        metadata_path: Path | None = None,
    ) -> None:
        self.metadata_path = (
          metadata_path or PRODUCTION_MODEL_DIR / "model_metadata.json"
        )

    def load_metadata(self) -> dict[str, Any]:
        logger.info("Loading model metadata from: %s", self.metadata_path)

        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = cast(dict[str, Any], json.load(file))

        logger.info("Model metadata loaded successfully")

        return metadata

    @staticmethod
    def _resolve_path(path_value: str) -> Path:
        path = Path(path_value)

        if path.is_absolute():
            return path

        return ROOT_DIR / path

    def load_model(self, metadata: dict[str, Any]) -> Any:
        model_path = self._resolve_path(metadata["model_path"])

        logger.info("Loading model artifact from: %s", model_path)

        model = joblib.load(model_path)

        logger.info("Model artifact loaded successfully")

        return model

    def load_preprocessing_pipeline(self, metadata: dict[str, Any]) -> Any:
        preprocessing_pipeline_path = self._resolve_path(
            metadata["preprocessing_pipeline_path"]
        )

        logger.info(
            "Loading preprocessing pipeline artifact from: %s",
            preprocessing_pipeline_path,
        )

        preprocessing_pipeline = joblib.load(preprocessing_pipeline_path)

        logger.info("Preprocessing pipeline artifact loaded successfully")

        return preprocessing_pipeline
