from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.config.path import PRODUCTION_MODEL_DIR, ROOT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPromoter:
    def promote(
        self,
        source_model_dir: Path,
        production_model_dir: Path | None = None,
    ) -> Path:
        production_model_dir = production_model_dir or PRODUCTION_MODEL_DIR

        required_files = [
            "model.joblib",
            "preprocessing_pipeline.joblib",
            "model_metadata.json",
        ]

        missing_files = [
            file_name
            for file_name in required_files
            if not (source_model_dir / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Cannot promote model. Missing files: {missing_files}"
            )

        production_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Promoting model artifacts from %s to %s",
            source_model_dir,
            production_model_dir,
        )

        shutil.copy2(
            source_model_dir / "model.joblib",
            production_model_dir / "model.joblib",
        )
        shutil.copy2(
            source_model_dir / "preprocessing_pipeline.joblib",
            production_model_dir / "preprocessing_pipeline.joblib",
        )

        self._write_production_metadata(
            source_metadata_path=source_model_dir / "model_metadata.json",
            production_metadata_path=(
                production_model_dir / "model_metadata.json"
            ),
            production_model_dir=production_model_dir,
        )

        logger.info("Model promoted successfully")

        return production_model_dir

    def _write_production_metadata(
        self,
        source_metadata_path: Path,
        production_metadata_path: Path,
        production_model_dir: Path,
    ) -> None:
        with source_metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        metadata["model_path"] = self._to_relative_path(
            production_model_dir / "model.joblib"
        )
        metadata["preprocessing_pipeline_path"] = self._to_relative_path(
            production_model_dir / "preprocessing_pipeline.joblib"
        )

        with production_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

    def _to_relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(ROOT_DIR))
        except ValueError:
            return str(path)
