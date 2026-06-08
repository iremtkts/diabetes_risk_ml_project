from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.artifact_loader import InferenceArtifactLoader
from src.inference.predictor import DiabetesRiskPredictor, PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(
        self,
        metadata_path: Path | None = None,
    ) -> None:
        self.artifact_loader = InferenceArtifactLoader(
            metadata_path=metadata_path
        )

        metadata = self.artifact_loader.load_metadata()
        model = self.artifact_loader.load_model(metadata)
        preprocessing_pipeline = (
            self.artifact_loader.load_preprocessing_pipeline(metadata)
        )

        self.predictor = DiabetesRiskPredictor(
            model=model,
            preprocessing_pipeline=preprocessing_pipeline,
            threshold=metadata["selected_threshold"],
            input_features=metadata["input_features"],
        )

    def predict_one(
        self,
        input_data: dict[str, Any],
    ) -> PredictionResult:
        logger.info("Running prediction pipeline")

        prediction_result = self.predictor.predict_one(input_data=input_data)

        logger.info(
            "Prediction completed | probability=%.4f | prediction=%s | threshold=%.2f",
            prediction_result.risk_probability,
            prediction_result.prediction,
            prediction_result.threshold,
        )

        return prediction_result