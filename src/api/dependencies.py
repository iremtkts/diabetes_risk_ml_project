from __future__ import annotations

from functools import lru_cache

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> PredictionPipeline:
    logger.info("Initializing prediction pipeline")
    return PredictionPipeline()