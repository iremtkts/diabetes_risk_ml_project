from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_prediction_pipeline
from src.api.schemas import (
    DiabetesPredictionRequest,
    DiabetesPredictionResponse,
)
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    tags=["prediction"],
)


@router.post("/predict", response_model=DiabetesPredictionResponse)
def predict(
    request: DiabetesPredictionRequest,
    prediction_pipeline: Annotated[
        PredictionPipeline,
        Depends(get_prediction_pipeline),
    ],
) -> DiabetesPredictionResponse:
    try:
        result = prediction_pipeline.predict_one(
            input_data=request.to_feature_dict()
        )

        return DiabetesPredictionResponse(
            risk_probability=result.risk_probability,
            prediction=result.prediction,
            threshold=result.threshold,
            model_name=prediction_pipeline.model_name,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc