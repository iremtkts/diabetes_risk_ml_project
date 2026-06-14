from __future__ import annotations

from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="Diabetes Risk Prediction API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(prediction_router)