from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str


class DiabetesPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 85,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 33,
            }
        },
    )

    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

    def to_feature_dict(self) -> dict[str, int | float]:
        return self.model_dump()


class DiabetesPredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
    threshold: float
    model_name: str