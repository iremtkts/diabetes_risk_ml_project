from __future__ import annotations

from dataclasses import asdict

from src.pipelines.prediction_pipeline import PredictionPipeline


def main() -> None:
    pipeline = PredictionPipeline()

    sample_input = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }

    prediction_result = pipeline.predict_one(input_data=sample_input)

    print(asdict(prediction_result))


if __name__ == "__main__":
    main()