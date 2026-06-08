import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.model_artifact_writer import ModelArtifactWriter


def test_model_artifact_writer_saves_model_and_preprocessing_artifacts(
    tmp_path,
) -> None:
    writer = ModelArtifactWriter()

    model = LogisticRegression()
    preprocessing_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    model_path = tmp_path / "model.joblib"
    preprocessing_pipeline_path = tmp_path / "preprocessing_pipeline.joblib"

    saved_model_path = writer.save_model(
        model=model,
        output_path=model_path,
    )

    saved_preprocessing_pipeline_path = writer.save_preprocessing_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        output_path=preprocessing_pipeline_path,
    )

    assert saved_model_path == model_path
    assert saved_preprocessing_pipeline_path == preprocessing_pipeline_path
    assert model_path.exists()
    assert preprocessing_pipeline_path.exists()

    loaded_model = joblib.load(model_path)
    loaded_preprocessing_pipeline = joblib.load(preprocessing_pipeline_path)

    assert isinstance(loaded_model, LogisticRegression)
    assert isinstance(loaded_preprocessing_pipeline, Pipeline)
    
def test_model_artifact_writer_saves_metadata_json(tmp_path) -> None:
    writer = ModelArtifactWriter()

    model_path = tmp_path / "model.joblib"
    preprocessing_pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
    metadata_path = tmp_path / "model_metadata.json"

    saved_metadata_path = writer.save_metadata(
        model_name="logistic_regression",
        model_path=model_path,
        preprocessing_pipeline_path=preprocessing_pipeline_path,
        selected_threshold=0.3,
        processed_features=[
            "numerical__Pregnancies",
            "numerical__Glucose",
        ],
        output_path=metadata_path,
    )

    assert saved_metadata_path == metadata_path
    assert metadata_path.exists()

    with metadata_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["model_name"] == "logistic_regression"
    assert payload["model_path"] == str(model_path)
    assert payload["preprocessing_pipeline_path"] == str(
        preprocessing_pipeline_path
    )
    assert payload["selected_threshold"] == 0.3
    assert payload["processed_features"] == [
        "numerical__Pregnancies",
        "numerical__Glucose",
    ]
    assert payload["input_features"] == [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]