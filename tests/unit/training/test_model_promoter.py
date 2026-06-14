import json
import textwrap
from pathlib import Path

import pytest

from src.training.model_promoter import ModelPromoter


def test_model_promoter_copies_required_artifacts_and_updates_metadata(
    tmp_path: Path,
) -> None:
    source_model_dir = tmp_path / "xgboost"
    production_model_dir = tmp_path / "production"
    source_model_dir.mkdir(parents=True)

    (source_model_dir / "model.joblib").write_text(
        "fake model",
        encoding="utf-8",
    )
    (source_model_dir / "preprocessing_pipeline.joblib").write_text(
        "fake preprocessing pipeline",
        encoding="utf-8",
    )

    metadata_content = textwrap.dedent(
        """\
        {
            "model_name": "xgboost",
            "model_path": "artifacts/models/xgboost/model.joblib",
            "preprocessing_pipeline_path":
                "artifacts/models/xgboost/preprocessing_pipeline.joblib",
            "selected_threshold": 0.3,
            "input_features": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ],
            "processed_features": [
                "numerical__Pregnancies",
                "numerical__Glucose",
                "numerical__BloodPressure",
                "numerical__SkinThickness",
                "numerical__Insulin",
                "numerical__BMI",
                "numerical__DiabetesPedigreeFunction",
                "numerical__Age"
            ]
        }
        """
    )

    (source_model_dir / "model_metadata.json").write_text(
        metadata_content,
        encoding="utf-8",
    )

    promoter = ModelPromoter()

    result_path = promoter.promote(
        source_model_dir=source_model_dir,
        production_model_dir=production_model_dir,
    )

    assert result_path == production_model_dir

    assert (production_model_dir / "model.joblib").exists()
    assert (production_model_dir / "preprocessing_pipeline.joblib").exists()
    assert (production_model_dir / "model_metadata.json").exists()

    assert (production_model_dir / "model.joblib").read_text(
        encoding="utf-8",
    ) == "fake model"

    assert (production_model_dir / "preprocessing_pipeline.joblib").read_text(
        encoding="utf-8",
    ) == "fake preprocessing pipeline"

    with (production_model_dir / "model_metadata.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["model_name"] == "xgboost"
    assert metadata["selected_threshold"] == 0.3
    assert metadata["model_path"].endswith("production/model.joblib")
    assert metadata["preprocessing_pipeline_path"].endswith(
        "production/preprocessing_pipeline.joblib"
    )


def test_model_promoter_raises_error_when_required_file_is_missing(
    tmp_path: Path,
) -> None:
    source_model_dir = tmp_path / "xgboost"
    production_model_dir = tmp_path / "production"
    source_model_dir.mkdir(parents=True)

    (source_model_dir / "model.joblib").write_text(
        "fake model",
        encoding="utf-8",
    )

    promoter = ModelPromoter()

    with pytest.raises(FileNotFoundError, match="Cannot promote model"):
        promoter.promote(
            source_model_dir=source_model_dir,
            production_model_dir=production_model_dir,
        )
