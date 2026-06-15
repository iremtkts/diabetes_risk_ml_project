import json

from src.explainability.shap_analyzer import ShapFeatureImportance
from src.explainability.shap_report_writer import ShapReportWriter


def test_write_feature_importance_creates_output_directory(tmp_path) -> None:
    output_path = tmp_path / "nested" / "reports" / "shap.json"

    writer = ShapReportWriter()
    writer.write_feature_importance(
        model_name="xgboost",
        n_samples=100,
        feature_importance=[],
        output_path=output_path,
    )

    assert output_path.exists()


def test_write_feature_importance_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "shap_feature_importance.json"
    feature_importance = [
        ShapFeatureImportance(
            feature_name="Glucose",
            mean_abs_shap_value=0.123,
        ),
        ShapFeatureImportance(
            feature_name="BMI",
            mean_abs_shap_value=0.045,
        ),
    ]

    writer = ShapReportWriter()
    writer.write_feature_importance(
        model_name="xgboost",
        n_samples=100,
        feature_importance=feature_importance,
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload == {
        "model_name": "xgboost",
        "n_samples": 100,
        "features": [
            {
                "feature_name": "Glucose",
                "mean_abs_shap_value": 0.123,
            },
            {
                "feature_name": "BMI",
                "mean_abs_shap_value": 0.045,
            },
        ],
    }
