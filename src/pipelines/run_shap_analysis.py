from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from src.config.path import RAW_DATA_DIR, REPORTS_DIR
from src.data_access.data_loader import DataLoader
from src.explainability.shap_analyzer import ShapAnalyzer
from src.explainability.shap_report_writer import ShapReportWriter
from src.inference.artifact_loader import InferenceArtifactLoader
from src.preprocessing.column_config import TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_dataframe(
    processed_features: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    if hasattr(processed_features, "toarray"):
        processed_features = processed_features.toarray()

    return pd.DataFrame(
        np.asarray(processed_features),
        columns=feature_names,
    )


def main() -> None:
    logger.info("Starting offline SHAP analysis")

    data_loader = DataLoader()
    dataframe = data_loader.load_data(RAW_DATA_DIR / "diabetes.csv")

    artifact_loader = InferenceArtifactLoader()
    metadata = artifact_loader.load_metadata()
    model = artifact_loader.load_model(metadata)
    preprocessing_pipeline = artifact_loader.load_preprocessing_pipeline(metadata)

    input_features = cast(list[str], metadata["input_features"])
    processed_feature_names = cast(list[str], metadata["processed_features"])
    model_name = str(metadata.get("model_name", "unknown"))

    raw_features = dataframe.drop(columns=[TARGET_COLUMN])
    raw_features = raw_features[input_features]

    processed_features = preprocessing_pipeline.transform(raw_features)
    processed_features_dataframe = _to_dataframe(
        processed_features=processed_features,
        feature_names=processed_feature_names,
    )

    analyzer = ShapAnalyzer(max_samples=100, random_state=42)
    sampled_features = analyzer.sample_features(processed_features_dataframe)
    shap_values = analyzer.calculate_shap_values(
        model=model,
        features=sampled_features,
    )
    feature_importance = analyzer.calculate_feature_importance(
        shap_values=shap_values,
        feature_names=processed_feature_names,
    )

    output_dir = REPORTS_DIR / "explainability"
    writer = ShapReportWriter()
    feature_importance_path = writer.write_feature_importance(
        model_name=model_name,
        n_samples=len(sampled_features),
        feature_importance=feature_importance,
        output_path=output_dir / "shap_feature_importance.json",
    )
    summary_plot_path = writer.write_summary_plot(
        shap_values=shap_values,
        features=sampled_features,
        output_path=output_dir / "shap_summary.png",
    )

    logger.info(
        "SHAP analysis completed | feature_importance=%s | summary_plot=%s",
        feature_importance_path,
        summary_plot_path,
    )


if __name__ == "__main__":
    main()
