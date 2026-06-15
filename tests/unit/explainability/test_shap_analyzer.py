import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_analyzer import ShapAnalyzer


def test_calculate_feature_importance_sorts_by_mean_absolute_shap_value() -> None:
    shap_values = np.array(
        [
            [1.0, -3.0, 0.5],
            [-1.0, 1.0, -0.5],
        ]
    )
    feature_names = ["feature_a", "feature_b", "feature_c"]

    feature_importance = ShapAnalyzer.calculate_feature_importance(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    assert [result.feature_name for result in feature_importance] == [
        "feature_b",
        "feature_a",
        "feature_c",
    ]
    assert feature_importance[0].mean_abs_shap_value == 2.0


def test_calculate_feature_importance_rejects_mismatched_feature_names() -> None:
    shap_values = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="must match feature_names"):
        ShapAnalyzer.calculate_feature_importance(
            shap_values=shap_values,
            feature_names=["feature_a"],
        )


def test_sample_features_uses_max_samples_and_random_state() -> None:
    features = pd.DataFrame({"feature": range(200)})
    analyzer = ShapAnalyzer(max_samples=100, random_state=42)

    sampled_features = analyzer.sample_features(features)

    assert len(sampled_features) == 100
    assert sampled_features.equals(
        features.sample(n=100, random_state=42)
    )


def test_sample_features_keeps_small_dataset_unchanged() -> None:
    features = pd.DataFrame({"feature": range(10)})
    analyzer = ShapAnalyzer(max_samples=100, random_state=42)

    sampled_features = analyzer.sample_features(features)

    assert sampled_features is features
