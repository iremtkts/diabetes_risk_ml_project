from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShapFeatureImportance:
    feature_name: str
    mean_abs_shap_value: float


class ShapAnalyzer:
    def __init__(
        self,
        max_samples: int = 100,
        random_state: int = 42,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")

        self.max_samples = max_samples
        self.random_state = random_state

    def sample_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(features) <= self.max_samples:
            return features

        return features.sample(
            n=self.max_samples,
            random_state=self.random_state,
        )

    def calculate_shap_values(
        self,
        model: Any,
        features: pd.DataFrame,
    ) -> NDArray[np.float64]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        return self._to_2d_array(shap_values)

    @staticmethod
    def calculate_feature_importance(
        shap_values: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[ShapFeatureImportance]:
        if shap_values.ndim != 2:
            raise ValueError("shap_values must be a 2D array")

        if shap_values.shape[1] != len(feature_names):
            raise ValueError(
                "Number of SHAP value columns must match feature_names"
            )

        mean_abs_values = np.abs(shap_values).mean(axis=0)

        feature_importance = [
            ShapFeatureImportance(
                feature_name=feature_name,
                mean_abs_shap_value=float(mean_abs_value),
            )
            for feature_name, mean_abs_value in zip(
                feature_names,
                mean_abs_values,
                strict=True,
            )
        ]

        return sorted(
            feature_importance,
            key=lambda result: result.mean_abs_shap_value,
            reverse=True,
        )

    @staticmethod
    def _to_2d_array(shap_values: Any) -> NDArray[np.float64]:
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values_array = np.asarray(shap_values, dtype=np.float64)

        if shap_values_array.ndim == 3:
            shap_values_array = shap_values_array[:, :, -1]

        if shap_values_array.ndim != 2:
            raise ValueError("Expected SHAP values to be a 2D array")

        return shap_values_array
