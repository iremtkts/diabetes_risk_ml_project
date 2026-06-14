from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SUPPORTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


class ModelFactory:
    """
    Factory class for creating ML model instances.
    """

    @staticmethod
    def create_model(
        model_name: str,
        **model_params: Any,
    ) -> Any:
        if model_name not in SUPPORTED_MODELS:
            supported_models = list(SUPPORTED_MODELS.keys())

            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {supported_models}"
            )

        model_class = SUPPORTED_MODELS[model_name]

        return model_class(**model_params)
