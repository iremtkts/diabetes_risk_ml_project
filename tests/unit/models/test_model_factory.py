import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.model_factory import ModelFactory


class TestModelFactory:
    def test_create_logistic_regression_model(self) -> None:
        model = ModelFactory.create_model(
            model_name="logistic_regression",
            max_iter=1000,
        )

        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 1000

    def test_create_random_forest_model(self) -> None:
        model = ModelFactory.create_model(
            model_name="random_forest",
            n_estimators=50,
            random_state=42,
        )

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.random_state == 42

    def test_unsupported_model_raises_error(self) -> None:
        with pytest.raises(ValueError):
            ModelFactory.create_model(
                model_name="unsupported_model",
            )