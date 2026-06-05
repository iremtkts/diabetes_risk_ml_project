import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.training.trainer import ModelTrainer, TrainingResult


class TestModelTrainer:
    def test_train_returns_training_result(self) -> None:
        X_train = pd.DataFrame(
            {
                "Glucose": [100, 120, 140, 160],
                "BMI": [25.5, 30.1, 32.0, 35.2],
            }
        )

        y_train = pd.Series([0, 0, 1, 1])

        model = LogisticRegression()

        trainer = ModelTrainer()

        result = trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        assert isinstance(result, TrainingResult)

    def test_train_returns_fitted_model(self) -> None:
        X_train = pd.DataFrame(
            {
                "Glucose": [100, 120, 140, 160],
                "BMI": [25.5, 30.1, 32.0, 35.2],
            }
        )

        y_train = pd.Series([0, 0, 1, 1])

        model = LogisticRegression()

        trainer = ModelTrainer()

        result = trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        predictions = result.model.predict(X_train)

        assert len(predictions) == len(y_train)