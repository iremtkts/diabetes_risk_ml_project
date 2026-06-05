from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingResult:
    model: object


class ModelTrainer:
    """
    Responsible for training ML models.
    """

    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> TrainingResult:

        model.fit(
            X_train,
            y_train,
        )

        return TrainingResult(
            model=model,
        )