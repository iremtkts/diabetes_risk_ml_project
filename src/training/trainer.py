from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class TrainingResult:
    model: Any


class ModelTrainer:
    """
    Responsible for training ML models.
    """

    def train(
        self,
        model: Any,
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
