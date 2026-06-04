from dataclasses import dataclass

import pandas as pd

from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    
    
    
class DataSplitter:

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> DatasetSplit:

        X = dataframe.drop(
            columns=[target_column]
        )

        y = dataframe[target_column]

        return DatasetSplit(
            X=X,
            y=y,
        )

    def split_train_test(
        self,
        dataset: DatasetSplit,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainTestSplit:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                dataset.X,
                dataset.y,
                test_size=test_size,
                random_state=random_state,
                stratify=dataset.y,  # Use stratify=dataset.y to maintain identical class distributions across train and test splits, preventing data imbalance issues.
            )
        )

        return TrainTestSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )