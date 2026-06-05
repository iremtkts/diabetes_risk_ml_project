from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

import numpy as np
import pandas as pd


class ZeroValueToNaNTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Replace medically impossible zero values with NaN.
    """

    def __init__(
        self,
        columns: list[str],
    ) -> None:

        self.columns = columns

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X.copy()

        X[self.columns] = (
            X[self.columns]
            .replace(0, np.nan)
        )

        return X