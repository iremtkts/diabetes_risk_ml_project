from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


class ZeroValueToNaNTransformer(
    BaseEstimator,  # type: ignore[misc]
    TransformerMixin,  # type: ignore[misc]
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
        y: Any = None,
    ) -> Self:

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
