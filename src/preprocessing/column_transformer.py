from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.preprocessing.column_config import (
    NUMERICAL_FEATURES,
)

def build_column_transformer() -> ColumnTransformer:
    """
    Build feature-wise preprocessing transformer.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                SimpleImputer(
                    strategy="median",
                ),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )