from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.column_config import NUMERICAL_FEATURES


def build_column_transformer() -> ColumnTransformer:
    """
    Build feature-wise preprocessing transformer.

    Numerical steps:
        1. Impute missing values with median.
        2. Scale features using StandardScaler.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )