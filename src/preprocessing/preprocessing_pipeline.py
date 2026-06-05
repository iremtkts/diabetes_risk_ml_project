from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import ZeroValueToNaNTransformer


ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for diabetes dataset.

    Steps:
        1. Convert medically impossible zero values to NaN.
        2. Impute missing values using median strategy.

    Returns:
        Pipeline: sklearn-compatible preprocessing pipeline.
    """

    return Pipeline(
        steps=[
            (
                "zero_to_nan",
                ZeroValueToNaNTransformer(
                    columns=ZERO_AS_MISSING_COLUMNS,
                ),
            ),
            (
                "median_imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )