from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.preprocessing.column_config import ZERO_AS_MISSING_COLUMNS
from src.preprocessing.column_transformer import build_column_transformer
from src.preprocessing.transformers import ZeroValueToNaNTransformer


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "zero_to_nan",
                ZeroValueToNaNTransformer(columns=ZERO_AS_MISSING_COLUMNS),
            ),
            ("column_transformer", build_column_transformer()),
        ]
    )