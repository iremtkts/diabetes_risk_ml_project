import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


class TestPreprocessingPipeline:
    def test_build_preprocessing_pipeline_returns_pipeline(self) -> None:
        pipeline = build_preprocessing_pipeline()

        assert isinstance(pipeline, Pipeline)

    def test_pipeline_transforms_zero_values_and_imputes_missing_values(
        self,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "Pregnancies": [1, 2, 3],
                "Glucose": [100, 0, 120],
                "BloodPressure": [70, 80, 0],
                "SkinThickness": [20, 0, 30],
                "Insulin": [80, 0, 100],
                "BMI": [25.5, 0.0, 30.1],
                "DiabetesPedigreeFunction": [0.5, 0.8, 0.4],
                "Age": [25, 35, 45],
            }
        )

        pipeline = build_preprocessing_pipeline()

        transformed = pipeline.fit_transform(dataframe)

        assert not np.isnan(transformed).any()

    def test_pipeline_preserves_row_count(
        self,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "Pregnancies": [1, 2, 3],
                "Glucose": [100, 0, 120],
                "BloodPressure": [70, 80, 0],
                "SkinThickness": [20, 0, 30],
                "Insulin": [80, 0, 100],
                "BMI": [25.5, 0.0, 30.1],
                "DiabetesPedigreeFunction": [0.5, 0.8, 0.4],
                "Age": [25, 35, 45],
            }
        )

        pipeline = build_preprocessing_pipeline()

        transformed = pipeline.fit_transform(dataframe)

        assert transformed.shape[0] == dataframe.shape[0]