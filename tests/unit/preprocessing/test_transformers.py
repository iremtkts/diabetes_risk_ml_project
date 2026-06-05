import numpy as np
import pandas as pd

from src.preprocessing.transformers import ZeroValueToNaNTransformer


class TestZeroValueToNaNTransformer:
    def test_replaces_zero_values_with_nan(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Glucose": [100, 0, 120],
                "BloodPressure": [70, 80, 0],
                "BMI": [25.5, 0.0, 30.1],
                "Age": [25, 35, 45],
            }
        )

        transformer = ZeroValueToNaNTransformer(
            columns=["Glucose", "BloodPressure", "BMI"]
        )

        transformed = transformer.fit_transform(dataframe)

        assert np.isnan(transformed.loc[1, "Glucose"])
        assert np.isnan(transformed.loc[2, "BloodPressure"])
        assert np.isnan(transformed.loc[1, "BMI"])

    def test_does_not_modify_original_dataframe(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Glucose": [100, 0, 120],
                "BMI": [25.5, 0.0, 30.1],
            }
        )

        original_dataframe = dataframe.copy(deep=True)

        transformer = ZeroValueToNaNTransformer(
            columns=["Glucose", "BMI"]
        )

        _ = transformer.fit_transform(dataframe)

        pd.testing.assert_frame_equal(
            dataframe,
            original_dataframe,
        )

    def test_columns_not_selected_are_unchanged(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Glucose": [100, 0, 120],
                "Age": [25, 0, 45],
            }
        )

        transformer = ZeroValueToNaNTransformer(
            columns=["Glucose"]
        )

        transformed = transformer.fit_transform(dataframe)

        assert np.isnan(transformed.loc[1, "Glucose"])
        assert transformed.loc[1, "Age"] == 0