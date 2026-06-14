from pathlib import Path

import pandas as pd
import pytest

from src.config.path import FIXTURES_DIR


@pytest.fixture
def valid_csv_path() -> Path:
    return FIXTURES_DIR / "valid_data.csv"


@pytest.fixture
def empty_csv_path() -> Path:
    return FIXTURES_DIR / "empty.csv"


@pytest.fixture
def unsupported_file_path() -> Path:
    return FIXTURES_DIR / "unsupported.txt"


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Pregnancies": [1, 2],
            "Glucose": [100, 120],
            "BloodPressure": [70, 80],
            "SkinThickness": [20, 30],
            "Insulin": [80, 100],
            "BMI": [25.5, 30.1],
            "DiabetesPedigreeFunction": [0.5, 0.8],
            "Age": [25, 35],
            "Outcome": [0, 1],
        }
    )

@pytest.fixture
def large_valid_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "Pregnancies": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Glucose": [100, 120, 130, 140, 110, 150, 160, 170, 180, 190],
            "BloodPressure": [70, 80, 75, 85, 72, 90, 78, 88, 82, 92],
            "SkinThickness": [20, 30, 25, 35, 22, 40, 28, 38, 32, 42],
            "Insulin": [80, 100, 90, 120, 85, 130, 95, 140, 105, 150],
            "BMI": [25.5, 30.1, 28.4, 35.2, 27.3, 38.5, 29.1, 36.4, 31.2, 39.7],
            "DiabetesPedigreeFunction": [
                0.5,
                0.8,
                0.4,
                1.1,
                0.7,
                1.3,
                0.6,
                1.0,
                0.9,
                1.2,
            ],
            "Age": [25, 35, 40, 45, 30, 50, 33, 55, 38, 60],
            "Outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )