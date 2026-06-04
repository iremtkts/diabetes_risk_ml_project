from pathlib import Path

import pytest
import pandas as pd

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
