from pathlib import Path

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