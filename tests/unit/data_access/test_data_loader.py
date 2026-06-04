from pathlib import Path

import pandas as pd
import pytest

from src.data_access.data_loader import DataLoader
from src.utils.exceptions import (
    EmptyDataError,
    InvalidFilePathError,
    UnsupportedFileExtensionError,
)


class TestDataLoader:

    def test_load_valid_csv(
        self,
        valid_csv_path: Path,
    ) -> None:

        loader = DataLoader()

        dataframe = loader.load_data(
            valid_csv_path
        )

        assert isinstance(
            dataframe,
            pd.DataFrame,
        )

        assert len(dataframe) == 2

    def test_file_not_found(self) -> None:

        loader = DataLoader()

        with pytest.raises(
            InvalidFilePathError
        ):
            loader.load_data(
                Path("missing.csv")
            )

    def test_unsupported_extension(
        self,
        unsupported_file_path: Path,
    ) -> None:

        loader = DataLoader()

        with pytest.raises(
            UnsupportedFileExtensionError
        ):
            loader.load_data(
                unsupported_file_path
            )

    def test_empty_dataframe(
        self,
        empty_csv_path: Path,
    ) -> None:

        loader = DataLoader()

        with pytest.raises(
            EmptyDataError
        ):
            loader.load_data(
                empty_csv_path
            )