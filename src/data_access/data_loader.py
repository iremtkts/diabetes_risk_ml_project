from pathlib import Path

import pandas as pd

from src.data_access.schemas import validate_dataframe

from src.utils.exceptions import (
    DataLoaderError,
    EmptyDataError,
    InvalidFilePathError,
    UnsupportedFileExtensionError,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
}


class DataLoader:
    """
    Production-grade data loader.
    """

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from a supported file.

        Args:
            file_path (Path): Path to data file.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """

        self._validate_file_path(file_path)

        extension = file_path.suffix

        reader_function = self._get_reader(extension)

        try:
            logger.info("Loading data from %s", file_path)

            dataframe = reader_function(file_path)

            self._validate_dataframe(dataframe)

            dataframe = validate_dataframe(dataframe)

            logger.info(
                "Data loaded successfully | shape=%s",
                dataframe.shape,
            )

            return dataframe

        except pd.errors.ParserError as error:
            logger.error("Failed to parse data file: %s", error)

            raise DataLoaderError(
                f"Failed to parse file: {file_path}"
            ) from error

        except Exception as error:
            logger.exception("Unexpected error occurred during data loading")

            raise DataLoaderError(
                "Unexpected error occurred while loading data"
            ) from error

    def _validate_file_path(self, file_path: Path) -> None:
        """
        Validate file path.
        """

        if not isinstance(file_path, Path):
            logger.error("file_path must be a Path object")

            raise InvalidFilePathError(
                "file_path must be a pathlib.Path object"
            )

        if not file_path.exists():
            logger.error("File does not exist: %s", file_path)

            raise InvalidFilePathError(
                f"File does not exist: {file_path}"
            )

    def _get_reader(self, extension: str):
        """
        Get dataframe reader function.
        """

        if extension not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file extension: %s", extension)

            raise UnsupportedFileExtensionError(
                f"Unsupported file extension: {extension}"
            )

        return SUPPORTED_EXTENSIONS[extension]

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        Validate loaded dataframe.
        """

        if dataframe.empty:
            logger.error("Loaded dataframe is empty")

            raise EmptyDataError(
                "Loaded dataframe is empty"
            )