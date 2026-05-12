class DiabetesRiskError(Exception):
    """
    Base exception class for the project.
    """

    pass


class DataLoaderError(DiabetesRiskError):
    """
    Raised when data loading fails.
    """

    pass


class InvalidFilePathError(DataLoaderError):
    """
    Raised when file path is invalid.
    """

    pass


class UnsupportedFileExtensionError(DataLoaderError):
    """
    Raised when file extension is unsupported.
    """

    pass


class EmptyDataError(DataLoaderError):
    """
    Raised when loaded data is empty.
    """

    pass


class DataValidationError(DiabetesRiskError):
    """
    Raised when schema validation fails.
    """

    pass