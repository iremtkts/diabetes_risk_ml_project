import pandas as pd
from pandera.errors import SchemaError

from src.data_access.schemas import validate_dataframe
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Responsible for dataframe validation.
    """

    @staticmethod
    def validate(dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            return validate_dataframe(dataframe)

        except SchemaError as error:
            logger.error(
                "Schema validation failed: %s",
                error,
            )

            raise DataValidationError(
                "Data schema validation failed."
            ) from error
