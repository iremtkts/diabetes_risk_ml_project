import pytest

from src.data_access.validators import DataValidator
from src.utils.exceptions import DataValidationError


class TestDataValidator:

    def test_valid_dataframe(
        self,
        valid_dataframe,
    ) -> None:

        validated_df = DataValidator.validate(
            valid_dataframe
        )

        assert validated_df is not None
    def test_missing_column(
    self,
    valid_dataframe,
) -> None:

     dataframe = valid_dataframe.drop(
        columns=["Outcome"]
    )

     with pytest.raises(
        DataValidationError
    ):
        DataValidator.validate(
            dataframe
        )
        
        
      
    def test_invalid_dtype(
    self,
    valid_dataframe,
) -> None:

      dataframe = valid_dataframe.copy()

      dataframe["Age"] = ["abc", "xyz"]

      with pytest.raises(
        DataValidationError
    ):
        DataValidator.validate(
            dataframe
        )   
    
    
    
    def test_null_values(
    self,
    valid_dataframe,
) -> None:

     dataframe = valid_dataframe.copy()

     dataframe.loc[0, "BMI"] = None

     with pytest.raises(
        DataValidationError
    ):
        DataValidator.validate(
            dataframe
        )