import pandera.pandas as pa
from pandera.typing import DataFrame


class DiabetesDataSchema(pa.DataFrameModel):
    """
    Schema for diabetes dataset validation.
    """

    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    Outcome: int


def validate_dataframe(
    dataframe: DataFrame[DiabetesDataSchema],
) -> DataFrame[DiabetesDataSchema]:
    """
    Validate dataframe against schema.

    Args:
        dataframe: Input dataframe.

    Returns:
        Validated dataframe.
    """

    return DiabetesDataSchema.validate(dataframe)