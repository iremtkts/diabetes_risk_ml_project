from sklearn.compose import ColumnTransformer

from src.preprocessing.column_transformer import (
    build_column_transformer,
)



class TestColumnTransformer:

    def test_returns_column_transformer(
        self,
    ) -> None:

        transformer = (
            build_column_transformer()
        )

        assert isinstance(
            transformer,
            ColumnTransformer,
        )