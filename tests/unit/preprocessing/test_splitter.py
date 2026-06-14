import pandas as pd

from src.preprocessing.splitter import (
    DatasetSplit,
    DataSplitter,
    TrainTestSplit,
)


class TestDataSplitter:
    def test_split_features_target(
    self,
    valid_dataframe: pd.DataFrame,
) -> None:

     splitter = DataSplitter()

     dataset = splitter.split_features_target(
        dataframe=valid_dataframe,
        target_column="Outcome",
    )

     assert isinstance(
        dataset,
        DatasetSplit,
    )

     assert "Outcome" not in dataset.X.columns

     assert len(dataset.X.columns) == (
        len(valid_dataframe.columns) - 1
    )

     assert dataset.y.name == "Outcome"
     
     
     
    def test_feature_target_lengths_match(
     self,
     valid_dataframe: pd.DataFrame,
) -> None:

      splitter = DataSplitter()

      dataset = splitter.split_features_target(
        dataframe=valid_dataframe,
        target_column="Outcome",
     )

      assert len(dataset.X) == len(dataset.y)
      
      
    def test_train_test_split(
    self,
    large_valid_dataframe: pd.DataFrame,
) -> None:

     splitter = DataSplitter()

     dataset = splitter.split_features_target(
        dataframe=large_valid_dataframe,
        target_column="Outcome",
    )

     split = splitter.split_train_test(
        dataset=dataset,
    )

     assert isinstance(
        split,
        TrainTestSplit,
    )

     assert len(split.X_train) > 0
     assert len(split.X_test) > 0

     assert len(split.y_train) > 0
     assert len(split.y_test) > 0
     
    def test_train_test_row_counts(
    self,
    large_valid_dataframe: pd.DataFrame,
) -> None:

     splitter = DataSplitter()

     dataset = splitter.split_features_target(
        dataframe=large_valid_dataframe,
        target_column="Outcome",
    )

     split = splitter.split_train_test(
        dataset=dataset,
        test_size=0.2,
    )

     total_rows = (
          len(split.X_train)
        + len(split.X_test)
    )

     assert total_rows == len(
        large_valid_dataframe
    )
     
     
    def test_class_distribution_preserved(
    self,
    large_valid_dataframe: pd.DataFrame,
) -> None:

     splitter = DataSplitter()

     dataset = splitter.split_features_target(
        dataframe=large_valid_dataframe,
        target_column="Outcome",
    )

     split = splitter.split_train_test(
        dataset=dataset,
    )

     original_ratio = dataset.y.mean()

     train_ratio = split.y_train.mean()

     assert abs(
         original_ratio - train_ratio
    ) < 0.20