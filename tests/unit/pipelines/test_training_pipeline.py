import pytest

from src.pipelines.training_pipeline import (
    TrainingPipeline,
    TrainingPipelineResult,
)


@pytest.fixture
def training_data_path(tmp_path, large_valid_dataframe):
    data_path = tmp_path / "diabetes.csv"
    large_valid_dataframe.to_csv(data_path, index=False)

    return data_path


def test_training_pipeline_runs_successfully(training_data_path):
    pipeline = TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    assert isinstance(result, TrainingPipelineResult)


def test_training_pipeline_returns_fitted_model(training_data_path):
    pipeline = TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    predictions = result.model.predict(result.X_test_processed)

    assert len(predictions) == len(result.y_test)


def test_training_pipeline_returns_preprocessing_pipeline(training_data_path):
    pipeline = TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    assert result.preprocessing_pipeline is not None


def test_training_pipeline_returns_train_and_test_data(training_data_path):
    pipeline = TrainingPipeline(
        data_path=training_data_path,
        model_name="logistic_regression",
        model_params={"max_iter": 1000},
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    assert result.X_train_processed is not None
    assert result.X_test_processed is not None
    assert len(result.y_train) > 0
    assert len(result.y_test) > 0